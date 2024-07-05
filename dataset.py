import torch

from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


class BilingualDataset(Dataset):
    """
    Dataset class for training a Transformer-based translation model.

    Attributes:
        ds: Original dataset containing translation pairs.
        tokenizer_src (PreTrainedTokenizer): Tokenizer for source language.
        tokenizer_tgt (PreTrainedTokenizer): Tokenizer for target language.
        src_lang (str): Source language identifier.
        tgt_lang (str): Target language identifier.
        seq_len (int): Maximum sequence length for input and output sequences.
        sos_token (torch.Tensor): Start-of-sequence token for target language.
        eos_token (torch.Tensor): End-of-sequence token for target language.
        pad_token (torch.Tensor): Padding token for target language.

    Args:
        ds: Original dataset containing translation pairs.
        tokenizer_src (PreTrainedTokenizer): Tokenizer for source language.
        tokenizer_tgt (PreTrainedTokenizer): Tokenizer for target language.
        src_lang (str): Source language identifier.
        tgt_lang (str): Target language identifier.
        seq_len (int): Maximum sequence length for input and output sequences.
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Retrieves a single training example from the dataset.

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            dict: Dictionary containing the encoder input, decoder input, masks, label, and source/target texts.
        """
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    """
    Generate a causal mask matrix for self-attention.

    Args:
        size (int): Size of the mask matrix (seq_len).

    Returns:
        torch.Tensor: Causal mask tensor of shape (1, size, size).
            The lower triangular part (including the diagonal) is filled with 1s,
            and the upper triangular part is filled with 0s.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_all_sentences(ds, lang):
    """
    Generator function to yield sentences in a specified language from a dataset.

    Args:
        ds (Iterable): Dataset containing items with "translation" field.
        lang (str): Language code indicating which language to retrieve ("src" or "tgt").

    Yields:
        str: Sentences in the specified language from the dataset.
    """
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieves an existing tokenizer or builds and saves a new tokenizer if it doesn't exist.

    Args:
        config (dict): Configuration containing paths and settings.
        ds (Iterable): Dataset containing items with "translation" field.
        lang (str): Language code indicating which language to retrieve ("src" or "tgt").

    Returns:
        Tokenizer: Instance of the tokenizer for the specified language.
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """
    Loads and preprocesses a bilingual dataset for training and validation.

    Args:
        config (dict): Configuration containing dataset source, languages, tokenizer settings, etc.

    Returns:
        tuple: Contains train dataset, validation dataset, source tokenizer, and target tokenizer.
    """
    ds_raw = load_dataset(f"{config['datasource']}", split="train")

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    return train_ds, val_ds, tokenizer_src, tokenizer_tgt
