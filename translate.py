from pathlib import Path
from config import get_config
from model import TranslationModel
from search_algo import greedy_decode
from tokenizers import Tokenizer
import torch
import sys


def translate(sentence: str):
    """
    Translates a given sentence from the source language to the target language using a pretrained model.

    Parameters:
    - sentence (str): The sentence to translate.

    Returns:
    - str: The translated sentence.
    """
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = get_config()

    tokenizer_src = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_src"])))
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_tgt"])))
    )

    model = TranslationModel(tokenizer_src, tokenizer_tgt, greedy_decode)

    # Load the pretrained weights
    model_filename = config["pretrained_weights"]
    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])
    model.model.to(config["device"])

    seq_len = config["seq_len"]

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Encode the source sentence and add special tokens (SOS, EOS, PAD)
        source = tokenizer_src.encode(sentence)
        source = torch.cat(
            [
                torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer_src.token_to_id("[PAD]")]
                    * (seq_len - len(source.ids) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(device)
        # Create a mask for the source sentence (to ignore padding tokens)
        source_mask = (
            (source != tokenizer_src.token_to_id("[PAD]"))
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            .to(device)
        )

        # Print the source sentence and start the translation process
        print(f"{f'SOURCE: ':>12}{sentence}")
        print(f"{f'PREDICTED: ':>12}", end="")

        # Use the model to translate the sentence
        decoder_input = model(
            {"encoder_input": source.cuda(), "encoder_mask": source_mask.cuda()}
        )
        print(decoder_input)

    # Return the translated sentence
    return decoder_input


if __name__ == "__main__":
    # read sentence from argument
    translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student.")
