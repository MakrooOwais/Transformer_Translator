import torch
import pytorch_lightning as pl
import torch.nn as nn

from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from config import get_config
from model_parts import build_transformer


class TranslationModel(pl.LightningModule):
    """
    LightningModule implementation for a Transformer-based translation model.

    Attributes:
        config (dict): Configuration dictionary containing model hyperparameters.
        tokenizer_src: Source language tokenizer object.
        tokenizer_tgt: Target language tokenizer object.
        model (Transformer): Transformer model instance for translation.
        search_algo: Search algorithm used during validation.
        loss_fn: Loss function for training, CrossEntropyLoss with label smoothing.
        bleu_score: BLEU score metric for evaluation.
        word_error_rate: Word Error Rate metric for evaluation.
        char_error_rate: Character Error Rate metric for evaluation.
        source_texts (list): List of source texts during validation.
        expected (list): List of expected target texts during validation.
        predicted (list): List of predicted target texts during validation.

    Args:
        tokenizer_src: Source language tokenizer object.
        tokenizer_tgt: Target language tokenizer object.
        search_algo: Search algorithm used during validation.
    """

    def __init__(self, tokenizer_src, tokenizer_tgt, search_algo) -> None:
        super().__init__()
        self.config = get_config()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.model = build_transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            self.config["seq_len"],
            self.config["seq_len"],
            d_model=self.config["d_model"],
        )

        self.search_algo = search_algo

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )

        self.bleu_score = BLEUScore()
        self.word_error_rate = WordErrorRate()
        self.char_error_rate = CharErrorRate()

        self.source_texts = []
        self.expected = []
        self.predicted = []

    def training_step(self, batch, batch_idx):
        """
        Training step for the LightningModule.

        Args:
            batch (dict): Dictionary containing batch tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary with the training loss.
        """
        encoder_input = batch["encoder_input"]  # (b, seq_len)
        decoder_input = batch["decoder_input"]  # (B, seq_len)
        encoder_mask = batch["encoder_mask"]  # (B, 1, 1, seq_len)
        decoder_mask = batch["decoder_mask"]  # (B, 1, seq_len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(
            encoder_input, encoder_mask
        )  # (B, seq_len, d_model)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )  # (B, seq_len, d_model)
        proj_output = self.model.project(decoder_output)  # (B, seq_len, vocab_size)

        # Compare the output with the label
        label = batch["label"]  # (B, seq_len)

        # Compute the loss using a simple cross entropy
        loss = self.loss_fn(
            proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )

        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=True, prog_bar=True)

        return {"loss": loss}

    def on_validation_epoch_start(self):
        """
        Preparation steps at the start of each validation epoch.
        """
        self.source_texts.append([])
        self.expected.append([])
        self.predicted.append([])

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the LightningModule.

        Args:
            batch (dict): Dictionary containing batch tensors.
            batch_idx (int): Index of the current batch.
        """
        encoder_input = batch["encoder_input"]  # (b, seq_len)
        encoder_mask = batch["encoder_mask"]  # (b, 1, 1, seq_len)

        model_out = self.search_algo(
            self.model,
            encoder_input,
            encoder_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["seq_len"],
            self.config["device"],
        )

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        self.source_texts[-1].append(source_text)
        self.expected[-1].append(target_text)
        self.predicted[-1].append(model_out_text)

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of each validation epoch.
        """
        self.log_dict(
            {
                "bleu": self.bleu_score(self.predicted[-1], self.expected[-1]),
                "wer": self.word_error_rate(self.predicted[-1], self.expected[-1]),
                "cer": self.char_error_rate(self.predicted[-1], self.expected[-1]),
            },
            on_epoch=True,
            prog_bar=True,
        )

    def forward(self, batch):
        """
        Performs a forward pass through the model for a given batch of input data.

        Parameters:
        - batch (dict): A dictionary containing the input data for the model.
            - "encoder_input" (Tensor): Tensor containing the input sequences for the encoder. Shape: (batch_size, seq_len)
            - "encoder_mask" (Tensor): Tensor containing the mask for the encoder input. Shape: (batch_size, 1, 1, seq_len)

        Returns:
        - str: The decoded output sequence from the model as a string.
        """
        encoder_input = batch["encoder_input"]  # (b, seq_len)
        encoder_mask = batch["encoder_mask"]  # (b, 1, 1, seq_len)

        # Perform the decoding using the search algorithm (e.g., greedy decoding or beam search)
        model_out = self.search_algo(
            self.model,
            encoder_input,
            encoder_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["seq_len"],
            self.config["device"],
        )
        # Decode the output tensor to a string
        return self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    def configure_optimizers(self):
        """
        Configure optimizer for training.

        Returns:
            list: List containing the optimizer.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], eps=1e-9
        )
        return [optimizer]
