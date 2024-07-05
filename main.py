from pytorch_lightning import Trainer

from data_module import DataModule
from dataset import get_ds
from model import TranslationModel
from config import get_config
from search_algo import greedy_decode

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Load configuration
    config = get_config()

    # Load and prepare training and validation datasets
    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Initialize the Lightning DataModule
    data_module = DataModule(train_ds, val_ds)
    data_module.setup("validate")

    # Initialize the TranslationModel
    model = TranslationModel(tokenizer_src, tokenizer_tgt, greedy_decode)

    # Initialize the PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator="gpu",  # Use GPU for training
        max_epochs=config["num_epochs"],  # Maximum number of epochs
        devices=[0],  # GPU device index to use
        log_every_n_steps=1,  # Log every training step
        enable_checkpointing=True,  # Enable model checkpointing
    )

    # Train the model using the Trainer and DataModule
    trainer.fit(model, datamodule=data_module)
