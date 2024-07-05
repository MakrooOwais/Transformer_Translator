import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset

from config import get_config



class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
    ):
        """
        Initializes a PyTorch Lightning DataModule for handling training and validation data.

        Args:
            train_ds (Dataset): Training dataset.
            val_ds (Dataset): Validation dataset.
        """
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.config = get_config()

    def prepare_data(self):
        """
        Optional method for data preparation.

        Typically used for downloading or preprocessing data that might affect training.
        Since the datasets (`train_ds` and `val_ds`) are already prepared in `get_ds`,
        this method is left empty (`pass`).
        """
        pass

    def setup(self, stage: str):
        """
        Sets up data loaders for training or validation stages.

        Args:
            stage (str): Current stage ('fit' for training, 'validate' for validation).
        """
        if stage == "fit":
            self.train = DataLoader(
                self.train_ds,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["num_workers"],
                pin_memory=self.config["pin_memory"],
                persistent_workers=self.config["num_workers"] != 0,
            )

        if stage == "validate":
            self.val = DataLoader(
                self.val_ds,
                batch_size=1,
                shuffle=True,
                num_workers=self.config["num_workers"],
                pin_memory=self.config["pin_memory"],
                persistent_workers=self.config["num_workers"] != 0,
            )

    def train_dataloader(self):
        """
        Returns the DataLoader for training.

        Returns:
            DataLoader: Training DataLoader.
        """
        return self.train

    def val_dataloader(self):
        """
        Returns the DataLoader for validation.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return self.val

    def predict_dataloader(self):
        """
        Placeholder method for setting up a DataLoader for prediction (inference).

        This method is not implemented in the current version.
        """
        pass
