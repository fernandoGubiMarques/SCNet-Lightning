from scnet.wav import Wavset
import lightning as L
from pathlib import Path
import json
from torch.utils.data import DataLoader, Subset


class WavModule(L.LightningDataModule):

    def __init__(
        self,
        partition_path: str,
        stems: list[str],
        segment: float,
        stride: float,
    ):

        super().__init__()

        self.train_set = Wavset(
            partition_path,
            "train",
            stems,
            segment=segment,
            stride=stride,
        )

        self.val_set = Wavset(
            partition_path,
            "val",
            stems,
            segment=None,
            stride=stride,
        )

        self.test_set = Wavset(
            partition_path,
            "test",
            stems,
            segment=None,
            stride=stride,
        )

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the training set.
        """
        return DataLoader(self.train_set, 8, True, 15)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the validation set.
        """
        return DataLoader(self.val_set, 1, False, 15)

    def test_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the validation set.
        """
        return DataLoader(self.test_set, 1, False, 15)
