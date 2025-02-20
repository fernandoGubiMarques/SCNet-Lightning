from scnet.wav import build_metadata, Wavset
import lightning as L
from pathlib import Path
import json
from torch.utils.data import DataLoader, Subset


class WavMixtureModule(L.LightningDataModule):
    """
    A LightningDataModule for handling WAV audio datasets.

    This module prepares training and validation datasets, builds metadata if necessary,
    and provides PyTorch DataLoaders for training and validation.

    Attributes:
        train_set (Wavset): The training dataset.
        val_set (Wavset): The validation dataset.
        loader_config (dict): Configuration dictionary for DataLoader settings.
    """

    def __init__(self, root):
        """
        Initializes the WavModule.

        Args:
            data_config (dict): Configuration dictionary containing dataset paths and sources.
            loader_config (dict): Configuration dictionary for DataLoader settings.
        """
        super().__init__()

        # Define dataset paths
        data_root = Path(root)
        metadata_file = data_root / "metadata.json"
        train_path = data_root / "train"

        with open(metadata_file) as f:
            train_metadata, _, _ = json.load(f)

        # Initialize datasets
        self.train_set = Wavset(train_path, train_metadata, ["mixture"], 11, 1, True)
        

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the training set.
        """
        return DataLoader(self.train_set, 8, True, num_workers=15)

    # def val_dataloader(self):
    #     """
    #     Returns the DataLoader for the validation dataset.

    #     Returns:
    #         DataLoader: A PyTorch DataLoader for the validation set.
    #     """
    #     return DataLoader(self.val_set, **(self.loader_config["val"]))
    
    # def test_dataloader(self):
    #     """
    #     Returns the DataLoader for the validation dataset.

    #     Returns:
    #         DataLoader: A PyTorch DataLoader for the validation set.
    #     """
    #     return DataLoader(self.test_set, **(self.loader_config["val"]))
