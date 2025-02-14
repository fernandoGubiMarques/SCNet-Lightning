from scnet.wav import build_metadata, Wavset
import lightning as L
from pathlib import Path
import json
from torch.utils.data import DataLoader, Subset


class WavModule(L.LightningDataModule):
    """
    A LightningDataModule for handling WAV audio datasets.

    This module prepares training and validation datasets, builds metadata if necessary,
    and provides PyTorch DataLoaders for training and validation.

    Attributes:
        train_set (Wavset): The training dataset.
        val_set (Wavset): The validation dataset.
        loader_config (dict): Configuration dictionary for DataLoader settings.
    """

    def __init__(self, data_config: dict, loader_config: dict):
        """
        Initializes the WavModule.

        Args:
            data_config (dict): Configuration dictionary containing dataset paths and sources.
            loader_config (dict): Configuration dictionary for DataLoader settings.
        """
        super().__init__()

        # Define dataset paths
        data_root = Path(data_config["root"])
        sources = data_config["sources"]
        metadata_file = data_root / "metadata.json"
        train_path = data_root / "train"
        val_path = data_root / "valid"
        test_path = data_root / "test"

        # Build metadata if it does not exist
        if not metadata_file.exists():
            train_metadata = build_metadata(str(train_path), sources)
            val_metadata = build_metadata(str(val_path), sources)
            test_metadata = build_metadata(str(test_path), sources)

            # Save metadata to a JSON file
            with open(metadata_file, "w") as f:
                json.dump([train_metadata, val_metadata, test_metadata], f)
        else:
            # Load existing metadata
            with open(metadata_file) as f:
                train_metadata, val_metadata, test_metadata = json.load(f)

        # Prepare dataset configurations
        train_config = data_config.copy()
        train_config["root"] = train_path
        train_config["metadata"] = train_metadata

        kw_cv = {}

        # Initialize datasets
        self.train_set = Wavset(**train_config)
        self.val_set = Wavset(
            val_path,
            val_metadata,
            ["mixture"] + data_config["sources"],
            samplerate=data_config["samplerate"],
            channels=data_config["channels"],
            normalize=False,
            **kw_cv,
        )

        self.test_set = Wavset(
            test_path,
            test_metadata,
            ["mixture"] + data_config["sources"],
            samplerate=data_config["samplerate"],
            channels=data_config["channels"],
            normalize=False,
            **kw_cv,
        )

        # Store DataLoader configuration
        self.loader_config = loader_config

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the training set.
        """
        return DataLoader(self.train_set, **(self.loader_config["train"]))

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the validation set.
        """
        return DataLoader(self.val_set, **(self.loader_config["val"]))
    
    def test_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the validation set.
        """
        return DataLoader(self.test_set, **(self.loader_config["val"]))
