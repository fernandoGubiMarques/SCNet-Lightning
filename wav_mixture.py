from scnet.wav import Wavset
import lightning as L
from pathlib import Path
import json
import torchaudio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class RandomPitchShift(nn.Module):

    def __init__(
        self,
        activation_prob=0.5,
        sr=44100,
        min_steps=-5,
        max_steps=5,
        bins_per_octave=48,
        n_fft=2**12,
    ):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.sr = sr
        self.bins_per_octave = bins_per_octave
        self.n_fft = n_fft
        self.activation_prob = activation_prob

    def forward(self, x):
        die = torch.rand((1,)).item()
        if die >= self.activation_prob:
            return x
        with torch.no_grad():
            steps = torch.randint(self.min_steps, self.max_steps, (1,)).item()
            x = torchaudio.functional.pitch_shift(
                x, self.sr, steps, self.bins_per_octave, self.n_fft
            )
        return x


class WavTransform(Dataset):

    def __init__(self, path, metadata):
        self.wavset = Wavset(path, metadata, ["mixture"], 11, 1, True)

        self.transform = RandomPitchShift()

    def __len__(self):
        return len(self.wavset)

    def __getitem__(self, index):
        audio = self.wavset[index]
        return self.transform(audio), self.transform(audio)


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
        self.train_set = WavTransform(train_path, train_metadata)

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
