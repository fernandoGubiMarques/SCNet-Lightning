from scnet.wav import Wavset
import lightning as L
from pathlib import Path
import json
import torchaudio
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
from torchaudio.sox_effects import apply_effects_tensor


class RandomPitchShift(torch.nn.Module):

    def __init__(
        self,
        activation_prob: float = 0.5,
        steps_range: tuple[float] = (-4, 4),
    ):
        super().__init__()
        self.activation_prob = activation_prob
        self.steps_range = steps_range
        self.sample_rate = 44_100

    def forward(self, x: torch.Tensor):
        # Return original waveform if effect is not applied
        if random.random() > self.activation_prob:
            return x
        
        with torch.no_grad():
            # Reshape and store original shape
            original_shape = x.shape
            x = x.view(-1, x.shape[-2], x.shape[-1])

            # Randomly choose a number of semitones to shift within the given range
            shift_steps = random.randint(*self.steps_range)

            # Define the effect
            effects = [["pitch", str(shift_steps * 100)], ["rate", str(self.sample_rate)]]

            # Compute outputs
            outputs = []
            original_length = x.shape[-1]

            for audio in x:
                new_wav, _ = apply_effects_tensor(audio.cpu(), self.sample_rate, effects)

                # Ensure the output has the same length as the input
                new_length = new_wav.shape[-1]
                if new_length < original_length:
                    # Pad with zeros if too short
                    pad_amount = original_length - new_length
                    new_wav = torch.nn.functional.pad(new_wav, (0, pad_amount))
                elif new_length > original_length:
                    # Truncate if too long
                    new_wav = new_wav[:, :original_length]

                outputs.append(new_wav)

            outputs = torch.stack(outputs).view(original_shape).to(x.device)
            return outputs


class RandomAddNoise(nn.Module):

    def __init__(
        self,
        activation_prob = 0.2,
        snr_range = (0.2, 20)
    ):
        super().__init__()
        self.activation_prob = activation_prob
        self.snr_range = snr_range
        self.sample_rate = 44100
    
    def forward(self, x: torch.Tensor):
        if random.random() > self.activation_prob:
            return x
        
        with torch.no_grad():
            snr = random.uniform(*self.snr_range) * torch.ones(x.shape[:-1], device=x.device)
            noise = torch.rand_like(x, device=x.device)
            return torchaudio.functional.add_noise(x, noise, snr)


class RandomBandPass(nn.Module):

    def __init__(
        self,
        activation_prob: float = 0.2,
        freq_range: tuple[int] = (1, 4000)
    ):
        super().__init__()
        self.activation_prob = activation_prob
        self.freq_range = freq_range
        self.sample_rate = 44100

    def forward(self, x: torch.Tensor):
        if random.random() > self.activation_prob:
            return x
        
        with torch.no_grad():
            # Reshape and store original shape
            original_shape = x.shape
            x = x.view(-1, x.shape[-2], x.shape[-1])

            # Randomly choose frequency
            freq = random.randint(*self.freq_range)
            width = max(freq // 4, 100)

            # Define the effect
            effects = [["bandpass", str(freq), str(width)], ["rate", str(self.sample_rate)]]

            # Compute outputs
            outputs = []
            original_length = x.shape[-1]
            
            for audio in x:
                new_wav, _ = apply_effects_tensor(audio.cpu(), self.sample_rate, effects)

                # Ensure the output has the same length as the input
                new_length = new_wav.shape[-1]
                if new_length < original_length:
                    # Pad with zeros if too short
                    pad_amount = original_length - new_length
                    new_wav = torch.nn.functional.pad(new_wav, (0, pad_amount))
                elif new_length > original_length:
                    # Truncate if too long
                    new_wav = new_wav[:, :original_length]

                outputs.append(new_wav)

            outputs = torch.stack(outputs).view(original_shape).to(x.device)
            return outputs


class RandomChorus(nn.Module):

    def __init__(
        self,
        activation_prob: float = 0.2
    ):
        super().__init__()
        self.activation_prob = activation_prob
        self.sample_rate = 44100

    def forward(self, x: torch.Tensor):
        if random.random() > self.activation_prob:
            return x
        
        with torch.no_grad():
            # Reshape and store original shape
            original_shape = x.shape
            x = x.view(-1, x.shape[-2], x.shape[-1])

            # Define the effect
            effects = ["chorus 0.5 0.9 50 0.4 0.25 2 -t 60 0.32 0.4 2.3 -t 40 0.3 0.3 1.3 -s".split(), ["rate", str(self.sample_rate)]]

            # Compute outputs
            outputs = []
            original_length = x.shape[-1]
            
            for audio in x:
                new_wav, _ = apply_effects_tensor(audio.cpu(), self.sample_rate, effects)

                # Ensure the output has the same length as the input
                new_length = new_wav.shape[-1]
                if new_length < original_length:
                    # Pad with zeros if too short
                    pad_amount = original_length - new_length
                    new_wav = torch.nn.functional.pad(new_wav, (0, pad_amount))
                elif new_length > original_length:
                    # Truncate if too long
                    new_wav = new_wav[:, :original_length]

                outputs.append(new_wav)

            outputs = torch.stack(outputs).view(original_shape).to(x.device)
            return outputs


class RandomPhaser(nn.Module):
    
    def __init__(
        self,
        activation_prob: float = 0.2
    ):
        super().__init__()
        self.activation_prob = activation_prob
        self.sample_rate = 44100

    def forward(self, x: torch.Tensor):
        if random.random() > self.activation_prob:
            return x
        
        with torch.no_grad():
            # Reshape and store original shape
            original_shape = x.shape
            x = x.view(-1, x.shape[-2], x.shape[-1])

            # Define the effect
            effects = ["phaser 0.8 0.74 3 0.4 0.5 -s".split(), ["rate", str(self.sample_rate)]]

            # Compute outputs
            outputs = []
            original_length = x.shape[-1]
            
            for audio in x:
                new_wav, _ = apply_effects_tensor(audio.cpu(), self.sample_rate, effects)

                # Ensure the output has the same length as the input
                new_length = new_wav.shape[-1]
                if new_length < original_length:
                    # Pad with zeros if too short
                    pad_amount = original_length - new_length
                    new_wav = torch.nn.functional.pad(new_wav, (0, pad_amount))
                elif new_length > original_length:
                    # Truncate if too long
                    new_wav = new_wav[:, :original_length]

                outputs.append(new_wav)

            outputs = torch.stack(outputs).view(original_shape).to(x.device)
            return outputs



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
        return DataLoader(self.train_set, batch_size=4, shuffle=True, num_workers=29)

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
