import torch
from torch import Tensor
from lightning import LightningModule
from scnet.utils import new_sdr
from scnet import augment
from scnet.loss import spec_rmse_loss
from scnet.SCNet import SCNet
from omegaconf import DictConfig


class SCNetLightning(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        optim_config: DictConfig,
        data_config: DictConfig,
        augment_config: DictConfig,
    ):
        super().__init__()
        self.model = SCNet(**model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optim_config)

        # Compute the window size in samples
        self.window_size = data_config["samplerate"] * data_config["segment"]
        self.model_config = model_config

        # STFT configuration for spectral loss computation
        self.stft_config = {
            "n_fft": model_config["nfft"],
            "hop_length": model_config["hop_size"],
            "win_length": model_config["win_size"],
            "center": True,
            "normalized": model_config["normalized"],
            # "window": torch.hamming_window(model_config["win_size"]).cuda()
        }

        # Data augmentations
        augments = [
            augment.Shift(
                shift=int(data_config["samplerate"] * data_config["shift"]),
                same=augment_config["shift_same"],
            )
        ]
        if augment_config["flip"]:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ["scale", "remix"]:
            kw = augment_config[aug]
            if kw["proba"]:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        self.inference_overlap = 0.3

    def forward(self, mix):
        """Forward pass through the SCNet model."""
        return self.model(mix)

    def configure_optimizers(self):
        """Returns the optimizer for training."""
        return self.optimizer


    def training_step(self, batch, batch_idx):
        """Single training step where the model learns from a batch."""
        batch, _, _ = batch
        sources = batch.to(self.device)
        sources = self.augment(sources)  # Apply augmentations
        mix = sources.sum(dim=1)  # Create mixture by summing sources

        estimate = self(mix)  # Forward pass
        assert estimate.shape == sources.shape, f"{estimate.shape=}, {sources.shape=}"

        loss = spec_rmse_loss(estimate, sources, self.stft_config)  # Compute loss
        self.log("train_loss", loss, prog_bar=True, logger=True)  # Log loss

        return loss

    def apply_model(self, x: Tensor, overlap: float, mean: Tensor, std: Tensor):
        """Applies the model in a windowed fashion with overlap handling."""
        B, C, T = x.shape  # (Batch, Channels, Time)
        assert B == 1, "Batch size should always be 1 during validation."

        x = x.squeeze(0)  # Remove batch dimension -> (2, T)

        # Pad input so that T is a multiple of window_size
        pad_length = (self.window_size - (T % self.window_size)) % self.window_size
        x_padded = torch.nn.functional.pad(x, (0, pad_length))  # (2, T + pad_length)
        T_padded = x_padded.shape[-1]

        # Compute step size based on overlap ratio
        step_size = int(self.window_size * (1 - overlap))
        window_starts = list(range(0, T_padded - self.window_size + 1, step_size))

        # Stack windows into a single tensor (N, 2, window_size)
        x_windows = torch.stack(
            [x_padded[:, start : start + self.window_size] for start in window_starts],
            dim=0,
        )

        # Forward pass in parallel for all windows
        y_windows = self.forward(x_windows)  # (N, 4, 2, window_size)

        # Prepare output tensor (4, 2, T_padded) with zeros for accumulation
        output = torch.zeros((4, 2, T_padded), device=x.device)
        weight = torch.zeros(
            (4, 2, T_padded), device=x.device
        )  # Used for blending overlapping regions

        # Accumulate results and weights
        window_weights = torch.hamming_window(self.window_size, device=output.device)
        for i, start in enumerate(window_starts):
            end = start + self.window_size
            output[:, :, start:end] += window_weights * y_windows[i]
            weight[:, :, start:end] += window_weights  # Track coverage count per sample

        # Normalize overlapping regions
        output /= torch.where(weight > 0, weight, torch.tensor(1.0, device=x.device))

        # Remove padding to restore original shape
        output = output[:, :, :T]

        # Add back batch dimension -> (1, 4, 2, T)
        output = output.unsqueeze(0)

        return output * std + mean

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx):
        """Validation step: evaluates model performance using NSDR metric."""
        batch, mean, std = batch
        mean = mean.half()
        std = mean.half()
        sources = batch.to(self.device)
        mix = sources[:, 0]  # First channel is the mixture
        sources = sources[:, 1:]  # Remaining channels are sources

        # Apply model with overlap
        mix = (mix - mean) / std
        estimate = self.apply_model(mix, self.inference_overlap, mean, std)

        # Compute NSDR for each source
        nsdrs = new_sdr(sources, estimate.detach()).mean(0)
        nsdrs = self.trainer.strategy.reduce(nsdrs, reduce_op="mean")

        # Log NSDR per source and compute total mean NSDR
        total_nsdr = 0
        for source, nsdr in zip(self.model_config["sources"], nsdrs):
            self.log(f"val_nsdr_{source}", nsdr, prog_bar=False, logger=True, sync_dist=True)
            total_nsdr += nsdr
        mean_nsdr = total_nsdr / len(self.model_config["sources"])
        self.log("val_nsdr", mean_nsdr, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx):
        """Test step: evaluates model performance using NSDR metric."""
        batch, mean, std = batch
        mean = mean.half()
        std = mean.half()
        sources = batch.to(self.device)
        mix = sources[:, 0]  # First channel is the mixture
        sources = sources[:, 1:]  # Remaining channels are sources

        # Apply model with overlap
        mix = (mix - mean) / std
        estimate = self.apply_model(mix, self.inference_overlap, mean, std)

        # Compute NSDR for each source
        nsdrs = new_sdr(sources, estimate.detach()).mean(0)
        nsdrs = self.trainer.strategy.reduce(nsdrs, reduce_op="mean")

        # Log NSDR per source and compute total mean NSDR
        total_nsdr = 0
        for source, nsdr in zip(self.model_config["sources"], nsdrs):
            self.log(f"test_nsdr_{source}", nsdr, prog_bar=False, logger=True, sync_dist=True)
            total_nsdr += nsdr
        mean_nsdr = total_nsdr / len(self.model_config["sources"])
        self.log("test_nsdr", mean_nsdr, prog_bar=True, logger=True, sync_dist=True)
