import torch
import torch.nn.functional as F
from torch import Tensor
from lightning import LightningModule
from scnet.utils import new_sdr
from scnet import augment
from scnet.SCNet import SCNet
from omegaconf import DictConfig
import einops


class SCNetLightning(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        optim_config: DictConfig,
        data_config: DictConfig,
        augment_config: DictConfig,
        inference_config: DictConfig,
        train_config: DictConfig,
    ):
        super().__init__()
        self.model = SCNet(**model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optim_config)

        # Compute the window size in samples
        self.window_size = data_config["samplerate"] * data_config["segment"]
        self.model_config = model_config
        self.inference_config = inference_config
        self.train_config = train_config

        # STFT configuration for spectral loss computation
        stft_config = model_config["stft_config"]
        self.stft_config = {
            **stft_config,
            "center": True,
            # "window": torch.hamming_window(stft_config["win_length"]).cuda(),
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

        self.window_weights = torch.hamming_window(self.window_size)

        assert len(train_config.loss_weights) == len(model_config.sources)
        self.loss_weights = torch.asarray(train_config.loss_weights).float().cuda()
        self.loss_weights /= self.loss_weights.sum()

    def forward(self, mix):
        """Forward pass through the SCNet model."""
        return self.model(mix)

    def waveform_forward(self, mix: Tensor) -> Tensor:
        mix, padding = self.ensure_even_frames(mix)
        mix = self.stft(mix)
        sources = self.forward(mix)
        sources = self.istft(sources)
        sources = sources[..., :-padding]
        return sources

    def ensure_even_frames(self, x: Tensor) -> tuple[Tensor, int]:
        """
        Pads audio tensor (..., L) to ensure that the number of frames after the STFT
        (the length of the last dimension in the frequency domain) is even,
        so that the RFFT operation can be used in the separation network.

        Returns padded tensor and padding size
        """
        hop_length = self.stft_config["hop_length"]
        L = x.shape[-1]
        padding = hop_length - L % hop_length
        if (L + padding) // hop_length % 2 == 0:
            padding += hop_length

        return F.pad(x, (0, padding)), padding

    def stft(self, x: Tensor) -> Tensor:
        """
        Performs batched stft with complex as channels for a tensor of shape (..., C, L).

        Returns spectrogram of shape (..., C * 2, Fr, T)
        """
        batch_shape = x.shape[:-2]
        C = x.shape[-2]

        x = einops.rearrange(x, "... C L -> (... C) L")
        y = torch.stft(x, **self.stft_config, return_complex=True)

        y = torch.view_as_real(y)
        y = einops.rearrange(y, "(B C) Fr T K -> B (C K) Fr T", C=C)

        # Split B into batch_shape
        y = y.view(*batch_shape, *y.shape[1:])

        return y.contiguous()

    def istft(self, y: Tensor) -> Tensor:
        """
        Inverse STFT for a tensor of shape (..., C * 2, Fr, T).

        Returns a signal of shape (..., C, L)
        """
        batch_shape = y.shape[:-3]
        C_total = y.shape[-3]
        C = C_total // 2

        dtype = y.dtype

        y = einops.rearrange(y, "... (C K) Fr T -> (... C) Fr T K", K=2)
        y = torch.view_as_complex(y.contiguous().half())

        x: Tensor = torch.istft(y, **self.stft_config)
        x = x.to(dtype=dtype)
        x = x.reshape(*batch_shape, C, -1)
        return x

    def configure_optimizers(self):
        """Returns the optimizer for training."""
        return self.optimizer

    def training_step(self, batch, batch_idx):
        """Single training step where the model learns from a batch."""
        batch, _, _ = batch
        sources = batch.to(self.device)
        sources = self.augment(sources)  # Apply augmentations

        sources, padding = self.ensure_even_frames(sources)
        mix = sources.sum(dim=1)  # Create mixture by summing sources

        with torch.no_grad():
            sources = self.stft(sources)

        estimate = self(mix)  # Forward pass

        loss = self.spec_rmse_loss(estimate, sources)  # Compute loss
        self.log("train_loss", loss, prog_bar=True, logger=True)  # Log loss

        return loss

    def spec_rmse_loss(self, estimate: Tensor, sources: Tensor) -> Tensor:
        # estimate, tensor: (B, S, CK, Fr, T)
        mse = torch.mean((estimate - sources) ** 2, dim=(2, 3, 4))
        rmse = torch.sqrt(mse)
        return rmse.mean(dim=0).sum()

    @torch.no_grad
    def apply_model(self, x: Tensor):
        """Applies the model in a windowed fashion with overlap handling."""

        n_sources = len(self.model_config["sources"])
        B, C, T = x.shape  # (Batch, Channels, Time)
        assert B == 1, "Batch size should always be 1 during validation."

        x = x.squeeze(0)  # Remove batch dimension -> (2, T)

        # Pad input so that T is a multiple of window_size
        pad_length = (self.window_size - (T % self.window_size)) % self.window_size
        x_padded = torch.nn.functional.pad(x, (0, pad_length))  # (2, T + pad_length)
        T_padded = x_padded.shape[-1]

        # Compute step size based on overlap ratio
        step_size = int(self.window_size * (1 - self.inference_config.overlap))
        window_starts = list(range(0, T_padded - self.window_size + 1, step_size))

        # Stack windows into a single tensor (N, 2, window_size)
        x_windows = torch.stack(
            [x_padded[:, start : start + self.window_size] for start in window_starts],
            dim=0,
        )

        # Forward pass in parallel for all windows. Determine processing strategy based
        # on max_win_chunks
        if self.inference_config.max_parallel_windows is None:
            y_windows = self.waveform_forward(x_windows)
            # Process all windows in parallel
        else:
            y_windows = torch.zeros(
                (len(window_starts), n_sources, 2, self.window_size), device=x.device
            )
            for i in range(
                0, len(window_starts), self.inference_config.max_parallel_windows
            ):
                batch_x = x_windows[
                    i : i + self.inference_config.max_parallel_windows
                ]  # Select a batch of windows
                batch_y = self.waveform_forward(batch_x)  # Forward pass
                y_windows[i : i + batch_y.shape[0]] = batch_y

        # Prepare output tensor (n_sources, 2, T_padded) with zeros for accumulation
        output = torch.zeros((n_sources, 2, T_padded), device=x.device)
        weight = torch.zeros((n_sources, 2, T_padded), device=x.device)

        # Accumulate results and weights
        self.window_weights = self.window_weights.to(x.device)
        for i, start in enumerate(window_starts):
            end = start + self.window_size
            output[:, :, start:end] += y_windows[i] * self.window_weights
            weight[:, :, start:end] += self.window_weights

        # Normalize overlapping regions
        output /= torch.where(weight > 0, weight, torch.tensor(1.0, device=x.device))

        # Remove padding to restore original shape
        output = output[:, :, :T]

        # Add back batch dimension -> (1, n_sources, 2, T)
        output = output.unsqueeze(0)

        return output

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
        estimate = self.apply_model(mix)
        estimate = estimate * std + mean

        # Compute NSDR for each source
        nsdrs = new_sdr(sources, estimate.detach()).mean(0)
        nsdrs = self.trainer.strategy.reduce(nsdrs, reduce_op="mean")

        # Log NSDR per source and compute total mean NSDR
        total_nsdr = 0
        for source, nsdr in zip(self.model_config["sources"], nsdrs):
            self.log(
                f"val_nsdr_{source}", nsdr, prog_bar=False, logger=True, sync_dist=True
            )
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
        estimate = self.apply_model(mix)
        estimate = estimate * std + mean

        # Compute NSDR for each source
        nsdrs = new_sdr(sources, estimate.detach()).mean(0)
        nsdrs = self.trainer.strategy.reduce(nsdrs, reduce_op="mean")

        # Log NSDR per source and compute total mean NSDR
        total_nsdr = 0
        for source, nsdr in zip(self.model_config["sources"], nsdrs):
            self.log(
                f"test_nsdr_{source}", nsdr, prog_bar=False, logger=True, sync_dist=True
            )
            total_nsdr += nsdr
        mean_nsdr = total_nsdr / len(self.model_config["sources"])
        self.log("test_nsdr", mean_nsdr, prog_bar=True, logger=True, sync_dist=True)
