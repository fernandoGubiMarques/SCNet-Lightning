import torch
from torch import nn
from torch.nn import functional as F
from scnet.SCNet import Encoder, SeparationNet, Decoder



class SCNet_Backbone(nn.Module):

    def __init__(
        self,
        sources=["drums", "bass", "other", "vocals"],
        audio_channels=2,
        dims: list[int] = [4, 32, 64, 128],
        stft_config=dict(n_fft=4096, hop_length=1024, win_length=4096, normalized=True),
        band_config=None,
        conv_depths=[3, 2, 1],
        conv_config=dict(compress=4, kernel=3),
        num_dplayer=6,
        expand=1,
    ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.conv_config = conv_config
        
        self.band_config = band_config if band_config else {
            "low": dict(SR=0.175, stride=1, kernel=3),
            "mid": dict(SR=0.392, stride=4, kernel=4),
            "high": dict(SR=0.433, stride=16, kernel=16),
        }

        self.stft_config = {
            **stft_config,
            "center": True,
            # "window": torch.hamming_window(stft_config["win_length"]).cuda(),
        }

        self.dims = dims.copy()
        self.encoder = Encoder(dims, self.band_config, self.conv_config, conv_depths)
        # dims[0] *= len(sources)
        # self.decoder = Decoder(dims, self.band_config)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def forward(self, x):
        # B, C, L = x.shape
        B = x.shape[0]
        # In the initial padding, ensure that the number of frames after the STFT
        # (the length of the T dimension) is even, so that the RFFT operation can be
        # used in the separation network.

        hop_length = self.stft_config["hop_length"]
        padding = hop_length - x.shape[-1] % hop_length
        if (x.shape[-1] + padding) // hop_length % 2 == 0:
            padding += hop_length
        x = F.pad(x, (0, padding))

        # STFT
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self.stft_config, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(
            x.shape[0] // self.audio_channels,
            x.shape[3] * self.audio_channels,
            x.shape[1],
            x.shape[2],
        )

        B, C, Fr, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        x, skips, lengths, original_lengths = self.encoder(x)
        x = self.separation_net(x)
        
        return x, skips, lengths, original_lengths, B, Fr, T, std, mean, padding


class SCNet_Head(nn.Module):

    def __init__(
        self,
        sources=["drums", "bass", "other", "vocals"],
        audio_channels=2,
        dims: list[int] = [4, 32, 64, 128],
        stft_config=dict(n_fft=4096, hop_length=1024, win_length=4096, normalized=True),
        band_config=None,
        conv_depths=[3, 2, 1],
        conv_config=dict(compress=4, kernel=3),
        num_dplayer=6,
        expand=1,
    ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.conv_config = conv_config
        
        self.band_config = band_config if band_config else {
            "low": dict(SR=0.175, stride=1, kernel=3),
            "mid": dict(SR=0.392, stride=4, kernel=4),
            "high": dict(SR=0.433, stride=16, kernel=16),
        }

        self.stft_config = {
            **stft_config,
            "center": True,
            # "window": torch.hamming_window(stft_config["win_length"]).cuda(),
        }

        self.dims = dims.copy()
        self.encoder = Encoder(dims, self.band_config, self.conv_config, conv_depths)
        dims[0] *= len(sources)
        self.decoder = Decoder(dims, self.band_config)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def forward(self, x):
        x, skips, lengths, original_lengths, B, Fr, T, std, mean, padding = x
        x = self.decoder(x, skips, lengths, original_lengths)

        # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        x = x * std[:, None] + mean[:, None]
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        x = torch.istft(x, **self.stft_config)
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)

        x = x[:, :, :, :-padding]

        return x
