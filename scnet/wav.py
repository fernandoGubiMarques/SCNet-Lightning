# From HT demucs https://github.com/facebookresearch/demucs/tree/release_v4?tab=readme-ov-file

import os
from pathlib import Path
import tqdm

from torch.utils.data import Dataset
import yaml
import torch
from tqdm import tqdm
from typing import Literal
import torchaudio
from torch.nn import functional as F


class Wavset(Dataset):

    def __init__(
        self,
        partitioning_path: str,
        partition_name: Literal["train", "val", "test"],
        stems: list[str],
        other_stem: bool = True,
        ignore_stems: str | list[str] = "mixture",
        extension: str = "wav",
        segment: float | None = 11,
        stride: float = 1,
        samplerate: int = 44_100,
    ):
        self.other_stem = other_stem
        self.stems = stems
        self.ignore_stems = ignore_stems if isinstance(ignore_stems, list) else [ignore_stems]
        self.extension = extension
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.segment_samples = int(segment * samplerate) if segment else -1
        stride_samples = int(stride * samplerate)

        assert Path(partitioning_path).exists()
        with open(partitioning_path) as f:
            partitioning = yaml.safe_load(f)

        root = Path(partitioning["root"])

        # Find all song directories listed in the partition
        song_dirs: list[Path] = [
            root / song_dir for song_dir in partitioning[partition_name]
        ]

        # Compute metadata
        self.metadata = []
        for song_dir in tqdm(song_dirs, f"Computing {partition_name} set metadata"):
            # Assert all desired stems exist
            for stem in stems:
                assert (song_dir / f"{stem}.{extension}").exists()

            # Compute stats
            audio = []
            for file in song_dir.glob(f"*.{self.extension}"):
                if file.stem in ignore_stems:
                    continue
                stem_audio, sr = torchaudio.load(file)
                assert sr == samplerate
                audio.append(stem_audio.to(self.device))
            
            audio = sum(audio)
            mean = audio.mean()
            std = audio.std()
            
            audio_len = audio.shape[-1]
            if self.segment_samples > 0 and audio_len >= self.segment_samples:
                segment_starts = list(range(0, audio_len - self.segment_samples, stride_samples))
            else:
                segment_starts = [0]
            segment_durations = [self.segment_samples] * len(segment_starts)
            dirs = [song_dir] * len(segment_starts)
            means = [mean] * len(segment_starts)
            stds = [std] * len(segment_starts)

            self.metadata += list(zip(dirs, segment_starts, segment_durations, means, stds))
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, i):
        dir, start, duration, mean, std = self.metadata[i]
        dir: Path

        stems = {s: 0 for s in self.stems}
        if self.other_stem:
            stems["other"] = 0

        for file in dir.glob(f"*.{self.extension}"):
            if file.stem in self.ignore_stems:
                continue
            a, _ = torchaudio.load(file, start, duration)
            a = a.to(self.device)
            if file.stem in self.stems:
                stems[file.stem] = a
            elif self.other_stem:
                stems["other"] += a
        
        if self.other_stem:
            r =  torch.stack([stems[s] for s in self.stems] + [stems["other"]])
        else:
            r = torch.stack([stems[s] for s in self.stems])
        
        if r.shape[-1] < self.segment_samples and self.segment_samples > 0:
            r = F.pad(r, (0, self.segment_samples - r.shape[-1]))
        
        r -= mean
        r /= std
        
        return r, mean, std