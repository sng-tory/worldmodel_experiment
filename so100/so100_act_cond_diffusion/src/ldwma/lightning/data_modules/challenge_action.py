from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ldwma.datasets.lerobot_so100 import preprocess_video


class ChallengeActionDataset(Dataset):
    """Dataset for fitting the action extractor on challenge answer videos."""

    def __init__(
        self,
        challenge_root: str,
        answer_video_root: str,
        target_height: int = 320,
        target_width: int = 512,
        temporal_length: int = 16,
        pad: bool = True,
        normalize_actions: bool = True,
        action_stats_path: str | None = None,
        sample_ids: list[str] | None = None,
    ) -> None:
        self.challenge_root = Path(challenge_root)
        self.answer_video_root = Path(answer_video_root)
        self.target_height = target_height
        self.target_width = target_width
        self.temporal_length = temporal_length
        self.pad = pad
        self.normalize_actions = normalize_actions

        if sample_ids is None:
            action_ids = {path.stem for path in (self.challenge_root / "actions").glob("*.npy")}
            answer_ids = {path.stem for path in self.answer_video_root.glob("*.mp4")}
            sample_ids = sorted(action_ids & answer_ids)
        self.sample_ids = list(sample_ids)
        if not self.sample_ids:
            raise ValueError(
                f"No matching action/video pairs. "
                f"actions={self.challenge_root / 'actions'}, videos={self.answer_video_root}"
            )

        self.action_mean = None
        self.action_std = None
        if self.normalize_actions:
            if not action_stats_path:
                raise ValueError("normalize_actions=True requires action_stats_path.")
            with Path(action_stats_path).open("r", encoding="utf-8") as f:
                stats = json.load(f)
            self.action_mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self.action_std = torch.tensor(stats["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _read_video(self, path: Path) -> torch.Tensor:
        frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) != self.temporal_length:
            raise ValueError(f"{path}: expected {self.temporal_length} frames, got {len(frames)}")
        return preprocess_video(np.stack(frames, axis=0), self.target_height, self.target_width, self.pad)

    def __getitem__(self, index: int) -> dict:
        sample_id = self.sample_ids[index]
        action = torch.from_numpy(np.load(self.challenge_root / "actions" / f"{sample_id}.npy").astype(np.float32))
        if action.shape[0] != self.temporal_length:
            raise ValueError(f"{sample_id}: expected {self.temporal_length} actions, got {action.shape[0]}")
        if self.action_mean is not None and self.action_std is not None:
            action = (action - self.action_mean) / self.action_std
        return {
            "sample_id": sample_id,
            "video": self._read_video(self.answer_video_root / f"{sample_id}.mp4"),
            "act": action,
        }


class ChallengeActionDataModule(LightningDataModule):
    def __init__(
        self,
        challenge_root: str,
        answer_video_root: str,
        batch_size: int = 16,
        num_workers: int = 2,
        pin_memory: bool = True,
        target_height: int = 320,
        target_width: int = 512,
        temporal_length: int = 16,
        pad: bool = True,
        normalize_actions: bool = True,
        action_stats_path: str | None = None,
        val_fraction: float = 0.1,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.challenge_root = challenge_root
        self.answer_video_root = answer_video_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_height = target_height
        self.target_width = target_width
        self.temporal_length = temporal_length
        self.pad = pad
        self.normalize_actions = normalize_actions
        self.action_stats_path = action_stats_path
        self.val_fraction = val_fraction
        self.seed = seed
        self.shuffle = shuffle
        self.train_dataset = None
        self.val_dataset = None

    def _make_dataset(self, sample_ids: list[str]) -> ChallengeActionDataset:
        return ChallengeActionDataset(
            challenge_root=self.challenge_root,
            answer_video_root=self.answer_video_root,
            target_height=self.target_height,
            target_width=self.target_width,
            temporal_length=self.temporal_length,
            pad=self.pad,
            normalize_actions=self.normalize_actions,
            action_stats_path=self.action_stats_path,
            sample_ids=sample_ids,
        )

    def setup(self, stage=None) -> None:
        full_dataset = self._make_dataset(sample_ids=None)
        sample_ids = list(full_dataset.sample_ids)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(sample_ids)

        if len(sample_ids) < 2 or self.val_fraction <= 0:
            train_ids = sample_ids
            val_ids = []
        else:
            val_count = int(round(len(sample_ids) * self.val_fraction))
            val_count = min(max(val_count, 1), len(sample_ids) - 1)
            val_ids = sorted(sample_ids[:val_count])
            train_ids = sorted(sample_ids[val_count:])

        self.train_dataset = self._make_dataset(train_ids)
        self.val_dataset = self._make_dataset(val_ids) if val_ids else None
        print(
            f"[challenge action data] total={len(sample_ids)} train={len(train_ids)} val={len(val_ids)}",
            flush=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )