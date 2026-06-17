import json
import random
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _format_lerobot_path(template: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> str:
    chunk_index = episode_index // chunks_size
    return template.format(
        episode_index=episode_index,
        episode_chunk=chunk_index,
        chunk_index=chunk_index,
        file_index=episode_index,
        video_key=video_key,
    )


def _select_video_key(features: dict, requested_key: str | None) -> str:
    video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    if not video_keys:
        raise ValueError("No video feature found in LeRobot metadata.")
    if requested_key and requested_key != "auto":
        if requested_key not in video_keys:
            raise ValueError(f"Requested camera_key={requested_key!r}, available video keys={video_keys}.")
        return requested_key
    for preferred in ("observation.image", "observation.images.image", "observation.images.cam_middle"):
        if preferred in video_keys:
            return preferred
    return video_keys[0]


def _decode_video_clip(video_path: Path, indices: list[int]) -> np.ndarray:
    wanted = set(indices)
    frames = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in wanted:
                frames[frame_idx] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break

    missing = [idx for idx in indices if idx not in frames]
    if missing:
        raise IndexError(f"Video {video_path} did not contain requested frames {missing[:5]}.")
    return np.stack([frames[idx] for idx in indices], axis=0)


def preprocess_video(video: np.ndarray, target_height: int, target_width: int, pad: bool) -> torch.Tensor:
    video_tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2) / 255.0
    if pad:
        _, _, height, width = video_tensor.shape
        scale = min(target_height / height, target_width / width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        video_tensor = F.interpolate(
            video_tensor,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        pad_top = (target_height - resized_height) // 2
        pad_bottom = target_height - resized_height - pad_top
        pad_left = (target_width - resized_width) // 2
        pad_right = target_width - resized_width - pad_left
        video_tensor = F.pad(video_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    else:
        _, _, height, width = video_tensor.shape
        scale = max(target_height / height, target_width / width)
        resized_height = max(target_height, round(height * scale))
        resized_width = max(target_width, round(width * scale))
        video_tensor = F.interpolate(
            video_tensor,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        top = (resized_height - target_height) // 2
        left = (resized_width - target_width) // 2
        video_tensor = video_tensor[:, :, top : top + target_height, left : left + target_width]

    video_tensor = (video_tensor - 0.5) * 2.0
    return video_tensor.permute(1, 0, 2, 3).contiguous()


class LeRobotSO100Dataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_paths: list[str],
        train: bool,
        traj_len: int,
        target_height: int,
        target_width: int,
        pad: bool = True,
        camera_key: str = "auto",
        val_fraction: float = 0.05,
        seed: int = 0,
        downsample: int = 1,
        use_language: bool = False,
        fps: int | None = None,
        frame_stride: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.traj_len = traj_len
        self.target_height = target_height
        self.target_width = target_width
        self.pad = pad
        self.camera_key = camera_key
        self.downsample = downsample
        self.use_language = use_language
        self.override_fps = fps
        self.override_frame_stride = frame_stride
        self.rng = random.Random(seed)

        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0.0, 1.0).")

        examples = []
        for rel_dataset_path in dataset_paths:
            dataset_root = self.root / rel_dataset_path
            examples.extend(self._load_dataset_examples(dataset_root))

        examples = [example for example in examples if example["length"] >= self.traj_len * self.downsample]
        self.rng.shuffle(examples)
        val_count = int(len(examples) * val_fraction)
        if val_fraction > 0 and val_count == 0 and len(examples) > 1:
            val_count = 1
        self.examples = examples[val_count:] if train else examples[:val_count]
        if not self.examples:
            split = "train" if train else "val"
            raise ValueError(f"No {split} examples available. Check dataset_paths, traj_len, and val_fraction.")

    def _load_dataset_examples(self, dataset_root: Path) -> list[dict]:
        info = _read_json(dataset_root / "meta" / "info.json")
        episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
        video_key = _select_video_key(info["features"], self.camera_key)
        chunks_size = int(info.get("chunks_size", 1000))
        fps = int(self.override_fps or info.get("fps", 30))
        frame_stride = int(self.override_frame_stride or fps)
        data_template = info["data_path"]
        video_template = info["video_path"]

        examples = []
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            data_path = dataset_root / _format_lerobot_path(data_template, episode_index, chunks_size)
            video_path = dataset_root / _format_lerobot_path(video_template, episode_index, chunks_size, video_key)
            task = episode.get("tasks", [""])[0] if self.use_language else ""
            examples.append(
                {
                    "data_path": data_path,
                    "video_path": video_path,
                    "length": int(episode["length"]),
                    "task": task,
                    "fps": fps,
                    "frame_stride": frame_stride,
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        max_start = example["length"] - self.traj_len * self.downsample
        start_idx = self.rng.randint(0, max_start)
        frame_indices = [start_idx + i * self.downsample for i in range(self.traj_len)]

        table = pd.read_parquet(example["data_path"], columns=["action"])
        actions = np.stack(table["action"].iloc[frame_indices].to_numpy()).astype(np.float32)
        if actions.shape[-1] != 6:
            raise ValueError(f"Expected SO-100 action dim 6, got {actions.shape[-1]} from {example['data_path']}.")

        video = _decode_video_clip(example["video_path"], frame_indices)
        return {
            "video": preprocess_video(video, self.target_height, self.target_width, self.pad),
            "act": torch.from_numpy(actions),
            "caption": example["task"],
            "fps": torch.tensor(example["fps"], dtype=torch.long),
            "frame_stride": torch.tensor(example["frame_stride"], dtype=torch.long),
            "start_idx": torch.tensor(start_idx, dtype=torch.long),
        }
