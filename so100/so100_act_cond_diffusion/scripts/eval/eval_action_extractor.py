
import argparse
import csv
import json
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from ldwma.datasets.lerobot_so100 import preprocess_video
from ldwma.models.action_extractor import SO100ActionExtractor
from torch.utils.data import DataLoader, Dataset


class ChallengeActionDataset(Dataset):
    def __init__(
        self,
        challenge_root: str,
        answer_video_root: str,
        target_height: int,
        target_width: int,
        pad: bool,
        temporal_length: int,
        action_stats_path: str | None,
        generated_video_root: str | None = None,
    ) -> None:
        self.challenge_root = Path(challenge_root)
        self.answer_video_root = Path(answer_video_root)
        self.generated_video_root = Path(generated_video_root) if generated_video_root else None
        image_ids = {path.stem for path in (self.challenge_root / "images").glob("*.png")}
        action_ids = {path.stem for path in (self.challenge_root / "actions").glob("*.npy")}
        answer_ids = {path.stem for path in self.answer_video_root.glob("*.mp4")}
        sample_ids = image_ids & action_ids & answer_ids
        if self.generated_video_root is not None:
            sample_ids &= {path.stem for path in self.generated_video_root.glob("*.mp4")}
        self.sample_ids = sorted(sample_ids)
        if not self.sample_ids:
            raise ValueError("No matching challenge actions and answer videos found.")
        self.target_height = target_height
        self.target_width = target_width
        self.pad = pad
        self.temporal_length = temporal_length
        self.action_mean = None
        self.action_std = None
        if action_stats_path:
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
        if self.action_mean is not None and self.action_std is not None:
            action = (action - self.action_mean) / self.action_std
        item = {
            "sample_id": sample_id,
            "act": action,
            "real_video": self._read_video(self.answer_video_root / f"{sample_id}.mp4"),
        }
        if self.generated_video_root is not None:
            item["generated_video"] = self._read_video(self.generated_video_root / f"{sample_id}.mp4")
        return item


def compute_errors(model, loader, device: torch.device) -> dict:
    real_abs = []
    real_sq = []
    gen_abs = []
    gen_sq = []
    per_dim_real = []
    per_dim_gen = []
    with torch.no_grad():
        for batch in loader:
            target = batch["act"].to(device)
            real_pred = model(batch["real_video"].to(device))
            real_err = real_pred - target
            real_abs.append(torch.abs(real_err).flatten(1).mean(dim=1).cpu())
            real_sq.append((real_err**2).flatten(1).mean(dim=1).cpu())
            per_dim_real.append(torch.abs(real_err).mean(dim=(0, 1)).cpu())
            if "generated_video" in batch:
                gen_pred = model(batch["generated_video"].to(device))
                gen_err = gen_pred - target
                gen_abs.append(torch.abs(gen_err).flatten(1).mean(dim=1).cpu())
                gen_sq.append((gen_err**2).flatten(1).mean(dim=1).cpu())
                per_dim_gen.append(torch.abs(gen_err).mean(dim=(0, 1)).cpu())

    real_mae = torch.cat(real_abs).mean().item()
    real_mse = torch.cat(real_sq).mean().item()
    row = {
        "num_samples": sum(len(x) for x in real_abs),
        "real_mae": real_mae,
        "real_mse": real_mse,
    }
    for idx, value in enumerate(torch.stack(per_dim_real).mean(dim=0).tolist()):
        row[f"real_mae_dim_{idx}"] = value
    if gen_abs:
        gen_mae = torch.cat(gen_abs).mean().item()
        gen_mse = torch.cat(gen_sq).mean().item()
        row.update(
            {
                "generated_mae": gen_mae,
                "generated_mse": gen_mse,
                "action_error_ratio": gen_mae / (real_mae + 1e-9),
            }
        )
        for idx, value in enumerate(torch.stack(per_dim_gen).mean(dim=0).tolist()):
            row[f"generated_mae_dim_{idx}"] = value
    return row


def write_csv(path: str, row: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SO-100 action extractor on challenge answer videos.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--challenge-root", default="/workspace/smolvla_eval_challenge_stride5/challenge")
    parser.add_argument("--answer-video-root", default="/workspace/smolvla_eval_challenge_stride5/answers/videos")
    parser.add_argument("--generated-video-root", default=None)
    parser.add_argument("--csv-path", default="/workspace/so100_action_extractor/eval_challenge.csv")
    parser.add_argument("--action-stats-path", default="/workspace/so100_stride5/so100_action_statistics.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-height", type=int, default=320)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--pad", action="store_true", default=True)
    parser.add_argument("--temporal-length", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SO100ActionExtractor.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    dataset = ChallengeActionDataset(
        challenge_root=args.challenge_root,
        answer_video_root=args.answer_video_root,
        target_height=args.target_height,
        target_width=args.target_width,
        pad=args.pad,
        temporal_length=args.temporal_length,
        action_stats_path=args.action_stats_path,
        generated_video_root=args.generated_video_root,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    row = compute_errors(model, loader, device)
    write_csv(args.csv_path, row)
    print(row)


if __name__ == "__main__":
    main()
