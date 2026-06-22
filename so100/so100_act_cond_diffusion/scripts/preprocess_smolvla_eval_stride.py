import argparse
import json
from pathlib import Path

import av
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

from ldwma.datasets.lerobot_so100 import (
    _format_lerobot_path,
    _materialize_hf_file,
    _read_json,
)


DEFAULT_REPOS = [
    "lerobot/svla_so100_sorting",
    "lerobot/svla_so100_stacking",
    "lerobot/svla_so100_pickplace",
]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_info(repo_id: str, info: dict, camera_key: str) -> None:
    if info.get("robot_type") != "so100":
        raise ValueError(f"{repo_id}: expected robot_type='so100', got {info.get('robot_type')!r}.")
    action_shape = info.get("features", {}).get("action", {}).get("shape")
    if action_shape != [6]:
        raise ValueError(f"{repo_id}: expected 6D actions, got shape={action_shape}.")
    camera = info.get("features", {}).get(camera_key)
    if camera is None or camera.get("dtype") != "video":
        video_keys = [key for key, value in info.get("features", {}).items() if value.get("dtype") == "video"]
        raise ValueError(f"{repo_id}: camera {camera_key!r} unavailable. video keys={video_keys}.")


def update_output_info(info: dict, camera_key: str, fps: int, stride: int, episodes: list[dict]) -> dict:
    output_info = dict(info)
    output_info["fps"] = fps
    output_info["original_fps"] = int(info.get("fps", 30))
    output_info["downsample_stride"] = stride
    output_info["total_episodes"] = len(episodes)
    output_info["total_frames"] = int(sum(int(episode["length"]) for episode in episodes))

    output_features = {}
    for key, feature in info["features"].items():
        if feature.get("dtype") == "video" and key != camera_key:
            continue
        updated = dict(feature)
        if "fps" in updated:
            updated["fps"] = fps
        if updated.get("dtype") == "video" and "info" in updated:
            updated["info"] = dict(updated["info"])
            updated["info"]["video.fps"] = float(fps)
        output_features[key] = updated
    output_info["features"] = output_features
    return output_info


def sample_parquet(source: Path, destination: Path, stride: int, fps: int) -> int:
    table = pd.read_parquet(source)
    indices = list(range(0, len(table), stride))
    sampled = table.iloc[indices].copy().reset_index(drop=True)
    if "frame_index" in sampled.columns:
        sampled["frame_index"] = np.arange(len(sampled), dtype=np.int64)
    if "timestamp" in sampled.columns:
        sampled["timestamp"] = np.arange(len(sampled), dtype=np.float64) / float(fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(destination, index=False)
    return len(sampled)


def sample_video(source: Path, destination: Path, stride: int, expected_frames: int, fps: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(source)) as container, imageio.get_writer(
        destination,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index % stride != 0:
                continue
            writer.append_data(frame.to_ndarray(format="rgb24"))
            written += 1
    if written != expected_frames:
        raise IndexError(f"{source}: video produced {written} frames, parquet expected {expected_frames}.")


def process_repo(
    repo_id: str,
    output_root: Path,
    cache_dir: str,
    token: str | None,
    stride: int,
    camera_key: str,
    max_episodes: int | None,
    overwrite: bool,
) -> dict:
    with _materialize_hf_file(repo_id, "meta/info.json", cache_dir=cache_dir, token=token) as info_path:
        info = _read_json(info_path)
    validate_info(repo_id, info, camera_key)

    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        raise ValueError(f"{repo_id}: info.json has no valid total_episodes.")
    if max_episodes is not None:
        total_episodes = min(total_episodes, max_episodes)

    original_fps = int(info.get("fps", 30))
    fps = max(1, int(round(original_fps / stride)))
    chunks_size = int(info.get("chunks_size", 1000))
    local_dataset_path = Path(*repo_id.split("/"))
    output_dataset_root = output_root / local_dataset_path
    output_episodes = []

    iterator = tqdm(range(total_episodes), desc=repo_id)
    for episode_index in iterator:
        data_relative = _format_lerobot_path(info["data_path"], episode_index, chunks_size)
        video_relative = _format_lerobot_path(info["video_path"], episode_index, chunks_size, camera_key)
        output_data = output_dataset_root / data_relative
        output_video = output_dataset_root / video_relative

        if overwrite or not output_data.exists():
            with _materialize_hf_file(repo_id, data_relative, cache_dir=cache_dir, token=token) as source_data:
                sampled_length = sample_parquet(source_data, output_data, stride, fps)
        else:
            sampled_length = len(pd.read_parquet(output_data, columns=["action"]))

        if overwrite or not output_video.exists():
            with _materialize_hf_file(repo_id, video_relative, cache_dir=cache_dir, token=token) as source_video:
                sample_video(source_video, output_video, stride, sampled_length, fps)

        output_episodes.append({"episode_index": episode_index, "length": sampled_length})

    output_info = update_output_info(info, camera_key, fps, stride, output_episodes)
    write_json(output_dataset_root / "meta" / "info.json", output_info)
    write_jsonl(output_dataset_root / "meta" / "episodes.jsonl", output_episodes)
    return {
        "repo_id": repo_id,
        "output_path": local_dataset_path.as_posix(),
        "camera_key": camera_key,
        "stride": stride,
        "fps": fps,
        "episodes": len(output_episodes),
        "frames": output_info["total_frames"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local stride-sampled SmolVLA SO100 evaluation datasets.")
    parser.add_argument("--repo-id", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--output-root", default="/workspace/smolvla_eval_stride10")
    parser.add_argument("--cache-dir", default="/tmp/smolvla_eval_preprocess_cache")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for repo_id in args.repo_id:
        print(f"[SmolVLA eval preprocess] processing {repo_id}", flush=True)
        summaries.append(
            process_repo(
                repo_id=repo_id,
                output_root=output_root,
                cache_dir=args.cache_dir,
                token=args.hf_token,
                stride=args.stride,
                camera_key=args.camera_key,
                max_episodes=args.max_episodes,
                overwrite=args.overwrite,
            )
        )

    write_json(output_root / "preprocess_summary.json", {"datasets": summaries})
    print(json.dumps({"datasets": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
