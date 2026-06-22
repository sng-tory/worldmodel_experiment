import argparse
import json
from pathlib import Path

import av
import imageio
import numpy as np
import pandas as pd
from huggingface_hub import list_repo_files
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
    output_info["data_path"] = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
    output_info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4"

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


def sample_parquet(source: Path, destination: Path, episode_index: int, stride: int, fps: int) -> int:
    table = pd.read_parquet(source)
    table = table.loc[table["episode_index"] == episode_index].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"{source}: episode_index={episode_index} has no data rows.")
    indices = list(range(0, len(table), stride))
    sampled = table.iloc[indices].copy().reset_index(drop=True)
    if "frame_index" in sampled.columns:
        sampled["frame_index"] = np.arange(len(sampled), dtype=np.int64)
    if "timestamp" in sampled.columns:
        sampled["timestamp"] = np.arange(len(sampled), dtype=np.float64) / float(fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(destination, index=False)
    return len(sampled)


def sample_video(
    source: Path,
    destination: Path,
    stride: int,
    expected_frames: int,
    fps: int,
    from_timestamp: float,
    to_timestamp: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(source)) as container, imageio.get_writer(
        destination,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        stream = container.streams.video[0]
        container.seek(int(from_timestamp / float(stream.time_base)), stream=stream, backward=True)
        episode_frame_index = 0
        for frame in container.decode(stream):
            frame_time = float(frame.time) if frame.time is not None else None
            if frame_time is not None and frame_time + 1e-6 < from_timestamp:
                continue
            if frame_time is not None and frame_time + 1e-6 >= to_timestamp:
                break
            if episode_frame_index % stride != 0:
                episode_frame_index += 1
                continue
            writer.append_data(frame.to_ndarray(format="rgb24"))
            written += 1
            episode_frame_index += 1
            if written == expected_frames:
                break
    if written != expected_frames:
        raise IndexError(f"{source}: video produced {written} frames, parquet expected {expected_frames}.")


def load_episode_metadata(repo_id: str, cache_dir: str, token: str | None) -> pd.DataFrame:
    repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    episode_files = sorted(
        filename
        for filename in repo_files
        if filename.startswith("meta/episodes/") and filename.endswith(".parquet")
    )
    if not episode_files:
        raise ValueError(f"{repo_id}: no LeRobot v3 meta/episodes parquet files found.")
    tables = []
    for filename in episode_files:
        with _materialize_hf_file(repo_id, filename, cache_dir=cache_dir, token=token) as metadata_path:
            tables.append(pd.read_parquet(metadata_path))
    return pd.concat(tables, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


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

    episode_metadata = load_episode_metadata(repo_id, cache_dir, token)
    total_episodes = len(episode_metadata)
    if max_episodes is not None:
        total_episodes = min(total_episodes, max_episodes)

    original_fps = int(info.get("fps", 30))
    fps = max(1, int(round(original_fps / stride)))
    local_dataset_path = Path(*repo_id.split("/"))
    output_dataset_root = output_root / local_dataset_path
    output_episodes = []

    iterator = tqdm(episode_metadata.iloc[:total_episodes].to_dict("records"), desc=repo_id)
    for episode in iterator:
        episode_index = int(episode["episode_index"])
        source_data_relative = info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
        video_prefix = f"videos/{camera_key}"
        source_video_relative = info["video_path"].format(
            video_key=camera_key,
            chunk_index=int(episode[f"{video_prefix}/chunk_index"]),
            file_index=int(episode[f"{video_prefix}/file_index"]),
        )
        output_chunk = episode_index // int(info.get("chunks_size", 1000))
        output_data = output_dataset_root / f"data/chunk-{output_chunk:03d}/episode_{episode_index:06d}.parquet"
        output_video = (
            output_dataset_root
            / f"videos/{camera_key}/chunk-{output_chunk:03d}/episode_{episode_index:06d}.mp4"
        )

        if overwrite or not output_data.exists():
            with _materialize_hf_file(repo_id, source_data_relative, cache_dir=cache_dir, token=token) as source_data:
                sampled_length = sample_parquet(source_data, output_data, episode_index, stride, fps)
        else:
            sampled_length = len(pd.read_parquet(output_data, columns=["action"]))

        if overwrite or not output_video.exists():
            with _materialize_hf_file(repo_id, source_video_relative, cache_dir=cache_dir, token=token) as source_video:
                sample_video(
                    source_video,
                    output_video,
                    stride,
                    sampled_length,
                    fps,
                    float(episode[f"{video_prefix}/from_timestamp"]),
                    float(episode[f"{video_prefix}/to_timestamp"]),
                )

        tasks = episode.get("tasks", [])
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        output_episodes.append({"episode_index": episode_index, "length": sampled_length, "tasks": tasks})

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
