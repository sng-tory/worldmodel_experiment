import argparse
import json
import shutil
from pathlib import Path

import av
import imageio
import numpy as np
import pandas as pd
from huggingface_hub import list_repo_files
from tqdm import tqdm

from ldwma.datasets.lerobot_so100 import (
    _format_lerobot_path,
    _get_hf_token,
    _materialize_hf_file,
    _read_json,
    _read_jsonl,
    _select_video_key,
    discover_remote_lerobot_so100_datasets,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_optional_metadata(
    repo_id: str,
    dataset_path: str,
    output_dataset_root: Path,
    cache_dir: str | None,
    token: str | None,
) -> None:
    files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=_get_hf_token(token))
    prefix = f"{dataset_path}/meta/"
    for filename in files:
        if not filename.startswith(prefix):
            continue
        rel_meta_path = filename[len(prefix) :]
        if rel_meta_path in {"info.json", "episodes.jsonl"} or not rel_meta_path:
            continue
        try:
            with _materialize_hf_file(repo_id, filename, cache_dir=cache_dir, token=token) as src_path:
                dst_path = output_dataset_root / "meta" / rel_meta_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
        except Exception as error:
            print(f"[SO100 preprocess] skipped optional metadata {filename}: {error}", flush=True)


def sampled_indices(length: int, stride: int) -> list[int]:
    return list(range(0, length, stride))


def write_sampled_parquet(src_path: Path, dst_path: Path, indices: list[int], fps: int) -> None:
    table = pd.read_parquet(src_path)
    sampled = table.iloc[indices].copy().reset_index(drop=True)
    if "frame_index" in sampled.columns:
        sampled["frame_index"] = np.arange(len(sampled), dtype=np.int64)
    if "timestamp" in sampled.columns:
        sampled["timestamp"] = np.arange(len(sampled), dtype=np.float64) / float(fps)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(dst_path, index=False)


def write_sampled_video(src_path: Path, dst_path: Path, indices: list[int], fps: int) -> None:
    wanted = set(indices)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src_path)) as container, imageio.get_writer(
        dst_path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        stream = container.streams.video[0]
        written = 0
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in wanted:
                writer.append_data(frame.to_ndarray(format="rgb24"))
                written += 1
                if written == len(indices):
                    break
    if written != len(indices):
        raise IndexError(f"{src_path} only produced {written}/{len(indices)} sampled frames.")


def process_episode(
    repo_id: str,
    dataset_path: str,
    output_dataset_root: Path,
    info: dict,
    episode: dict,
    video_key: str,
    stride: int,
    fps: int,
    cache_dir: str | None,
    token: str | None,
    overwrite: bool,
) -> dict:
    episode_index = int(episode["episode_index"])
    length = int(episode["length"])
    indices = sampled_indices(length, stride)
    if not indices:
        raise ValueError(f"Episode {episode_index} has no frames after stride={stride}.")

    chunks_size = int(info.get("chunks_size", 1000))
    data_rel = _format_lerobot_path(info["data_path"], episode_index, chunks_size)
    video_rel = _format_lerobot_path(info["video_path"], episode_index, chunks_size, video_key)
    output_data_path = output_dataset_root / data_rel
    output_video_path = output_dataset_root / video_rel

    if overwrite or not output_data_path.exists():
        with _materialize_hf_file(
            repo_id,
            f"{dataset_path}/{data_rel}",
            cache_dir=cache_dir,
            token=token,
        ) as src_data_path:
            write_sampled_parquet(src_data_path, output_data_path, indices, fps)

    if overwrite or not output_video_path.exists():
        with _materialize_hf_file(
            repo_id,
            f"{dataset_path}/{video_rel}",
            cache_dir=cache_dir,
            token=token,
        ) as src_video_path:
            write_sampled_video(src_video_path, output_video_path, indices, fps)

    new_episode = dict(episode)
    new_episode["length"] = len(indices)
    if "duration_s" in new_episode:
        new_episode["duration_s"] = len(indices) / float(fps)
    return new_episode


def process_dataset(
    repo_id: str,
    dataset_path: str,
    output_root: Path,
    stride: int,
    cache_dir: str | None,
    token: str | None,
    camera_key: str,
    max_episodes: int | None,
    overwrite: bool,
    copy_metadata: bool,
) -> None:
    output_dataset_root = output_root / dataset_path
    with _materialize_hf_file(repo_id, f"{dataset_path}/meta/info.json", cache_dir=cache_dir, token=token) as info_path:
        info = _read_json(info_path)
    with _materialize_hf_file(
        repo_id,
        f"{dataset_path}/meta/episodes.jsonl",
        cache_dir=cache_dir,
        token=token,
    ) as episodes_path:
        episodes = _read_jsonl(episodes_path)

    video_key = _select_video_key(info["features"], camera_key)
    original_fps = int(info.get("fps", 30))
    fps = max(1, int(round(original_fps / stride)))
    episodes_to_process = episodes[:max_episodes] if max_episodes is not None else episodes

    new_episodes = []
    iterator = tqdm(episodes_to_process, desc=f"{dataset_path} episodes")
    for episode in iterator:
        new_episode = process_episode(
            repo_id=repo_id,
            dataset_path=dataset_path,
            output_dataset_root=output_dataset_root,
            info=info,
            episode=episode,
            video_key=video_key,
            stride=stride,
            fps=fps,
            cache_dir=cache_dir,
            token=token,
            overwrite=overwrite,
        )
        new_episodes.append(new_episode)

    new_info = dict(info)
    new_info["fps"] = fps
    new_info["original_fps"] = original_fps
    new_info["downsample_stride"] = stride
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = int(sum(int(episode["length"]) for episode in new_episodes))

    write_json(output_dataset_root / "meta" / "info.json", new_info)
    write_jsonl(output_dataset_root / "meta" / "episodes.jsonl", new_episodes)
    if copy_metadata:
        copy_optional_metadata(repo_id, dataset_path, output_dataset_root, cache_dir, token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local stride-sampled SO100 LeRobot dataset.")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/community_dataset_v1")
    parser.add_argument("--output-root", default="/workspace/so100_stride5")
    parser.add_argument("--cache-dir", default="/tmp/so100_preprocess_cache")
    parser.add_argument("--dataset-path", nargs="+", default=["auto"])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--camera-key", default="auto")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-metadata", action="store_true")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive.")

    token = args.hf_token
    if args.dataset_path == ["auto"]:
        dataset_paths = discover_remote_lerobot_so100_datasets(args.repo_id, cache_dir=args.cache_dir, token=token)
    else:
        dataset_paths = args.dataset_path
    if args.max_datasets is not None:
        dataset_paths = dataset_paths[: args.max_datasets]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[SO100 preprocess] writing local dataset to {output_root}", flush=True)
    print(f"[SO100 preprocess] stride={args.stride}, datasets={len(dataset_paths)}", flush=True)

    for dataset_path in dataset_paths:
        process_dataset(
            repo_id=args.repo_id,
            dataset_path=dataset_path,
            output_root=output_root,
            stride=args.stride,
            cache_dir=args.cache_dir,
            token=token,
            camera_key=args.camera_key,
            max_episodes=args.max_episodes,
            overwrite=args.overwrite,
            copy_metadata=args.copy_metadata,
        )

    print("[SO100 preprocess] done", flush=True)


if __name__ == "__main__":
    main()
