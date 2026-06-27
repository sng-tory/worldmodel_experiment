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
    _select_video_keys,
    discover_remote_lerobot_so100_datasets,
)

EXCLUDED_CAMERA_NAME_PARTS = ("wrist", "gripper", "arm")
PROCESSING_SUMMARY_FILES = (
    "PROCESSING_SUMMARY_backup_20250811_090157.json",
    "PROCESSING_SUMMARY.json",
)




def has_excluded_camera_name(camera_key: str) -> bool:
    return any(part in camera_key.lower() for part in EXCLUDED_CAMERA_NAME_PARTS)


def collect_mapping_entries(node) -> list[dict]:
    entries = []
    if isinstance(node, dict):
        mapping = node.get("mapping_applied")
        if isinstance(mapping, dict):
            entries.append(node)
        for value in node.values():
            entries.extend(collect_mapping_entries(value))
    elif isinstance(node, list):
        for value in node:
            entries.extend(collect_mapping_entries(value))
    return entries


def entry_matches_dataset(entry: dict, dataset_path: str) -> bool:
    dataset_parts = [part for part in dataset_path.replace("\\", "/").split("/") if part]
    if not dataset_parts:
        return False
    leaf_name = dataset_parts[-1]
    for key, value in entry.items():
        if key == "mapping_applied" or not isinstance(value, str):
            continue
        normalized = value.replace("\\", "/")
        if dataset_path in normalized or leaf_name == normalized or normalized.endswith(f"/{leaf_name}"):
            return True
    return False


def mapping_for_dataset(summary: dict | None, dataset_path: str) -> dict[str, str]:
    if summary is None:
        return {}
    entries = collect_mapping_entries(summary)
    for entry in entries:
        if entry_matches_dataset(entry, dataset_path):
            return entry["mapping_applied"]
    if len(entries) == 1:
        return entries[0]["mapping_applied"]
    return {}


def load_processing_summaries(
    repo_id: str,
    filenames: list[str] | tuple[str, ...],
    cache_dir: str | None,
    token: str | None,
) -> list[tuple[str, dict]]:
    summaries = []
    for filename in filenames:
        if not filename:
            continue
        try:
            with _materialize_hf_file(repo_id, filename, cache_dir=cache_dir, token=token) as summary_path:
                summaries.append((filename, _read_json(summary_path)))
        except Exception as error:
            print(f"[SO100 preprocess] skipped processing summary {filename}: {error}", flush=True)
    return summaries


def mapping_for_dataset_from_summaries(summaries: list[tuple[str, dict]], dataset_path: str) -> dict[str, str]:
    for filename, summary in summaries:
        mapping = mapping_for_dataset(summary, dataset_path)
        if mapping:
            print(f"[SO100 preprocess] {dataset_path}: using mapping from {filename}", flush=True)
            return mapping
    print(f"[SO100 preprocess] {dataset_path}: no processing-summary mapping; using observation.image* auto fallback", flush=True)
    return {}


def _is_observation_image_key(camera_key: str) -> bool:
    return camera_key.startswith("observation.image") and not camera_key.startswith("observation.images.")


def _select_observation_image_auto_key(video_keys: list[str]) -> str:
    candidates = [key for key in video_keys if _is_observation_image_key(key) and not has_excluded_camera_name(key)]
    if not candidates:
        raise ValueError(f"No observation.image* auto fallback video key available. video keys={video_keys}.")
    for preferred in ("observation.image", "observation.image1", "observation.image2", "observation.image3"):
        if preferred in candidates:
            return preferred
    return sorted(candidates)[0]


def select_video_key_pairs(features: dict, camera_key: str, mapping_applied: dict[str, str]) -> list[tuple[str, str]]:
    video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    if camera_key != "auto":
        return [(key, key) for key in _select_video_keys(features, camera_key)]

    pairs = []
    for source_key, output_key in mapping_applied.items():
        if has_excluded_camera_name(source_key):
            continue
        if source_key in video_keys:
            pairs.append((source_key, output_key))
        elif output_key in video_keys:
            pairs.append((output_key, output_key))
    if pairs:
        return list(dict.fromkeys(pairs))

    fallback_key = _select_observation_image_auto_key(video_keys)
    return [(fallback_key, fallback_key)]


def build_output_info(
    info: dict,
    video_key_pairs: list[tuple[str, str]],
    fps: int,
    original_fps: int,
    stride: int,
    episodes: list[dict],
) -> dict:
    output_info = dict(info)
    output_info["fps"] = fps
    output_info["original_fps"] = original_fps
    output_info["downsample_stride"] = stride
    output_info["total_episodes"] = len(episodes)
    output_info["total_frames"] = int(sum(int(episode["length"]) for episode in episodes))
    output_info["total_videos"] = len(episodes) * len(video_key_pairs)

    output_features = {}
    pair_by_output_key = {output_key: source_key for source_key, output_key in video_key_pairs}
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") == "video":
            continue
        output_features[key] = dict(feature)
    for output_key, source_key in pair_by_output_key.items():
        updated = dict(info["features"][source_key])
        if "info" in updated:
            updated["info"] = dict(updated["info"])
            updated["info"]["video.fps"] = float(fps)
        if "fps" in updated:
            updated["fps"] = fps
        output_features[output_key] = updated
    output_info["features"] = output_features
    return output_info

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
    video_key_pairs: list[tuple[str, str]],
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
    output_data_path = output_dataset_root / data_rel

    if overwrite or not output_data_path.exists():
        with _materialize_hf_file(
            repo_id,
            f"{dataset_path}/{data_rel}",
            cache_dir=cache_dir,
            token=token,
        ) as src_data_path:
            write_sampled_parquet(src_data_path, output_data_path, indices, fps)

    for source_video_key, output_video_key in video_key_pairs:
        source_video_rel = _format_lerobot_path(info["video_path"], episode_index, chunks_size, source_video_key)
        output_video_rel = _format_lerobot_path(info["video_path"], episode_index, chunks_size, output_video_key)
        output_video_path = output_dataset_root / output_video_rel
        if overwrite or not output_video_path.exists():
            with _materialize_hf_file(
                repo_id,
                f"{dataset_path}/{source_video_rel}",
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
    processing_summaries: list[tuple[str, dict]],
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

    mapping_applied = mapping_for_dataset_from_summaries(processing_summaries, dataset_path)
    video_key_pairs = select_video_key_pairs(info["features"], camera_key, mapping_applied)
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
            video_key_pairs=video_key_pairs,
            stride=stride,
            fps=fps,
            cache_dir=cache_dir,
            token=token,
            overwrite=overwrite,
        )
        new_episodes.append(new_episode)

    new_info = build_output_info(info, video_key_pairs, fps, original_fps, stride, new_episodes)

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
    parser.add_argument("--processing-summary-file", nargs="*", default=list(PROCESSING_SUMMARY_FILES))
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive.")

    token = args.hf_token
    processing_summaries = load_processing_summaries(args.repo_id, args.processing_summary_file, args.cache_dir, token)
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
            processing_summaries=processing_summaries,
        )

    print("[SO100 preprocess] done", flush=True)


if __name__ == "__main__":
    main()
