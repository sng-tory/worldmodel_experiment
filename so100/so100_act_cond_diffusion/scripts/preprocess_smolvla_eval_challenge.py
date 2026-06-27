import argparse
import json
import random
from pathlib import Path

import av
import imageio
import numpy as np
import pandas as pd
from huggingface_hub import list_repo_files
from tqdm import tqdm

from ldwma.datasets.lerobot_so100 import _materialize_hf_file, _read_json


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


def load_episode_actions(source: Path, episode_index: int) -> pd.DataFrame:
    table = pd.read_parquet(source)
    table = table.loc[table["episode_index"] == episode_index].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"{source}: episode_index={episode_index} has no data rows.")
    if "action" not in table.columns:
        raise ValueError(f"{source}: missing action column.")
    return table


def sample_indices(length: int, stride: int) -> list[int]:
    return list(range(0, length, stride))


def write_video_window_and_start_image(
    source: Path,
    video_destination: Path,
    image_destination: Path,
    wanted_indices: list[int],
    fps: int,
    from_timestamp: float,
    to_timestamp: float,
) -> None:
    wanted = set(wanted_indices)
    video_destination.parent.mkdir(parents=True, exist_ok=True)
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    start_image_written = False
    with av.open(str(source)) as container, imageio.get_writer(
        video_destination,
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
            if episode_frame_index in wanted:
                frame_array = frame.to_ndarray(format="rgb24")
                if episode_frame_index == wanted_indices[0] and not start_image_written:
                    imageio.imwrite(image_destination, frame_array)
                    start_image_written = True
                writer.append_data(frame_array)
                written += 1
                if written == len(wanted_indices):
                    break
            episode_frame_index += 1
    if written != len(wanted_indices):
        raise IndexError(f"{source}: video produced {written}/{len(wanted_indices)} requested frames.")
    if not start_image_written:
        raise IndexError(f"{source}: failed to write start image for frame {wanted_indices[0]}.")


def process_repo(
    repo_id: str,
    output_root: Path,
    cache_dir: str,
    token: str | None,
    stride: int,
    camera_key: str,
    window_len: int,
    samples_per_episode: int,
    max_episodes: int | None,
    seed: int,
    overwrite: bool,
) -> dict:
    with _materialize_hf_file(repo_id, "meta/info.json", cache_dir=cache_dir, token=token) as info_path:
        info = _read_json(info_path)
    validate_info(repo_id, info, camera_key)

    episode_metadata = load_episode_metadata(repo_id, cache_dir, token)
    if max_episodes is not None:
        episode_metadata = episode_metadata.iloc[:max_episodes]

    original_fps = int(info.get("fps", 30))
    fps = max(1, int(round(original_fps / stride)))
    local_dataset_path = Path(*repo_id.split("/"))
    challenge_root = output_root / "challenge" / local_dataset_path
    answers_root = output_root / "answers" / local_dataset_path
    rng = random.Random(seed + sum(ord(ch) for ch in repo_id))
    challenge_rows = []
    answer_rows = []

    iterator = tqdm(episode_metadata.to_dict("records"), desc=repo_id)
    sample_counter = 0
    skipped = 0
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

        with _materialize_hf_file(repo_id, source_data_relative, cache_dir=cache_dir, token=token) as source_data:
            table = load_episode_actions(source_data, episode_index)
        sampled = sample_indices(len(table), stride)
        if len(sampled) < window_len:
            skipped += 1
            continue

        max_start = len(sampled) - window_len
        starts = [rng.randint(0, max_start) for _ in range(samples_per_episode)]
        for start_idx in starts:
            sample_id = f"{repo_id.replace('/', '__')}__episode_{episode_index:06d}__start_{start_idx:06d}"
            window_positions = list(range(start_idx, start_idx + window_len))
            original_frame_indices = [sampled[position] for position in window_positions]
            action_sequence = np.stack(table["action"].iloc[original_frame_indices].to_numpy()).astype(np.float32)

            image_rel = Path("images") / f"{sample_id}.png"
            actions_rel = Path("actions") / f"{sample_id}.npy"
            challenge_meta_rel = Path("metadata") / f"{sample_id}.json"
            answer_video_rel = Path("videos") / camera_key / f"{sample_id}.mp4"
            image_path = challenge_root / image_rel
            actions_path = challenge_root / actions_rel
            challenge_meta_path = challenge_root / challenge_meta_rel
            answer_video_path = answers_root / answer_video_rel

            if overwrite or not actions_path.exists():
                actions_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(actions_path, action_sequence)

            if overwrite or not answer_video_path.exists() or not image_path.exists():
                with _materialize_hf_file(repo_id, source_video_relative, cache_dir=cache_dir, token=token) as source_video:
                    write_video_window_and_start_image(
                        source_video,
                        answer_video_path,
                        image_path,
                        original_frame_indices,
                        fps,
                        float(episode[f"{video_prefix}/from_timestamp"]),
                        float(episode[f"{video_prefix}/to_timestamp"]),
                    )

            tasks = episode.get("tasks", [])
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            challenge_row = {
                "sample_id": sample_id,
                "repo_id": repo_id,
                "episode_index": episode_index,
                "camera_key": camera_key,
                "stride": stride,
                "fps": fps,
                "window_len": window_len,
                "start_index": start_idx,
                "original_frame_indices": original_frame_indices,
                "tasks": tasks,
                "start_image": (local_dataset_path / image_rel).as_posix(),
                "actions": (local_dataset_path / actions_rel).as_posix(),
                "action_sequence": action_sequence.astype(float).tolist(),
            }
            answer_row = {
                "sample_id": sample_id,
                "repo_id": repo_id,
                "episode_index": episode_index,
                "camera_key": camera_key,
                "answer_video": (local_dataset_path / answer_video_rel).as_posix(),
                "original_frame_indices": original_frame_indices,
            }
            write_json(challenge_meta_path, challenge_row)
            challenge_rows.append(challenge_row)
            answer_rows.append(answer_row)
            sample_counter += 1

    write_jsonl(challenge_root / "challenge_index.jsonl", challenge_rows)
    write_jsonl(answers_root / "answer_index.jsonl", answer_rows)
    return {
        "repo_id": repo_id,
        "challenge_path": (Path("challenge") / local_dataset_path).as_posix(),
        "answers_path": (Path("answers") / local_dataset_path).as_posix(),
        "camera_key": camera_key,
        "stride": stride,
        "fps": fps,
        "window_len": window_len,
        "samples": sample_counter,
        "skipped_short_episodes": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SmolVLA SO100 competition challenge inputs and answer videos.")
    parser.add_argument("--repo-id", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--output-root", default="/workspace/smolvla_eval_challenge_stride5")
    parser.add_argument("--cache-dir", default="/tmp/smolvla_eval_challenge_cache")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument("--samples-per-episode", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.window_len <= 0:
        raise ValueError("--window-len must be positive.")
    if args.samples_per_episode <= 0:
        raise ValueError("--samples-per-episode must be positive.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for repo_id in args.repo_id:
        print(f"[SmolVLA challenge preprocess] processing {repo_id}", flush=True)
        summaries.append(
            process_repo(
                repo_id=repo_id,
                output_root=output_root,
                cache_dir=args.cache_dir,
                token=args.hf_token,
                stride=args.stride,
                camera_key=args.camera_key,
                window_len=args.window_len,
                samples_per_episode=args.samples_per_episode,
                max_episodes=args.max_episodes,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        )

    write_json(output_root / "challenge_summary.json", {"datasets": summaries})
    print(json.dumps({"datasets": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()