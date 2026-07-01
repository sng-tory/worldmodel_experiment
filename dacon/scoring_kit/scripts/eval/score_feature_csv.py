from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


EPS = 1e-9
DINO_COMPONENT = "DINO Component"
VIDEO_COMPONENT = "Video Feature Component"
ACTION_COMPONENT = "Action Component"


def _as_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def load_feature_csv(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load feature CSVs produced by make_submission/answer_feature_csv.py.

    The compact format has one row per sample/component. This loader tolerates
    frame-wise duplicate sample/component rows by stacking them.
    """
    buckets: dict[str, dict[str, list[np.ndarray]]] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "feature_component", "feature_json"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}; got {reader.fieldnames}.")
        for row in reader:
            sample_id = row["sample_id"]
            component = row["feature_component"].strip()
            feature = _as_array(json.loads(row["feature_json"]))
            buckets.setdefault(sample_id, {}).setdefault(component, []).append(feature)

    features: dict[str, dict[str, np.ndarray]] = {}
    for sample_id, component_map in buckets.items():
        features[sample_id] = {}
        for component, values in component_map.items():
            features[sample_id][component] = values[0] if len(values) == 1 else np.stack(values, axis=0)
    return features


def cosine_distance(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    a = _as_array(a)
    b = _as_array(b)
    numerator = np.sum(a * b, axis=axis)
    denom = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + EPS
    return 1.0 - numerator / denom


def feature_distance(a: np.ndarray, b: np.ndarray, mode: str) -> float:
    a = _as_array(a)
    b = _as_array(b)
    if a.shape != b.shape:
        raise ValueError(f"Feature shape mismatch: {a.shape} vs {b.shape}")
    if mode == "cosine":
        if a.ndim == 1:
            return float(cosine_distance(a, b))
        return float(np.mean(cosine_distance(a, b, axis=-1)))
    if mode == "rmse":
        return float(np.sqrt(np.mean((a - b) ** 2)))
    raise ValueError(f"Unknown distance mode: {mode}")


def normalize_distance(distance: float, mode: str) -> float:
    if math.isnan(distance):
        return math.nan
    if mode == "cosine":
        return float(np.clip(distance / 2.0, 0.0, 1.0))
    return distance


def load_action_stats(path: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not path:
        return None, None
    with Path(path).open("r", encoding="utf-8") as f:
        stats = json.load(f)
    return _as_array(stats["mean"]), _as_array(stats["std"])


def load_target_action(challenge_root: Path, sample_id: str, action_mean: np.ndarray | None, action_std: np.ndarray | None) -> np.ndarray:
    target = np.load(challenge_root / "actions" / f"{sample_id}.npy").astype(np.float64)
    if action_mean is not None and action_std is not None:
        target = (target - action_mean) / action_std
    return target.reshape(-1)


def action_mae_from_feature(feature: np.ndarray, target: np.ndarray | None) -> float:
    feature = _as_array(feature).reshape(-1)
    if feature.size == 1:
        return float(feature[0])
    if target is None:
        raise ValueError("Action feature is a prediction vector, so --challenge-root is required to compute MAE.")
    if feature.shape != target.shape:
        raise ValueError(f"Action feature and target shape mismatch: {feature.shape} vs {target.shape}")
    return float(np.mean(np.abs(feature - target)))


def action_component(ratio: float, mode: str, ratio_cap: float) -> float:
    if math.isnan(ratio):
        return math.nan
    if mode == "ratio_minus_one":
        penalty = max(0.0, ratio - 1.0)
        if ratio_cap > 1.0:
            return float(np.clip(penalty / (ratio_cap - 1.0), 0.0, 1.0))
        return penalty
    if mode == "ratio":
        return float(np.clip(ratio / ratio_cap, 0.0, 1.0)) if ratio_cap > 0 else ratio
    raise ValueError(f"Unknown action component mode: {mode}")


def weighted_score(components: dict[str, float], weights: dict[str, float], allow_missing: bool) -> float:
    numerator = 0.0
    denominator = 0.0
    missing = []
    for name, weight in weights.items():
        if weight <= 0:
            continue
        value = components.get(name, math.nan)
        if math.isnan(value):
            missing.append(name)
            continue
        numerator += weight * value
        denominator += weight
    if missing and not allow_missing:
        raise ValueError(f"Missing score components with nonzero weights: {missing}")
    if denominator <= 0:
        raise ValueError("No valid weighted components were available.")
    return numerator / denominator


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_sample_id(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return f"sample_{int(token):06d}"
    return token


def parse_sample_id_text(text: str) -> set[str]:
    sample_ids = set()
    for chunk in text.replace(",", "\n").splitlines():
        sample_id = normalize_sample_id(chunk)
        if sample_id:
            sample_ids.add(sample_id)
    return sample_ids


def load_public_sample_ids(public_sample_ids: str | None, public_sample_id_file: str | None) -> set[str]:
    sample_ids = set()
    if public_sample_ids:
        sample_ids.update(parse_sample_id_text(public_sample_ids))
    if public_sample_id_file:
        sample_ids.update(parse_sample_id_text(Path(public_sample_id_file).read_text(encoding="utf-8")))
    return sample_ids


def safe_mean(rows: list[dict], key: str) -> float:
    if not rows:
        return math.nan
    values = [row[key] for row in rows]
    if all(math.isnan(value) for value in values):
        return math.nan
    return float(np.nanmean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score submission feature CSV against answer feature CSV.")
    parser.add_argument("--submission-csv", required=True)
    parser.add_argument("--answer-csv", default="../private_csv/answer_features.csv")
    parser.add_argument("--challenge-root", default=None)
    parser.add_argument("--action-stats-path", default=None)
    parser.add_argument("--details-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--distance", choices=["cosine", "rmse"], default="cosine")
    parser.add_argument("--weight-dino", type=float, default=0.3)
    parser.add_argument("--weight-video", type=float, default=0.3)
    parser.add_argument("--weight-action", type=float, default=0.4)
    parser.add_argument("--action-component", choices=["ratio_minus_one", "ratio"], default="ratio_minus_one")
    parser.add_argument("--action-ratio-cap", type=float, default=5.0)
    parser.add_argument("--allow-missing-action", action="store_true")
    parser.add_argument("--public-sample-ids", default=None, help="Comma/newline separated public sample ids.")
    parser.add_argument("--public-sample-id-file", default="../private_csv/public_ids.txt", help="Text file containing public sample ids.")
    args = parser.parse_args()

    submission = load_feature_csv(args.submission_csv)
    answer = load_feature_csv(args.answer_csv)
    sample_ids = sorted(set(submission) & set(answer))
    if not sample_ids:
        raise ValueError("No common sample_id values between submission and answer CSVs.")
    sample_id_set = set(sample_ids)
    public_sample_ids = load_public_sample_ids(args.public_sample_ids, args.public_sample_id_file)
    unknown_public_ids = sorted(public_sample_ids - sample_id_set)
    if unknown_public_ids:
        print(f"[score] warning: ignoring {len(unknown_public_ids)} public ids not found in common samples: {unknown_public_ids[:10]}")
    public_sample_ids &= sample_id_set

    dino_component_name = DINO_COMPONENT
    video_component_name = VIDEO_COMPONENT
    action_component_name = ACTION_COMPONENT

    action_mean, action_std = load_action_stats(args.action_stats_path) if args.challenge_root else (None, None)
    challenge_root = Path(args.challenge_root) if args.challenge_root else None
    weights = {"dino": args.weight_dino, "video": args.weight_video, "action": args.weight_action}

    rows = []
    for sample_id in sample_ids:
        dino_distance = math.nan
        dino_norm = math.nan
        if dino_component_name in submission[sample_id] and dino_component_name in answer[sample_id]:
            dino_distance = feature_distance(
                submission[sample_id][dino_component_name],
                answer[sample_id][dino_component_name],
                args.distance,
            )
            dino_norm = normalize_distance(dino_distance, args.distance)

        video_distance = math.nan
        video_norm = math.nan
        if video_component_name in submission[sample_id] and video_component_name in answer[sample_id]:
            submission_video = submission[sample_id][video_component_name].reshape(-1)
            answer_video = answer[sample_id][video_component_name].reshape(-1)
            video_distance = feature_distance(submission_video, answer_video, args.distance)
            video_norm = normalize_distance(video_distance, args.distance)

        real_action_mae = math.nan
        generated_action_mae = math.nan
        if action_component_name in submission[sample_id] and action_component_name in answer[sample_id]:
            target = None
            if challenge_root is not None:
                target = load_target_action(challenge_root, sample_id, action_mean, action_std)
            real_action_mae = action_mae_from_feature(answer[sample_id][action_component_name], target)
            generated_action_mae = action_mae_from_feature(submission[sample_id][action_component_name], target)

        rows.append(
            {
                "sample_id": sample_id,
                "split": "public" if sample_id in public_sample_ids else "private",
                "dino_distance": dino_distance,
                "dino_component": dino_norm,
                "video_feature_distance": video_distance,
                "video_feature_component": video_norm,
                "real_action_mae": real_action_mae,
                "generated_action_mae": generated_action_mae,
                "action_error_ratio": math.nan,
                "action_component": math.nan,
                "weighted_score": math.nan,
            }
        )

    def summarize_split(split_rows: list[dict]) -> dict:
        mean_real_action_mae = safe_mean(split_rows, "real_action_mae")
        mean_generated_action_mae = safe_mean(split_rows, "generated_action_mae")
        if math.isnan(mean_real_action_mae) or math.isnan(mean_generated_action_mae):
            ratio = math.nan
            action_norm = math.nan
        else:
            ratio = mean_generated_action_mae / (mean_real_action_mae + EPS)
            action_norm = action_component(ratio, args.action_component, args.action_ratio_cap)
        components = {
            "dino": safe_mean(split_rows, "dino_component"),
            "video": safe_mean(split_rows, "video_feature_component"),
            "action": action_norm,
        }
        score = weighted_score(components, weights, allow_missing=args.allow_missing_action)
        return {
            "score": score,
            "mean_dino_distance": safe_mean(split_rows, "dino_distance"),
            "mean_dino_component": components["dino"],
            "mean_video_feature_distance": safe_mean(split_rows, "video_feature_distance"),
            "mean_video_feature_component": components["video"],
            "mean_real_action_mae": mean_real_action_mae,
            "mean_generated_action_mae": mean_generated_action_mae,
            "action_error_ratio": ratio,
            "action_component": action_norm,
        }

    public_rows = [row for row in rows if row["split"] == "public"]
    private_rows = [row for row in rows if row["split"] == "private"]
    all_summary = summarize_split(rows)
    public_summary = summarize_split(public_rows) if public_rows else {"score": math.nan}
    private_summary = summarize_split(private_rows) if private_rows else {"score": math.nan}

    split_summaries = {"public": public_summary, "private": private_summary}
    for row in rows:
        split_summary = split_summaries[row["split"]]
        row["action_error_ratio"] = split_summary.get("action_error_ratio", math.nan)
        row["action_component"] = split_summary.get("action_component", math.nan)
        components = {
            "dino": row["dino_component"],
            "video": row["video_feature_component"],
            "action": row["action_component"],
        }
        row["weighted_score"] = weighted_score(components, weights, allow_missing=args.allow_missing_action)

    if args.details_csv:
        write_csv(args.details_csv, rows)

    summary = {
        "num_samples": len(rows),
        "num_public_samples": len(public_rows),
        "num_private_samples": len(private_rows),
        "final_score": all_summary["score"],
        "public_score": public_summary["score"],
        "private_score": private_summary["score"],
        "mean_dino_distance": all_summary["mean_dino_distance"],
        "mean_dino_component": all_summary["mean_dino_component"],
        "mean_video_feature_distance": all_summary["mean_video_feature_distance"],
        "mean_video_feature_component": all_summary["mean_video_feature_component"],
        "mean_real_action_mae": all_summary["mean_real_action_mae"],
        "mean_generated_action_mae": all_summary["mean_generated_action_mae"],
        "mean_action_error_ratio": all_summary["action_error_ratio"],
        "mean_action_component": all_summary["action_component"],
        "public_action_error_ratio": public_summary.get("action_error_ratio", math.nan),
        "private_action_error_ratio": private_summary.get("action_error_ratio", math.nan),
        "weight_dino": args.weight_dino,
        "weight_video": args.weight_video,
        "weight_action": args.weight_action,
        "dino_feature_component": dino_component_name,
        "video_feature_component_name": video_component_name,
        "action_feature_component": action_component_name,
        "distance": args.distance,
        "action_component_mode": args.action_component,
        "action_ratio_cap": args.action_ratio_cap,
    }
    if args.summary_csv:
        write_csv(args.summary_csv, [summary])
    print(f"public_score={summary['public_score']:.8f}")
    print(f"private_score={summary['private_score']:.8f}")


if __name__ == "__main__":
    main()
