from __future__ import annotations

import argparse
from pathlib import Path

import torch

from feature_csv_utils import list_challenge_sample_ids, list_video_sample_ids, write_feature_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export answer feature CSV from answer videos.")
    parser.add_argument("--answer-video-root", default="../data/answers/videos")
    parser.add_argument("--challenge-root", default="../data/challenge")
    parser.add_argument("--output-csv", default="../csv/answer_features.csv")
    parser.add_argument("--action-extractor-ckpt", default="../checkpoints/action_extractor.ckpt")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--target-height", type=int, default=320)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--pad", action="store_true", default=True)
    parser.add_argument("--temporal-length", type=int, default=16)
    parser.add_argument("--include-fid", action="store_true")
    parser.add_argument("--no-fvd", action="store_true")
    parser.add_argument("--no-dino", action="store_true")
    parser.add_argument("--no-fvd-pretrained", action="store_true")
    parser.add_argument("--dino-model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--dino-image-size", type=int, default=0)
    parser.add_argument("--no-dino-pretrained", action="store_true")
    parser.add_argument("--feature-precision", type=int, default=6)
    args = parser.parse_args()

    answer_root = Path(args.answer_video_root)
    restrict_to = None
    if args.challenge_root:
        restrict_to = set(list_challenge_sample_ids(Path(args.challenge_root)))
    sample_ids = list_video_sample_ids(answer_root, restrict_to=restrict_to)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    write_feature_csv(
        video_root=answer_root,
        sample_ids=sample_ids,
        output_csv=Path(args.output_csv),
        source="answer",
        device=device,
        temporal_length=args.temporal_length,
        target_height=args.target_height,
        target_width=args.target_width,
        pad=args.pad,
        feature_batch_size=args.feature_batch_size,
        precision=args.feature_precision,
        use_fid=args.include_fid,
        use_fvd=not args.no_fvd,
        use_dino=not args.no_dino,
        fvd_pretrained=not args.no_fvd_pretrained,
        dino_model_name=args.dino_model,
        dino_pretrained=not args.no_dino_pretrained,
        dino_image_size=args.dino_image_size,
        action_extractor_ckpt=args.action_extractor_ckpt,
    )
    print(f"[feature csv] saved to {args.output_csv}")


if __name__ == "__main__":
    main()
