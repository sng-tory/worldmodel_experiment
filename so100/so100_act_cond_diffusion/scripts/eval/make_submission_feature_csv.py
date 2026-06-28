from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.utils.train import get_model
from omegaconf import OmegaConf

from feature_csv_utils import (
    build_inference_batch,
    list_challenge_sample_ids,
    load_action_stats,
    save_video_tensor,
    write_feature_csv,
)


def load_diffusion_model(config_path: str, checkpoint_path: str, device: torch.device):
    eval_config = OmegaConf.load(config_path)
    model_config = OmegaConf.load(eval_config.model_config_file).model
    model = get_model(model_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return eval_config, model, DDIMSampler(model)


def generate_predictions(args, sample_ids: list[str], device: torch.device) -> None:
    eval_config, model, sampler = load_diffusion_model(args.config, args.checkpoint, device)
    ddim_kwargs = OmegaConf.to_container(eval_config.ddim_kwargs, resolve=True)
    if args.ddim_steps is not None:
        ddim_kwargs["ddim_steps"] = args.ddim_steps

    action_mean, action_std = load_action_stats(args.action_stats_path)
    if action_mean is not None:
        action_mean = action_mean.to(device)
        action_std = action_std.to(device)

    amp_context = torch.cuda.amp.autocast() if args.precision == 16 and device.type == "cuda" else nullcontext()
    challenge_root = Path(args.challenge_root)
    prediction_root = Path(args.prediction_root)

    for start in range(0, len(sample_ids), args.batch_size):
        batch_ids = sample_ids[start : start + args.batch_size]
        pred_paths = [prediction_root / f"{sample_id}.mp4" for sample_id in batch_ids]
        if not args.overwrite and all(path.exists() for path in pred_paths):
            continue

        batch = build_inference_batch(
            challenge_root,
            batch_ids,
            args.target_height,
            args.target_width,
            args.pad,
            args.fps,
            action_mean,
            action_std,
            device,
        )
        z, c, uc, cond_mask, _logs, kwargs = model.prepare_batch_for_inference(batch)
        sample_kwargs = dict(ddim_kwargs)
        sample_kwargs.update(kwargs)
        sample_steps = sample_kwargs.pop("ddim_steps")
        shape = (model.channels, model.temporal_length, *model.image_size)
        with torch.no_grad(), model.ema_scope("Submission Feature CSV"):
            with amp_context:
                samples, _ = sampler.sample(
                    sample_steps,
                    batch_size=z.shape[0],
                    shape=shape,
                    conditioning=c,
                    unconditional_conditioning=uc,
                    mask=cond_mask,
                    x0=z,
                    **sample_kwargs,
                )
            generated = model.decode_first_stage(samples)

        for sample_id, video in zip(batch_ids, generated):
            save_video_tensor(video, prediction_root / f"{sample_id}.mp4", args.fps)
        print(f"[generate] wrote predictions for {start + len(batch_ids)}/{len(sample_ids)} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions from challenge data and export submission feature CSV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--challenge-root", default="/workspace/smolvla_eval_challenge_stride5/challenge")
    parser.add_argument("--prediction-root", default="/workspace/smolvla_eval_predictions/videos")
    parser.add_argument("--output-csv", default="/workspace/smolvla_eval_predictions/submission_features.csv")
    parser.add_argument("--action-stats-path", default="/workspace/so100_stride5/so100_action_statistics.json")
    parser.add_argument("--action-extractor-ckpt", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--target-height", type=int, default=320)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--pad", action="store_true", default=True)
    parser.add_argument("--temporal-length", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--no-fvd", action="store_true")
    parser.add_argument("--no-dino", action="store_true")
    parser.add_argument("--no-fvd-pretrained", action="store_true")
    parser.add_argument("--dino-model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--dino-image-size", type=int, default=0)
    parser.add_argument("--no-dino-pretrained", action="store_true")
    parser.add_argument("--feature-precision", type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_ids = list_challenge_sample_ids(Path(args.challenge_root))

    if not args.skip_generation:
        generate_predictions(args, sample_ids, device)

    write_feature_csv(
        video_root=Path(args.prediction_root),
        sample_ids=sample_ids,
        output_csv=Path(args.output_csv),
        source="submission",
        device=device,
        temporal_length=args.temporal_length,
        target_height=args.target_height,
        target_width=args.target_width,
        pad=args.pad,
        feature_batch_size=args.feature_batch_size,
        precision=args.feature_precision,
        use_fid=not args.no_fid,
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
