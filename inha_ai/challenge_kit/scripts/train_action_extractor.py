
import argparse
from pathlib import Path

import torch
from lvdm.utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SO-100 action extractor on real videos.")
    parser.add_argument("--config", default="configs/train/so100_action_extractor.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    seed_everything(seed)

    data = instantiate_from_config(config.data)
    data.setup()
    model = instantiate_from_config(config.model)
    trainable_params = count_parameters(model)
    print(f"[action extractor] trainable params: {trainable_params / 1e6:.2f}M")

    lightning = config.get("lightning", OmegaConf.create())
    trainer_cfg = OmegaConf.to_container(lightning.get("trainer", OmegaConf.create()), resolve=True)
    logger = instantiate_from_config(lightning.logger) if "logger" in lightning else True
    if hasattr(logger, "log_hyperparams"):
        hparams = OmegaConf.to_container(config, resolve=True)
        hparams["trainable_params"] = trainable_params
        logger.log_hyperparams(hparams)
    watch_cfg = lightning.get("watch_model", OmegaConf.create())
    if getattr(watch_cfg, "enabled", False) and hasattr(logger, "watch"):
        logger.watch(
            model,
            log=getattr(watch_cfg, "log", "gradients"),
            log_freq=int(getattr(watch_cfg, "log_freq", 200)),
        )
    callbacks = []
    if "callbacks" in lightning:
        for callback_cfg in lightning.callbacks.values():
            callbacks.append(instantiate_from_config(callback_cfg))

    ckpt_dir = trainer_cfg.get("default_root_dir") or config.get("logdir", None)
    if ckpt_dir:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(logger=logger, callbacks=callbacks, **trainer_cfg)
    trainer.fit(model, datamodule=data, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
