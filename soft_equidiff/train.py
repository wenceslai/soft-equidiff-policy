"""
Training script for SoftEquiDiffPolicy on Push-T.

Usage:
    python -m soft_equidiff.train [OPTIONS]

    # Exact equivariance (baseline — recovers EquiDiff):
    python -m soft_equidiff.train --penalty_mode constant --lambda_base 1000.0 --run_name equidiff_exact

    # Constant soft penalty:
    python -m soft_equidiff.train --penalty_mode constant --lambda_base 0.1 --run_name soft_constant

    # Step-dependent soft penalty (novel):
    python -m soft_equidiff.train --penalty_mode step_dependent --lambda_base 0.1 --run_name soft_step

    # With camera tilt:
    python -m soft_equidiff.train --tilt_degrees 30 --run_name soft_step_tilt30

Requires:
    pip install lerobot escnn einops diffusers torch
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import SoftEquiDiffConfig
from .policy import SoftEquiDiffPolicy
from .camera_tilt import make_tilt_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="soft_equi_pusht")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Soft equivariance hyperparams
    p.add_argument("--penalty_mode", default="step_dependent", choices=["constant", "step_dependent"])
    p.add_argument("--lambda_base", type=float, default=0.1)
    p.add_argument("--lambda_power", type=float, default=1.0)

    # Which layers to soften
    p.add_argument("--no_soften_image", action="store_true") # image encoder
    p.add_argument("--no_soften_state", action="store_true") # proprio encoder actions and proprio have the same format (x, y)
    p.add_argument("--no_soften_action", action="store_true") # action encoder
    p.add_argument("--no_soften_decoder", action="store_true") # noise decoder

    # Training
    p.add_argument("--num_steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=50_000)

    # Camera tilt
    p.add_argument("--tilt_degrees", type=float, default=0.0)

    # Architecture
    p.add_argument("--N", type=int, default=8, help="C_N group order")
    p.add_argument("--n_hidden", type=int, default=64) # number of features per group element
    p.add_argument("--n_obs_steps", type=int, default=2)

    return p.parse_args()


def build_dataset(config: SoftEquiDiffConfig, tilt_transform=None):
    """Load LeRobot Push-T dataset."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError(
            "LeRobot not found. Install with:\n"
            "  git clone https://github.com/huggingface/lerobot && cd lerobot && pip install -e ."
        )

    dataset = LeRobotDataset(
        config.dataset_repo_id,
        delta_timestamps={
            "observation.image": [i / 10.0 for i in range(-config.n_obs_steps + 1, 1)],
            "observation.state": [i / 10.0 for i in range(-config.n_obs_steps + 1, 1)],
            "action": [i / 10.0 for i in range(config.horizon)],
        },
        image_transforms=tilt_transform,
    )

    # Build normaliser stats from dataset
    stats = {
        "observation.state": {
            "min": dataset.meta.stats["observation.state"]["min"],
            "max": dataset.meta.stats["observation.state"]["max"],
        },
        "action": {
            "min": dataset.meta.stats["action"]["min"],
            "max": dataset.meta.stats["action"]["max"],
        },
    }

    return dataset, stats


def train(args):
    device = torch.device(args.device)
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    config = SoftEquiDiffConfig(
        n_rotations=args.N,
        enc_n_hidden=args.n_hidden,
        n_obs_steps=args.n_obs_steps,
        penalty_mode=args.penalty_mode,
        lambda_base=args.lambda_base,
        lambda_power=args.lambda_power,
        soften_image_encoder=not args.no_soften_image,
        soften_state_encoder=not args.no_soften_state,
        soften_action_encoder=not args.no_soften_action,
        soften_decoder=not args.no_soften_decoder,
        num_train_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        camera_tilt_degrees=args.tilt_degrees,
    )

    # Camera tilt transform (applied to images in the dataset)
    tilt_transform = make_tilt_transform(args.tilt_degrees) if args.tilt_degrees > 0 else None

    print(f"Loading dataset: {config.dataset_repo_id}")
    dataset, stats = build_dataset(config, tilt_transform)

    policy = SoftEquiDiffPolicy(config, dataset_stats=stats).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(dataloader) # 

    print(f"Starting training: {args.run_name}")
    print(f"  Mode: {config.penalty_mode}, λ={config.lambda_base}, tilt={config.camera_tilt_degrees}°")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {n_params:,}")

    policy.train()
    t0 = time.time()

    for step in range(1, config.num_train_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        optimizer.zero_grad()
        losses = policy(batch)
        loss = losses["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step:>7d} | loss {losses['loss'].item():.4f} "
                f"| mse {losses['mse_loss'].item():.4f} "
                f"| equi {losses['equi_penalty'].item():.4f} "
                f"| λ {losses['lambda'].item():.4f} "
                f"| {elapsed:.0f}s"
            )

        if step % args.save_every == 0 or step == config.num_train_steps:
            ckpt_path = out_dir / f"policy_step{step:07d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    train(parse_args())
