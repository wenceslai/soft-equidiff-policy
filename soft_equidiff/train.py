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

    # Disable wandb:
    python -m soft_equidiff.train --no_wandb

Requires:
    pip install lerobot escnn einops diffusers torch wandb
"""

import argparse
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
    p.add_argument("--no_soften_image", action="store_true")  # image encoder
    p.add_argument("--no_soften_state", action="store_true")  # proprio encoder (x, y)
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
    p.add_argument("--n_hidden", type=int, default=64)  # features per group element
    p.add_argument("--n_obs_steps", type=int, default=2)

    # Wandb
    p.add_argument("--wandb_project", default="soft-equidiff-pusht")
    p.add_argument("--wandb_entity", default=None, help="wandb team/username (optional)")
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")

    return p.parse_args()


def build_dataset(config: SoftEquiDiffConfig, tilt_transform=None):
    """Load LeRobot Push-T dataset."""
    try:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
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


def _grad_norm(policy):
    """Compute total gradient L2 norm across all parameters."""
    total = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


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

    # --- Wandb init ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config={
                    "penalty_mode": config.penalty_mode,
                    "lambda_base": config.lambda_base,
                    "lambda_power": config.lambda_power,
                    "N": config.n_rotations,
                    "enc_n_hidden": config.enc_n_hidden,
                    "state_features": config.state_features,
                    "action_features": config.action_features,
                    "n_obs_steps": config.n_obs_steps,
                    "horizon": config.horizon,
                    "num_diffusion_steps": config.num_diffusion_steps,
                    "soften_image": config.soften_image_encoder,
                    "soften_state": config.soften_state_encoder,
                    "soften_action": config.soften_action_encoder,
                    "soften_decoder": config.soften_decoder,
                    "tilt_degrees": config.camera_tilt_degrees,
                    "lr": config.lr,
                    "batch_size": config.batch_size,
                    "unet_down_dims": config.unet_down_dims,
                },
                dir=str(out_dir),
            )
        except ImportError:
            print("wandb not installed — run `pip install wandb` to enable logging. Continuing without it.")
            use_wandb = False

    # Camera tilt transform
    tilt_transform = make_tilt_transform(args.tilt_degrees) if args.tilt_degrees > 0 else None

    print(f"Loading dataset: {config.dataset_repo_id}")
    dataset, stats = build_dataset(config, tilt_transform)

    policy = SoftEquiDiffPolicy(config, dataset_stats=stats).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Starting training: {args.run_name}")
    print(f"  Mode: {config.penalty_mode}, λ={config.lambda_base}, tilt={config.camera_tilt_degrees}°")
    print(f"  Parameters: {n_params:,}")

    if use_wandb:
        wandb.summary["n_params"] = n_params

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(dataloader)

    policy.train()
    t0 = time.time()
    step_t0 = t0

    for step in range(1, config.num_train_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        optimizer.zero_grad()
        losses = policy(batch)
        losses["loss"].backward()

        grad_norm = _grad_norm(policy)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = args.log_every / (time.time() - step_t0)
            step_t0 = time.time()

            print(
                f"step {step:>7d} | loss {losses['loss'].item():.4f} "
                f"| mse {losses['mse_loss'].item():.4f} "
                f"| equi {losses['equi_penalty'].item():.4f} "
                f"| λ {losses['lambda'].item():.4f} "
                f"| grad {grad_norm:.3f} "
                f"| {steps_per_sec:.1f} s/s "
                f"| {elapsed:.0f}s elapsed"
            )

            if use_wandb:
                wandb.log({
                    "train/loss":         losses["loss"].item(),
                    "train/mse_loss":     losses["mse_loss"].item(),
                    "train/equi_penalty": losses["equi_penalty"].item(),
                    "train/lambda":       losses["lambda"].item(),
                    "train/grad_norm":    grad_norm,
                    "train/steps_per_sec": steps_per_sec,
                    "train/elapsed_sec":  elapsed,
                }, step=step)

        if step % args.save_every == 0 or step == config.num_train_steps:
            ckpt_path = out_dir / f"policy_step{step:07d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "dataset_stats": stats,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"{args.run_name}-checkpoint",
                    type="model",
                    metadata={"step": step},
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)

    if use_wandb:
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    train(parse_args())
