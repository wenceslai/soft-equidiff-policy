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

from .augmentation import C4Augmentation
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
    p.add_argument("--weight_decay", type=float, default=1e-6,
                   help="AdamW weight decay (default 1e-6; try 1e-4 or 1e-3 to combat overfitting)")
    p.add_argument("--mixed_precision", action="store_true",
                   help="Use bfloat16 autocast for forward/backward (2-4x speedup on L40S/A100/H100)")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=25_000)
    p.add_argument("--val_every", type=int, default=10_000,
                   help="Compute and log validation MSE loss every N steps")
    p.add_argument("--val_batches", type=int, default=32,
                   help="Number of val batches to average over per validation pass")
    p.add_argument("--resume", default=None, metavar="CHECKPOINT",
                   help="Path to a .pt checkpoint to resume training from")

    # Camera tilt
    p.add_argument("--tilt_degrees", type=float, default=0.0)

    # Architecture
    p.add_argument("--N", type=int, default=8, help="C_N group order")
    p.add_argument("--n_hidden", type=int, default=64)  # features per group element
    p.add_argument("--n_obs_steps", type=int, default=2) # how many past observation frames to stack
    p.add_argument("--unet_down_dims", type=int, nargs="+", default=[256, 512, 1024],
                   help="U-Net encoder channel widths, e.g. --unet_down_dims 64 128 256")

    # Rotation augmentation
    p.add_argument("--rot_aug", action="store_true",
                   help="Apply C₄ rotation augmentation (rotate image+state+action together). "
                        "Use with --N 4 so the model group order matches the augmentation.")

    # Dataset
    p.add_argument("--video_backend", default="pyav", choices=["pyav", "torchcodec"],
                   help="LeRobot video decode backend (default: pyav, works everywhere)")

    # Wandb
    p.add_argument("--wandb_project", default="soft-equidiff-pusht")
    p.add_argument("--wandb_entity", default=None, help="wandb team/username (optional)")
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")

    return p.parse_args()


def build_dataset(
    config: SoftEquiDiffConfig,
    tilt_transform=None,
    video_backend: str = "pyav",
    val_fraction: float = 0.1,
):
    """
    Load LeRobot Push-T dataset and split into train / val by episode index.

    The last `val_fraction` of episodes (sorted by index) become the val set.
    Stats are derived from the full dataset so the normalizer is stable regardless
    of split size.

    Returns:
        train_dataset, val_dataset, stats
    """
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

    delta_ts = {
        "observation.image": [i / 10.0 for i in range(-config.n_obs_steps + 1, 1)],
        "observation.state": [i / 10.0 for i in range(-config.n_obs_steps + 1, 1)],
        "action":            [i / 10.0 for i in range(config.horizon)],
    }

    # Load full dataset once — cheap, only reads metadata and caches video index.
    full_dataset = LeRobotDataset(
        config.dataset_repo_id,
        delta_timestamps=delta_ts,
        video_backend=video_backend,
    )

    total_episodes = full_dataset.meta.total_episodes
    n_val = max(1, round(total_episodes * val_fraction))
    n_train = total_episodes - n_val
    train_eps = list(range(n_train))
    val_eps   = list(range(n_train, total_episodes))
    print(f"  Dataset split: {n_train} train episodes / {n_val} val episodes "
          f"(total {total_episodes})")

    # Stats from the full dataset (negligible leakage; avoids skewed normalization
    # from a small val subset).
    stats = {
        "observation.state": {
            "min": full_dataset.meta.stats["observation.state"]["min"],
            "max": full_dataset.meta.stats["observation.state"]["max"],
        },
        "action": {
            "min": full_dataset.meta.stats["action"]["min"],
            "max": full_dataset.meta.stats["action"]["max"],
        },
    }

    # Equivariant normalization: x and y must use the same scale so that
    # normalized vectors remain proper 2D Euclidean vectors under irrep(1).
    # Per-component min-max would distort rotations if x_range ≠ y_range.
    for key in ["observation.state", "action"]:
        lo = torch.as_tensor(stats[key]["min"]).float().clone()
        hi = torch.as_tensor(stats[key]["max"]).float().clone()
        max_range = (hi - lo).max()
        centers   = (lo + hi) / 2.0
        stats[key]["min"] = centers - max_range / 2.0
        stats[key]["max"] = centers + max_range / 2.0

    # Train split — with image augmentation (tilt transform if any).
    train_dataset = LeRobotDataset(
        config.dataset_repo_id,
        delta_timestamps=delta_ts,
        episodes=train_eps,
        image_transforms=tilt_transform,
        video_backend=video_backend,
    )
    # Val split — no image augmentation so loss is comparable across runs.
    val_dataset = LeRobotDataset(
        config.dataset_repo_id,
        delta_timestamps=delta_ts,
        episodes=val_eps,
        video_backend=video_backend,
    )

    return train_dataset, val_dataset, stats


@torch.no_grad()
def _compute_val_loss(policy, val_loader, device, n_batches: int, use_amp: bool = False) -> float:
    """
    Average MSE loss over up to `n_batches` batches from the val dataloader.
    Switches policy to eval mode and back to train mode afterwards.
    """
    policy.eval()
    total_mse = 0.0
    count = 0
    for batch in val_loader:
        if count >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            losses = policy(batch)
        total_mse += losses["mse_loss"].item()
        count += 1
    policy.train()
    return total_mse / max(count, 1)


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
        unet_down_dims=tuple(args.unet_down_dims),
        num_train_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
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
                    "weight_decay": config.weight_decay,
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
    train_dataset, val_dataset, stats = build_dataset(
        config, tilt_transform, video_backend=args.video_backend
    )

    # C₄ rotation augmentation — instantiated once, applied per batch on GPU.
    # Workspace center is derived from the isotropic-normalised stats so that
    # the rotation pivot matches the geometric centre of the coordinate space.
    rot_aug = None
    if args.rot_aug:
        # Use the geometric center of the workspace (256, 256), which corresponds to
        # the image center (47.5, 47.5) in the 96×96 rendering of the 512×512 workspace.
        # The stats-derived center is biased by where the agent spends time (~272 in y),
        # not by the image rotation pivot — using it would misalign image vs. coordinate rotations.
        ws_center = torch.tensor([256.0, 256.0])
        rot_aug = C4Augmentation(ws_center)
        print(f"  C₄ rotation augmentation enabled  (workspace center: {ws_center.tolist()})")

    policy = SoftEquiDiffPolicy(config, dataset_stats=stats).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_step = 1
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"  Resuming from step {ckpt['step']}")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Starting training: {args.run_name}")
    print(f"  Mode: {config.penalty_mode}, λ={config.lambda_base}, tilt={config.camera_tilt_degrees}°")
    print(f"  Parameters: {n_params:,}")
    print(f"  Val loss every {args.val_every} steps ({args.val_batches} batches × {config.batch_size})")

    if use_wandb:
        wandb.summary["n_params"] = n_params

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,      # shuffle so each val pass samples different frames
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    data_iter = iter(dataloader)
    use_amp = args.mixed_precision and device.type == "cuda"
    if use_amp:
        print("  Mixed precision: bfloat16 autocast enabled")

    policy.train()
    t0 = time.time()
    step_t0 = t0

    _t_data_total = 0.0
    _t_fwd_total = 0.0

    for step in range(start_step, config.num_train_steps + 1):
        _t = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            print("DataLoader ran out of data, restarting...")
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        _t_data_total += time.time() - _t

        if rot_aug is not None:
            batch = rot_aug(batch)

        _t = time.time()
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            losses = policy(batch)
        losses["loss"].backward()
        _t_fwd_total += time.time() - _t

        grad_norm = _grad_norm(policy)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = args.log_every / (time.time() - step_t0)
            step_t0 = time.time()

            _pct_data = 100 * _t_data_total / (_t_data_total + _t_fwd_total + 1e-9)
            print(
                f"step {step:>7d} | loss {losses['loss'].item():.4f} "
                f"| mse {losses['mse_loss'].item():.4f} "
                f"| equi {losses['equi_penalty'].item():.4f} "
                f"| λ {losses['lambda'].item():.4f} "
                f"| grad {grad_norm:.3f} "
                f"| {steps_per_sec:.1f} steps/s "
                f"| data {_t_data_total:.1f}s  fwd {_t_fwd_total:.1f}s  ({_pct_data:.0f}% data)"
                f"| {elapsed:.0f}s elapsed"
            )
            _t_data_total = 0.0
            _t_fwd_total = 0.0

            if use_wandb:
                log_dict = {
                    "train/loss":         losses["loss"].item(),
                    "train/mse_loss":     losses["mse_loss"].item(),
                    "train/equi_penalty": losses["equi_penalty"].item(),
                    "train/lambda":       losses["lambda"].item(),
                    "train/grad_norm":    grad_norm,
                    "train/steps_per_sec": steps_per_sec,
                    "train/elapsed_sec":  elapsed,
                }
                # Degeneracy diagnostics: read side-channel written by model.forward()
                # debug/unet_inter_group_var ≈ 0 → symmetric collapse (decoder outputs ≈ 0)
                if hasattr(policy.model, "_last_diagnostics"):
                    log_dict.update(policy.model._last_diagnostics)
                wandb.log(log_dict, step=step)

        if step % args.val_every == 0:
            val_mse = _compute_val_loss(policy, val_loader, device, n_batches=args.val_batches, use_amp=use_amp)
            print(f"step {step:>7d} | val/mse_loss {val_mse:.4f}")
            if use_wandb:
                wandb.log({"val/mse_loss": val_mse}, step=step)

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
