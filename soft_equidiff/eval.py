"""
Evaluation and analysis tools for SoftEquiDiffPolicy.

Two main functions:

1. measure_equivariance_error(policy, dataset, k, n_samples)
   — For a fixed denoising step k, compute the average equivariance error:
       ||ε(g·obs, g·a^k, k) - g·ε(obs, a^k, k)||
   across random group elements g.  This should be near 0 for an exactly
   equivariant model and larger for a model that has relaxed equivariance.

2. plot_equivariance_vs_step(checkpoints, dataset, steps, ...)
   — Sweep over multiple denoising steps and plot the equivariance error curve.
   For SoftEqui-step, we expect the error to be larger at small k (near-clean)
   and smaller at large k (near-noisy).

Usage:
    python -m soft_equidiff.eval --checkpoint outputs/soft_step/policy_step0200000.pt \
                                  --n_samples 200 --device cuda

    # Compare multiple runs and log to the same wandb project:
    python -m soft_equidiff.eval \\
        --checkpoint outputs/soft_step/policy_step0200000.pt \\
        --wandb_project soft-equidiff-pusht --wandb_run_name eval_comparison
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import SoftEquiDiffConfig
from .policy import SoftEquiDiffPolicy


# ---------------------------------------------------------------------------
# Rotation utilities (C_N group actions on observations and actions)
# ---------------------------------------------------------------------------

def rotate_image(image: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rotate a (*, 3, H, W) image tensor by angle_rad around the image centre."""
    import torchvision.transforms.functional as TF
    import math
    angle_deg = math.degrees(angle_rad)
    orig_shape = image.shape
    img = image.reshape(-1, *orig_shape[-3:])
    rotated = torch.stack([TF.rotate(img[i], angle_deg) for i in range(img.shape[0])])
    return rotated.reshape(orig_shape)


def rotate_action(action: torch.Tensor, angle_rad: float, center: tuple = (256.0, 256.0)) -> torch.Tensor:
    """Rotate 2D (x, y) actions by angle_rad around a given centre."""
    import math
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = center
    x = action[..., 0] - cx
    y = action[..., 1] - cy
    rx = cos_a * x - sin_a * y + cx
    ry = sin_a * x + cos_a * y + cy
    return torch.stack([rx, ry], dim=-1)


def rotate_state(state: torch.Tensor, angle_rad: float, center: tuple = (256.0, 256.0)) -> torch.Tensor:
    """Same as rotate_action (state is also a 2D position)."""
    return rotate_action(state, angle_rad, center)


# ---------------------------------------------------------------------------
# Equivariance error measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_equivariance_error(
    policy: SoftEquiDiffPolicy,
    dataset,
    k: int,
    n_samples: int = 100,
    device: str = "cpu",
    center: tuple = (256.0, 256.0),
) -> float:
    """
    Measure ||ε(g·obs, g·a^k, k) - g·ε(obs, a^k, k)|| averaged over samples and group elements.

    Returns:
        mean equivariance error (float)
    """
    import math
    from torch.utils.data import DataLoader

    policy.eval()
    N = policy.config.n_rotations
    angles = [2 * math.pi * n / N for n in range(N)]

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    errors = []

    for i, batch in enumerate(loader):
        if i >= n_samples:
            break

        batch = {key: val.to(device) for key, val in batch.items() if isinstance(val, torch.Tensor)}
        obs_images, obs_state, actions = policy._preprocess_batch(batch)

        noise = torch.randn_like(actions)
        timestep = torch.tensor([k], device=device)
        noisy_actions = policy.noise_scheduler.add_noise(actions, noise, timestep)

        eps_orig = policy.model(obs_images, obs_state, noisy_actions, timestep)

        batch_errors = []
        for angle in angles[1:]:  # skip identity
            obs_rot    = rotate_image(obs_images, angle)
            state_rot  = rotate_state(obs_state, angle, center)
            noisy_rot  = rotate_action(noisy_actions, angle, center)
            eps_rot    = policy.model(obs_rot, state_rot, noisy_rot, timestep)

            # Equivariance: ε(g·x) should equal g·ε(x)
            # Noise lives in displacement space so rotate around origin
            eps_expected = rotate_action(eps_orig, angle, center=(0.0, 0.0))
            err = (eps_rot - eps_expected).norm(dim=-1).mean().item()
            batch_errors.append(err)

        errors.append(float(np.mean(batch_errors)))

    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Free-weight norm analysis
# ---------------------------------------------------------------------------

def analyze_free_weights(policy: SoftEquiDiffPolicy) -> dict:
    """
    Collect ||W_free||² for every SoftEquiWrapper layer.

    Returns:
        dict mapping layer name → norm² value
    """
    from .model.soft_wrapper import SoftEquiWrapper

    norms = {}
    total = 0.0
    print("\n--- Free weight norms (||W_free||²) ---")
    for name, module in policy.model.named_modules():
        if isinstance(module, SoftEquiWrapper):
            norm_sq = module.free_weight_norm_sq().item()
            norms[name] = norm_sq
            total += norm_sq
            print(f"  {name:<60s}  {norm_sq:.6f}")
    norms["__total__"] = total
    print(f"  {'TOTAL':<60s}  {total:.6f}")
    return norms


# ---------------------------------------------------------------------------
# Multi-checkpoint comparison + wandb logging
# ---------------------------------------------------------------------------

def plot_equivariance_vs_step(
    checkpoints: dict,
    dataset,
    steps: list,
    n_samples: int = 100,
    device: str = "cpu",
    save_path: str = "equivariance_vs_step.png",
    wandb_run=None,
) -> dict:
    """
    Plot equivariance error vs denoising step for multiple methods.

    Args:
        checkpoints:  {label: checkpoint_path}
        dataset:      LeRobot dataset
        steps:        denoising steps to evaluate, e.g. [0, 10, 25, 50, 75, 99]
        n_samples:    samples per step per method
        device:       torch device string
        save_path:    where to save the PNG
        wandb_run:    active wandb run object (or None to skip wandb logging)

    Returns:
        {label: [error_at_step_0, error_at_step_1, ...]}
    """
    results = {}

    for label, ckpt_path in checkpoints.items():
        print(f"Loading {label} from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        config = ckpt["config"]
        policy = SoftEquiDiffPolicy(config).to(device)
        policy.load_state_dict(ckpt["model_state_dict"])

        errors = []
        for k in steps:
            err = measure_equivariance_error(policy, dataset, k, n_samples, device)
            errors.append(err)
            print(f"  k={k:3d}: error={err:.4f}")
        results[label] = errors

    # --- Matplotlib figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, errors in results.items():
        ax.plot(steps, errors, marker="o", label=label)
    ax.set_xlabel("Denoising step k  (0=clean, K=noisy)")
    ax.set_ylabel("Mean equivariance error")
    ax.set_title("Equivariance error vs denoising step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved figure: {save_path}")

    # --- Wandb logging ---
    if wandb_run is not None:
        import wandb

        # Log the figure as an image
        wandb_run.log({"eval/equivariance_vs_step": wandb.Image(save_path)})

        # Log as a wandb Table so the interactive line chart works too
        columns = ["method", "step_k", "equivariance_error"]
        table = wandb.Table(columns=columns)
        for label, errors in results.items():
            for k, err in zip(steps, errors):
                table.add_data(label, k, err)
        wandb_run.log({"eval/equivariance_table": table})

        # Also log per-step errors as separate series (one metric per method)
        # so wandb's line chart can group them automatically
        for label, errors in results.items():
            safe_label = label.replace(" ", "_").replace("-", "_")
            for k, err in zip(steps, errors):
                wandb_run.log({f"eval/equi_error/{safe_label}": err, "eval/step_k": k})

    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint (or comma-separated list for comparison)")
    p.add_argument("--labels", default=None,
                   help="Comma-separated labels matching --checkpoint order")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--steps", nargs="+", type=int, default=[0, 10, 25, 50, 75, 99])
    p.add_argument("--save_dir", default=".")

    # Wandb
    p.add_argument("--wandb_project", default="soft-equidiff-pusht")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default="eval")
    p.add_argument("--no_wandb", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Support comma-separated list of checkpoints for multi-run comparison
    ckpt_paths = [p.strip() for p in args.checkpoint.split(",")]
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
    else:
        labels = [Path(p).parent.name for p in ckpt_paths]  # use parent dir name as label
    checkpoints = dict(zip(labels, ckpt_paths))

    # Load first checkpoint's config for dataset setup
    first_ckpt = torch.load(ckpt_paths[0], map_location=device)
    config = first_ckpt["config"]

    # --- Wandb init ---
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                job_type="eval",
                config={
                    "checkpoints": ckpt_paths,
                    "labels": labels,
                    "n_samples": args.n_samples,
                    "steps": args.steps,
                    "device": str(device),
                },
            )
        except ImportError:
            print("wandb not installed — skipping wandb logging.")
            use_wandb = False

    # Load dataset
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(
            config.dataset_repo_id,
            delta_timestamps={
                "observation.image": [i / 10.0 for i in range(-config.n_obs_steps + 1, 1)],
                "observation.state": [i / 10.0 for i in range(-config.n_obs_steps + 1, 1)],
                "action": [i / 10.0 for i in range(config.horizon)],
            },
        )
    except ImportError:
        raise ImportError("LeRobot required for evaluation. See requirements.txt.")

    # --- Free weight norms (single checkpoint) ---
    policy = SoftEquiDiffPolicy(config).to(device)
    policy.load_state_dict(first_ckpt["model_state_dict"])
    free_norms = analyze_free_weights(policy)

    if wandb_run is not None:
        import wandb

        # Log as a bar-chart table
        norm_table = wandb.Table(columns=["layer", "free_weight_norm_sq"])
        for layer_name, norm_sq in free_norms.items():
            if layer_name != "__total__":
                norm_table.add_data(layer_name, norm_sq)
        wandb_run.log({
            "eval/free_weight_norms": norm_table,
            "eval/total_free_weight_norm_sq": free_norms["__total__"],
        })

    # --- Equivariance vs step ---
    print(f"\nMeasuring equivariance error at steps: {args.steps}")
    results = plot_equivariance_vs_step(
        checkpoints=checkpoints,
        dataset=dataset,
        steps=args.steps,
        n_samples=args.n_samples,
        device=str(device),
        save_path=str(save_dir / "equivariance_vs_step.png"),
        wandb_run=wandb_run,
    )

    # Also print a clean summary table
    print("\n--- Equivariance error summary ---")
    header = f"{'step':>6}" + "".join(f"  {lab:>20}" for lab in results.keys())
    print(header)
    for i, k in enumerate(args.steps):
        row = f"{k:>6}" + "".join(f"  {errs[i]:>20.4f}" for errs in results.values())
        print(row)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
