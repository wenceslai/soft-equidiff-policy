"""
Evaluation and analysis tools for SoftEquiDiffPolicy.

Two main functions:

1. measure_equivariance_error(model, dataset, k, n_samples)
   — For a fixed denoising step k, compute the average equivariance error:
       ||ε(g·obs, g·a^k, k) - g·ε(obs, a^k, k)||
   across random group elements g.  This should be near 0 for an exactly
   equivariant model and larger for a model that has relaxed equivariance.

2. plot_equivariance_vs_step(model, dataset, steps, ...)
   — Sweep over multiple denoising steps and plot the equivariance error curve.
   For SoftEqui-step, we expect the error to be larger at small k (near-clean)
   and smaller at large k (near-noisy).

Usage:
    python -m soft_equidiff.eval --checkpoint outputs/soft_step/policy_step0200000.pt \
                                  --n_samples 200 --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import SoftEquiDiffConfig
from .policy import SoftEquiDiffPolicy


# ---------------------------------------------------------------------------
# Rotation utilities (C_N group actions on observations and actions)
# ---------------------------------------------------------------------------

def rotate_image(image: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """
    Rotate a (*, 3, H, W) image tensor by angle_rad around the image centre.
    Uses torchvision.transforms.functional for exact affine warping.
    """
    import torchvision.transforms.functional as TF
    import math
    angle_deg = math.degrees(angle_rad)
    # Handle batch dims
    orig_shape = image.shape
    img = image.reshape(-1, *orig_shape[-3:])
    rotated = torch.stack([TF.rotate(img[i], angle_deg) for i in range(img.shape[0])])
    return rotated.reshape(orig_shape)


def rotate_action(action: torch.Tensor, angle_rad: float, center: tuple = (256.0, 256.0)) -> torch.Tensor:
    """
    Rotate 2D (x, y) actions by angle_rad around a given centre (in pixel coords).
    action: (*, 2)  [x, y coordinates]
    """
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

    Args:
        policy:    trained SoftEquiDiffPolicy
        dataset:   LeRobot dataset (iterable of dicts)
        k:         denoising step at which to measure
        n_samples: number of data samples to average over
        device:    torch device
        center:    rotation centre in pixel coordinates

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
        # obs_images: (1, n_obs, 3, H, W), obs_state: (1, n_obs, 2), actions: (1, H, 2)

        # Sample noisy actions at step k
        noise = torch.randn_like(actions)
        timestep = torch.tensor([k], device=device)
        noisy_actions = policy.noise_scheduler.add_noise(actions, noise, timestep)

        # Predict noise for original observation
        eps_orig = policy.model(obs_images, obs_state, noisy_actions, timestep)
        # eps_orig: (1, horizon, 2)

        batch_errors = []
        for angle in angles[1:]:  # skip identity
            # Rotate observation and noisy actions
            obs_rot = rotate_image(obs_images, angle)
            state_rot = rotate_state(obs_state, angle, center)
            noisy_rot = rotate_action(noisy_actions, angle, center)

            # Predict noise for rotated input
            eps_rot = policy.model(obs_rot, state_rot, noisy_rot, timestep)

            # Equivariance: ε(g·x) should equal g·ε(x)
            eps_expected = rotate_action(eps_orig, angle, center=(0.0, 0.0))
            # Note: noise lives in relative displacement space, so center=(0,0)

            err = (eps_rot - eps_expected).norm(dim=-1).mean().item()
            batch_errors.append(err)

        errors.append(np.mean(batch_errors))

    return float(np.mean(errors))


def plot_equivariance_vs_step(
    checkpoints: dict,
    dataset,
    steps: list,
    n_samples: int = 100,
    device: str = "cpu",
    save_path: str = "equivariance_vs_step.png",
):
    """
    Plot equivariance error vs denoising step for multiple methods.

    Args:
        checkpoints: {label: checkpoint_path} dict
        dataset:     LeRobot dataset
        steps:       list of denoising steps to evaluate (e.g. [0, 10, 25, 50, 75, 99])
        n_samples:   samples per step
        device:      torch device
        save_path:   where to save the figure
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

    # Plot
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
    print(f"Saved: {save_path}")
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Free-weight norm analysis (how much symmetry is broken per layer)
# ---------------------------------------------------------------------------

def analyze_free_weights(policy: SoftEquiDiffPolicy):
    """Print the free weight norm² for each SoftEquiWrapper in the model."""
    from .model.soft_wrapper import SoftEquiWrapper

    print("\n--- Free weight norms (||W_free||²) ---")
    total = 0.0
    for name, module in policy.model.named_modules():
        if isinstance(module, SoftEquiWrapper):
            norm_sq = module.free_weight_norm_sq().item()
            total += norm_sq
            print(f"  {name:<60s}  {norm_sq:.6f}")
    print(f"  {'TOTAL':<60s}  {total:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--steps", nargs="+", type=int, default=[0, 10, 25, 50, 75, 99])
    p.add_argument("--save_dir", default=".")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

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

    # Build policy
    policy = SoftEquiDiffPolicy(config).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])

    analyze_free_weights(policy)

    print(f"\nMeasuring equivariance error at steps: {args.steps}")
    for k in args.steps:
        err = measure_equivariance_error(policy, dataset, k, args.n_samples, str(device))
        print(f"  k={k:3d}: error={err:.4f}")


if __name__ == "__main__":
    main()
