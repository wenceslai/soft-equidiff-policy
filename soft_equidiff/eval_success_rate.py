"""
Post-training success rate evaluation on Push-T.

Loads one or more checkpoints, runs N rollout episodes per checkpoint in the
Push-T gym environment, and reports mean success rate ± std.

Success criterion (standard Push-T): coverage of T-block target area >= 0.95.

Usage:
    # Single checkpoint:
    python -m soft_equidiff.eval_success_rate \
        --checkpoint outputs/baseline_no_softening/policy_step0200000.pt \
        --n_episodes 50 --device cuda

    # Compare multiple checkpoints:
    python -m soft_equidiff.eval_success_rate \
        --checkpoint outputs/run_a/policy_step0200000.pt,outputs/run_b/policy_step0200000.pt \
        --labels "EquiDiff-exact,SoftEqui-step" \
        --n_episodes 50 --device cuda

    # Disable wandb:
    python -m soft_equidiff.eval_success_rate --checkpoint ... --no_wandb

Requires:
    pip install gym_pusht gymnasium
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .baseline_diffusion import BaseDiffConfig
from .config import SoftEquiDiffConfig
from .policy import SoftEquiDiffPolicy


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 0):
    """Create a seeded Push-T gym environment."""
    try:
        import gymnasium as gym
        import gym_pusht  # noqa: F401 — registers the env
    except ImportError:
        raise ImportError(
            "Push-T gym not found. Install with:\n"
            "  pip install gym_pusht gymnasium"
        )
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def obs_to_batch(obs: dict, device: torch.device, obs_buffer: list, n_obs_steps: int) -> dict:
    """
    Convert a gymnasium obs dict to a policy batch dict.

    obs_buffer is a list of the last n_obs_steps raw observations (oldest first).
    Each obs has:
        "pixels"    — (H, W, 3) uint8
        "agent_pos" — (2,) float32

    Returns batch with:
        "observation.image" — (1, n_obs_steps, 3, H, W) float32
        "observation.state" — (1, n_obs_steps, 2)       float32
    """
    # Stack obs frames (oldest first)
    images = []
    states = []
    for o in obs_buffer:
        img = torch.from_numpy(o["pixels"]).permute(2, 0, 1).float()  # (3, H, W)
        images.append(img)
        states.append(torch.from_numpy(o["agent_pos"]).float())

    image_tensor = torch.stack(images, dim=0).unsqueeze(0).to(device)   # (1, n, 3, H, W)
    state_tensor = torch.stack(states, dim=0).unsqueeze(0).to(device)   # (1, n, 2)

    return {
        "observation.image": image_tensor,
        "observation.state": state_tensor,
    }


def run_episode(
    policy: SoftEquiDiffPolicy,
    env,
    seed: int,
    max_steps: int = 300,
    device: torch.device = torch.device("cpu"),
    success_threshold: float = 0.95,
) -> dict:
    """
    Run a single Push-T episode.

    Returns:
        dict with keys: "success" (bool), "coverage" (float), "n_steps" (int)
    """
    n_obs_steps = policy.config.n_obs_steps
    policy.reset()

    obs, _ = env.reset(seed=seed)

    # Pre-fill the obs buffer by repeating the first frame
    obs_buffer = [obs] * n_obs_steps

    max_coverage = 0.0

    for step in range(max_steps):
        batch = obs_to_batch(obs, device, obs_buffer, n_obs_steps)
        action = policy.select_action(batch)  # (2,)
        action_np = action.cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)

        # Push-T reports coverage in info
        coverage = info.get("coverage", 0.0)
        max_coverage = max(max_coverage, coverage)

        # Update obs buffer (slide window)
        obs_buffer.pop(0)
        obs_buffer.append(obs)

        if terminated or truncated:
            break

    success = max_coverage >= success_threshold
    return {"success": success, "coverage": max_coverage, "n_steps": step + 1}


# ---------------------------------------------------------------------------
# Multi-episode evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    n_episodes: int = 50,
    max_steps: int = 300,
    device: torch.device = torch.device("cpu"),
    base_seed: int = 42,
    success_threshold: float = 0.95,
) -> dict:
    """
    Load a checkpoint and evaluate over n_episodes rollouts.

    Returns:
        dict with "success_rate", "mean_coverage", "std_coverage",
                  "successes" (list of bool), "coverages" (list of float)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: SoftEquiDiffConfig = ckpt["config"]

    # Pass dataset_stats so the normalizer is created before loading state_dict
    dataset_stats = ckpt.get("dataset_stats", None)
    policy = SoftEquiDiffPolicy(config, dataset_stats=dataset_stats).to(device)

    # strict=False: escnn registers filter/matrix/expanded_bias as non-persistent
    # buffers (recomputed on init, not saved), so they'll appear as missing keys
    policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    policy.eval()

    env = make_env(seed=base_seed)

    successes = []
    coverages = []

    for i in range(n_episodes):
        seed = base_seed + i
        result = run_episode(policy, env, seed=seed, max_steps=max_steps,
                             device=device, success_threshold=success_threshold)
        successes.append(result["success"])
        coverages.append(result["coverage"])

        if (i + 1) % 10 == 0:
            sr_so_far = np.mean(successes)
            print(f"  episode {i+1:3d}/{n_episodes}  success_rate={sr_so_far:.3f}  "
                  f"last_coverage={result['coverage']:.3f}")

    env.close()

    return {
        "success_rate": float(np.mean(successes)),
        "mean_coverage": float(np.mean(coverages)),
        "std_coverage": float(np.std(coverages)),
        "successes": successes,
        "coverages": coverages,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint, or comma-separated list for comparison")
    p.add_argument("--labels", default=None,
                   help="Comma-separated labels matching --checkpoint order")
    p.add_argument("--n_episodes", type=int, default=50,
                   help="Number of rollout episodes per checkpoint")
    p.add_argument("--max_steps", type=int, default=300,
                   help="Maximum env steps per episode")
    p.add_argument("--success_threshold", type=float, default=0.95,
                   help="Coverage threshold to count as success (default: 0.95)")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default=".")

    # Wandb
    p.add_argument("--wandb_project", default="soft-equidiff-pusht")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default="eval_success_rate")
    p.add_argument("--no_wandb", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = [p.strip() for p in args.checkpoint.split(",")]
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
    else:
        labels = [Path(p).parent.name for p in ckpt_paths]

    # --- Wandb init ---
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                job_type="eval_success_rate",
                config={
                    "checkpoints": ckpt_paths,
                    "labels": labels,
                    "n_episodes": args.n_episodes,
                    "max_steps": args.max_steps,
                    "success_threshold": args.success_threshold,
                    "base_seed": args.base_seed,
                    "device": str(device),
                },
            )
        except ImportError:
            print("wandb not installed — skipping wandb logging.")

    # --- Run evaluations ---
    all_results = {}
    for label, ckpt_path in zip(labels, ckpt_paths):
        print(f"\nEvaluating: {label}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Episodes:   {args.n_episodes}")

        results = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            device=device,
            base_seed=args.base_seed,
            success_threshold=args.success_threshold,
        )
        all_results[label] = results

        print(f"  SUCCESS RATE:   {results['success_rate']:.3f}  "
              f"({int(results['success_rate'] * args.n_episodes)}/{args.n_episodes})")
        print(f"  Mean coverage:  {results['mean_coverage']:.3f} "
              f"± {results['std_coverage']:.3f}")

        if wandb_run is not None:
            import wandb as wb
            safe_label = label.replace(" ", "_").replace("-", "_")
            wandb_run.summary[f"{safe_label}/success_rate"] = results["success_rate"]
            wandb_run.summary[f"{safe_label}/mean_coverage"] = results["mean_coverage"]
            wandb_run.summary[f"{safe_label}/std_coverage"] = results["std_coverage"]

            # Log per-episode coverage as a table for distribution analysis
            cov_table = wb.Table(columns=["episode", "coverage", "success"])
            for i, (cov, suc) in enumerate(zip(results["coverages"], results["successes"])):
                cov_table.add_data(i, cov, int(suc))
            wandb_run.log({f"{safe_label}/episode_coverages": cov_table})

    # --- Summary table ---
    print("\n--- Success Rate Summary ---")
    print(f"{'Method':<30}  {'Success Rate':>12}  {'Mean Coverage':>14}  {'Std Coverage':>12}")
    print("-" * 72)
    for label, results in all_results.items():
        print(f"{label:<30}  {results['success_rate']:>12.3f}  "
              f"{results['mean_coverage']:>14.3f}  {results['std_coverage']:>12.3f}")

    if wandb_run is not None:
        # Summary comparison table
        import wandb as wb
        summary_table = wb.Table(
            columns=["method", "success_rate", "mean_coverage", "std_coverage"]
        )
        for label, results in all_results.items():
            summary_table.add_data(
                label,
                results["success_rate"],
                results["mean_coverage"],
                results["std_coverage"],
            )
        wandb_run.log({"eval/success_rate_summary": summary_table})
        wandb_run.finish()


if __name__ == "__main__":
    main()
