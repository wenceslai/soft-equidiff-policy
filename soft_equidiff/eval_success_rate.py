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

    # Print generated actions + save a behaviour GIF:
    python -m soft_equidiff.eval_success_rate \
        --checkpoint outputs/run_a/policy_step0200000.pt \
        --print_actions --gif_fps 15 --save_dir /tmp/evals

    # Disable wandb:
    python -m soft_equidiff.eval_success_rate --checkpoint ... --no_wandb

Requires:
    pip install gym_pusht gymnasium
    pip install imageio          # for GIF export (optional)
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .baseline_diffusion import BaseDiffConfig, BaseDiffPolicy
from .camera_tilt import apply_camera_tilt
from .config import SoftEquiDiffConfig
from .policy import SoftEquiDiffPolicy


# ---------------------------------------------------------------------------
# Test-time rotation helpers
# ---------------------------------------------------------------------------

# Workspace centre in raw Push-T coordinates (same as augmentation.py)
_WS_CENTER = (256.0, 256.0)

# Rotation matrices for k=1,2,3 quarter-turns (same sign convention as C4Augmentation)
_ROTMATS = {
    0: ( 1.0,  0.0,  0.0,  1.0),   # identity
    1: ( 0.0, -1.0,  1.0,  0.0),   # 90°  CCW: x'=-y, y'= x
    2: (-1.0,  0.0,  0.0, -1.0),   # 180°
    3: ( 0.0,  1.0, -1.0,  0.0),   # 270° CCW: x'= y, y'=-x
}


def _rotate_vec(xy: np.ndarray, k: int, cx: float = _WS_CENTER[0],
                cy: float = _WS_CENTER[1]) -> np.ndarray:
    """Rotate a (..., 2) numpy array by k quarter-turns around (cx, cy)."""
    cos_a, neg_sin_a, sin_a, _ = _ROTMATS[k % 4]
    dx, dy = xy[..., 0] - cx, xy[..., 1] - cy
    return np.stack([cx + cos_a * dx + neg_sin_a * dy,
                     cy + sin_a * dx +       cos_a * dy], axis=-1)


def _unrotate_vec(xy: np.ndarray, k: int, cx: float = _WS_CENTER[0],
                  cy: float = _WS_CENTER[1]) -> np.ndarray:
    """Inverse of _rotate_vec — rotate by -k quarter-turns."""
    return _rotate_vec(xy, (-k) % 4, cx, cy)


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


def obs_to_batch(obs: dict, device: torch.device, obs_buffer: list, n_obs_steps: int,
                 test_rotation: int = 0, tilt_degrees: float = 0.0) -> dict:
    """
    Convert a gymnasium obs dict to a policy batch dict.

    obs_buffer is a list of the last n_obs_steps raw observations (oldest first).
    Each obs has:
        "pixels"    — (H, W, 3) uint8
        "agent_pos" — (2,) float32

    If test_rotation > 0, the image is rotated by test_rotation × 90° and the
    agent position is rotated by the same amount around the workspace centre.
    The caller is responsible for unrotating the returned action before executing
    it in the environment.

    Returns batch with:
        "observation.image" — (1, n_obs_steps, 3, H, W) float32
        "observation.state" — (1, n_obs_steps, 2)       float32
    """
    images = []
    states = []
    for o in obs_buffer:
        img = torch.from_numpy(o["pixels"]).permute(2, 0, 1).float()  # (3, H, W)
        if test_rotation:
            img = torch.rot90(img, k=test_rotation, dims=[-2, -1])
        if tilt_degrees:
            img = apply_camera_tilt(img.unsqueeze(0), tilt_degrees).squeeze(0)
        images.append(img)

        pos = o["agent_pos"].copy()
        if test_rotation:
            pos = _rotate_vec(pos, test_rotation)
        states.append(torch.from_numpy(pos).float())

    image_tensor = torch.stack(images, dim=0).unsqueeze(0).to(device)   # (1, n, 3, H, W)
    state_tensor = torch.stack(states, dim=0).unsqueeze(0).to(device)   # (1, n, 2)

    return {
        "observation.image": image_tensor,
        "observation.state": state_tensor,
    }


def run_episode(
    policy,
    env,
    seed: int,
    max_steps: int = 300,
    device: torch.device = torch.device("cpu"),
    success_threshold: float = 0.95,
    record: bool = False,
    print_actions: bool = False,
    test_rotation: int = 0,
    tilt_degrees: float = 0.0,
    rot_aug: bool = False,
) -> dict:
    """
    Run a single Push-T episode.

    Args:
        record:        if True, collect rendered frames and actions for GIF / logging
        print_actions: if True, print each action as it is executed

    Returns:
        dict with keys:
            "success"   — bool
            "coverage"  — float (max coverage seen)
            "n_steps"   — int
        If record=True, also:
            "frames"    — list of (H, W, 3) uint8 arrays (one per step)
            "actions"   — list of (2,) float32 arrays (unnormalised workspace coords)
            "agent_pos" — list of (2,) float32 arrays (agent position at each step)
            "coverages" — list of float (per-step coverage)
    """
    n_obs_steps = policy.config.n_obs_steps
    policy.reset()

    obs, _ = env.reset(seed=seed)

    # If rot_aug, pick a random C4 rotation for this episode (overrides test_rotation)
    if rot_aug:
        rng = np.random.default_rng(seed)
        test_rotation = int(rng.integers(0, 4))

    obs_buffer = [obs] * n_obs_steps

    max_coverage = 0.0

    frames    = [] if record else None
    actions   = [] if record else None
    agent_pos = [] if record else None
    cov_trace = [] if record else None

    for step in range(max_steps):
        # Capture frame BEFORE the step so the GIF shows the state the policy saw
        if record:
            frames.append(obs["pixels"].copy())
            agent_pos.append(obs["agent_pos"].copy())

        batch = obs_to_batch(obs, device, obs_buffer, n_obs_steps,
                             test_rotation=test_rotation, tilt_degrees=tilt_degrees)
        action = policy.select_action(batch)  # (2,) in (possibly rotated) workspace coords
        action_np = action.cpu().numpy()

        # Unrotate action back to original frame before executing in env
        if test_rotation:
            action_np = _unrotate_vec(action_np, test_rotation)

        if print_actions:
            pos = obs["agent_pos"]
            print(f"    step {step:4d}  agent=({pos[0]:7.1f},{pos[1]:7.1f})  "
                  f"action=({action_np[0]:7.1f},{action_np[1]:7.1f})")

        if record:
            actions.append(action_np.copy())

        obs, reward, terminated, truncated, info = env.step(action_np)

        coverage = info.get("coverage", 0.0)
        max_coverage = max(max_coverage, coverage)

        if record:
            cov_trace.append(coverage)

        obs_buffer.pop(0)
        obs_buffer.append(obs)

        if terminated or truncated:
            break

    success = max_coverage >= success_threshold
    result = {
        "success":  success,
        "coverage": max_coverage,
        "n_steps":  step + 1,
    }
    if record:
        # Append the final frame so the GIF ends on the terminal state
        frames.append(obs["pixels"].copy())
        result["frames"]    = frames
        result["actions"]   = np.array(actions)    # (T, 2)
        result["agent_pos"] = np.array(agent_pos)  # (T, 2)
        result["coverages"] = np.array(cov_trace)  # (T,)

    return result


# ---------------------------------------------------------------------------
# GIF helpers
# ---------------------------------------------------------------------------

def save_gif(frames: list, path: Path, fps: int = 10) -> bool:
    """
    Save a list of (H, W, 3) uint8 arrays as an animated GIF.

    Uses Pillow for maximum compatibility with macOS Preview / browsers.
    Returns True on success, False if Pillow is missing.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [gif] Pillow not installed — skipping GIF. "
              "Install with: pip install Pillow")
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = int(1000 / fps)   # Pillow takes milliseconds per frame
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,   # skip palette optimisation — faster and more compatible
    )
    print(f"  [gif] saved → {path}  ({len(frames)} frames @ {fps} fps)")
    return True


def print_action_summary(actions: np.ndarray, agent_pos: np.ndarray, coverages: np.ndarray):
    """Print a compact statistical summary of actions generated during an episode."""
    print("  Action statistics (workspace coordinates):")
    print(f"    x  — min={actions[:,0].min():.1f}  max={actions[:,0].max():.1f}  "
          f"mean={actions[:,0].mean():.1f}  std={actions[:,0].std():.1f}")
    print(f"    y  — min={actions[:,1].min():.1f}  max={actions[:,1].max():.1f}  "
          f"mean={actions[:,1].mean():.1f}  std={actions[:,1].std():.1f}")
    print(f"  Agent-pos statistics:")
    print(f"    x  — min={agent_pos[:,0].min():.1f}  max={agent_pos[:,0].max():.1f}  "
          f"mean={agent_pos[:,0].mean():.1f}  std={agent_pos[:,0].std():.1f}")
    print(f"    y  — min={agent_pos[:,1].min():.1f}  max={agent_pos[:,1].max():.1f}  "
          f"mean={agent_pos[:,1].mean():.1f}  std={agent_pos[:,1].std():.1f}")
    # Detect degenerate case: actions clustered tightly (std < 20 world units ≈ 3.75% of 512)
    action_std = actions.std(axis=0).mean()
    if action_std < 20.0:
        print(f"  ⚠  LOW ACTION DIVERSITY (mean std={action_std:.1f}) — "
              f"model may be outputting near-constant actions (degenerate decoder?)")
    peak_cov = coverages.max() if len(coverages) else 0.0
    print(f"  Peak coverage during episode: {peak_cov:.3f}")


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
    gif_save_path: Path = None,
    gif_fps: int = 10,
    print_actions: bool = False,
    test_rotation: int = 0,
    tilt_degrees: float = 0.0,
    rot_aug: bool = False,
) -> dict:
    """
    Load a checkpoint and evaluate over n_episodes rollouts.

    If gif_save_path is given, records episode 0 (base_seed) and saves a GIF there.
    If print_actions is True, prints every action in that same recorded episode.

    Returns:
        dict with "success_rate", "mean_coverage", "std_coverage",
                  "successes", "coverages",
                  "gif_path" (Path or None)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    dataset_stats = ckpt.get("dataset_stats", None)
    if isinstance(config, BaseDiffConfig):
        policy = BaseDiffPolicy(config, dataset_stats=dataset_stats).to(device)
    else:
        policy = SoftEquiDiffPolicy(config, dataset_stats=dataset_stats).to(device)

    # strict=False: escnn registers filter/matrix/expanded_bias as non-persistent
    # buffers (recomputed on init, not saved), so they'll appear as missing keys
    policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    policy.eval()

    env = make_env(seed=base_seed)

    successes = []
    coverages = []
    saved_gif_path = None

    for i in range(n_episodes):
        seed = base_seed + i
        # Record the FIRST episode for GIF / action printing
        record_this = (i == 0) and (gif_save_path is not None or print_actions)

        result = run_episode(
            policy, env, seed=seed,
            max_steps=max_steps,
            device=device,
            success_threshold=success_threshold,
            record=record_this,
            print_actions=(print_actions and record_this),
            test_rotation=test_rotation,
            tilt_degrees=tilt_degrees,
            rot_aug=rot_aug,
        )
        successes.append(result["success"])
        coverages.append(result["coverage"])

        if record_this:
            print(f"  Episode 0 (seed={seed}):  "
                  f"success={result['success']}  coverage={result['coverage']:.3f}  "
                  f"steps={result['n_steps']}")
            print_action_summary(result["actions"], result["agent_pos"], result["coverages"])

            if gif_save_path is not None:
                ok = save_gif(result["frames"], gif_save_path, fps=gif_fps)
                if ok:
                    saved_gif_path = gif_save_path

        if (i + 1) % 10 == 0:
            sr_so_far = np.mean(successes)
            print(f"  episode {i+1:3d}/{n_episodes}  success_rate={sr_so_far:.3f}  "
                  f"last_coverage={result['coverage']:.3f}")

    env.close()

    return {
        "success_rate":  float(np.mean(successes)),
        "mean_coverage": float(np.mean(coverages)),
        "std_coverage":  float(np.std(coverages)),
        "successes":     successes,
        "coverages":     coverages,
        "gif_path":      saved_gif_path,
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
    p.add_argument("--save_dir", default=".",
                   help="Directory to save GIFs and artefacts")

    # Test-time rotation
    p.add_argument("--test_rotation", type=int, default=0, choices=[0, 1, 2, 3],
                   help="Rotate observations by N×90° at test time and unrotate actions before "
                        "execution. 0=none, 1=90°, 2=180°, 3=270°. Tests equivariance robustness.")
    p.add_argument("--rot_aug", action="store_true",
                   help="Apply random C4 rotation per episode (same as training augmentation). "
                        "Overrides --test_rotation. Tests robustness across all 4 orientations.")

    # Camera tilt
    p.add_argument("--tilt_degrees", type=float, default=0.0,
                   help="Apply camera tilt warp to observations at eval time (e.g. 20.0). "
                        "Should match the --tilt_degrees used during training.")

    # Action printing
    p.add_argument("--print_actions", action="store_true",
                   help="Print every action generated in the first recorded episode. "
                        "Also prints per-episode action statistics. "
                        "Useful for diagnosing degenerate / constant-action policies.")

    # GIF
    p.add_argument("--no_gif", action="store_true",
                   help="Disable GIF recording (default: record one episode per checkpoint)")
    p.add_argument("--gif_fps", type=int, default=10,
                   help="Frames per second for the output GIF (default: 10)")

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
                    "gif_fps": args.gif_fps,
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

        # Build GIF path: <save_dir>/<label>_ep0.gif
        gif_path = None
        if not args.no_gif:
            safe_label = label.replace(" ", "_").replace("/", "_")
            gif_path = save_dir / f"{safe_label}_ep0.gif"

        if args.rot_aug:
            print(f"  Rotation augmentation: random C4 per episode")
        elif args.test_rotation:
            print(f"  Test-time rotation: {args.test_rotation}×90° = {args.test_rotation*90}°")
        if args.tilt_degrees:
            print(f"  Camera tilt: {args.tilt_degrees}°")

        results = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            device=device,
            base_seed=args.base_seed,
            success_threshold=args.success_threshold,
            gif_save_path=gif_path,
            gif_fps=args.gif_fps,
            print_actions=args.print_actions,
            test_rotation=args.test_rotation,
            tilt_degrees=args.tilt_degrees,
            rot_aug=args.rot_aug,
        )
        all_results[label] = results

        print(f"  SUCCESS RATE:   {results['success_rate']:.3f}  "
              f"({int(results['success_rate'] * args.n_episodes)}/{args.n_episodes})")
        print(f"  Mean coverage:  {results['mean_coverage']:.3f} "
              f"± {results['std_coverage']:.3f}")

        if wandb_run is not None:
            import wandb as wb
            safe_label = label.replace(" ", "_").replace("-", "_")

            # Extract step number from checkpoint filename (e.g. policy_step0200000.pt → 200000)
            ckpt_stem = Path(ckpt_path).stem  # e.g. "policy_step0200000"
            import re as _re
            step_match = _re.search(r"step(\d+)", ckpt_stem)
            ckpt_step = int(step_match.group(1)) if step_match else -1

            wandb_run.summary[f"{safe_label}/checkpoint_path"] = ckpt_path
            wandb_run.summary[f"{safe_label}/checkpoint_step"] = ckpt_step
            wandb_run.summary[f"{safe_label}/success_rate"] = results["success_rate"]
            wandb_run.summary[f"{safe_label}/mean_coverage"] = results["mean_coverage"]
            wandb_run.summary[f"{safe_label}/std_coverage"] = results["std_coverage"]

            # Per-episode coverage table
            cov_table = wb.Table(columns=["episode", "coverage", "success"])
            for i, (cov, suc) in enumerate(zip(results["coverages"], results["successes"])):
                cov_table.add_data(i, cov, int(suc))
            wandb_run.log({f"{safe_label}/episode_coverages": cov_table})

            # Upload GIF if it was saved
            if results["gif_path"] is not None:
                try:
                    gif_video = wb.Video(
                        str(results["gif_path"]),
                        fps=args.gif_fps,
                        format="gif",
                        caption=f"{label} — seed={args.base_seed}  "
                                f"coverage={results['coverages'][0]:.3f}",
                    )
                    wandb_run.log({f"{safe_label}/episode_gif": gif_video})
                    print(f"  [wandb] uploaded GIF for {label}")
                except Exception as e:
                    print(f"  [wandb] GIF upload failed: {e}")

    # --- Summary table ---
    print("\n--- Success Rate Summary ---")
    print(f"{'Method':<30}  {'Success Rate':>12}  {'Mean Coverage':>14}  {'Std Coverage':>12}")
    print("-" * 72)
    for label, results in all_results.items():
        print(f"{label:<30}  {results['success_rate']:>12.3f}  "
              f"{results['mean_coverage']:>14.3f}  {results['std_coverage']:>12.3f}")

    if wandb_run is not None:
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
