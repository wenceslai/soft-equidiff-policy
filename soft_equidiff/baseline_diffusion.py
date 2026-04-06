"""
Vanilla (non-equivariant) diffusion policy baseline for Push-T.
Self-contained single file — config, model, policy, training, and eval all here.

Architecture mirrors the equivariant policy for direct comparison:
    obs_cond_dim  = n_obs_steps * (n_hidden + state_features) = 2*(64+32) = 192  (identical)
    action_features = 32                                                           (identical)
    U-Net down_dims, kernel_size, n_groups, diffusion hyperparameters             (identical)
    Image encoder channel progression matches equivariant raw channel counts
    Standard Conv2d/Linear instead of escnn equivariant layers

Usage:
    # Train
    python -m soft_equidiff.baseline_diffusion train --run_name base_diffusion

    # Eval
    python -m soft_equidiff.baseline_diffusion eval \\
        --checkpoint outputs/base_diffusion/policy_step0200000.pt \\
        --n_episodes 50 --device cuda
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader

from .model.unet1d import ConditionalUnet1D


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BaseDiffConfig:
    # Encoder dims — matched to equivariant config for fair comparison
    n_hidden: int = 64          # image encoder output dim  (= equi n_hidden per group element)
    state_features: int = 32    # state MLP output dim      (= equi state_features)
    action_features: int = 32   # action encoder/decoder dim (= equi action_features)

    # Diffusion — identical to SoftEquiDiffConfig
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    num_diffusion_steps: int = 100
    num_inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # U-Net — identical to SoftEquiDiffConfig
    unet_down_dims: Tuple[int, ...] = (64, 128, 256)
    unet_diffusion_step_embed_dim: int = 128
    unet_kernel_size: int = 5
    unet_n_groups: int = 8

    # Training — identical to SoftEquiDiffConfig
    normalize_inputs: bool = True
    lr: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 64
    num_train_steps: int = 200_000
    grad_clip_norm: float = 10.0

    # Push-T
    image_size: int = 96
    action_dim: int = 2
    state_dim: int = 2
    dataset_repo_id: str = "lerobot/pusht"


# ---------------------------------------------------------------------------
# Normalizer  (identical to policy.py)
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    def __init__(self, stats: Dict[str, Dict[str, torch.Tensor]]):
        super().__init__()
        for key, val in stats.items():
            self.register_buffer(f"{key.replace('.', '_')}_min", torch.as_tensor(val["min"]).float())
            self.register_buffer(f"{key.replace('.', '_')}_max", torch.as_tensor(val["max"]).float())
        self._keys = list(stats.keys())

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        k = key.replace(".", "_")
        lo = getattr(self, f"{k}_min").to(x.device)
        hi = getattr(self, f"{k}_max").to(x.device)
        return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0

    def unnormalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        k = key.replace(".", "_")
        lo = getattr(self, f"{k}_min").to(x.device)
        hi = getattr(self, f"{k}_max").to(x.device)
        return (x + 1.0) / 2.0 * (hi - lo + 1e-8) + lo


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Standard 2D residual block with GroupNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False)
        self.gn1   = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(8, out_ch)
        self.skip  = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_ch),
            )
            if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


class BaseImageEncoder(nn.Module):
    """
    CNN encoder mirroring the equivariant encoder structure and channel counts.

    The equivariant encoder produces n_fields * N channels at each stage
    (n_fields × 8 for N=8, n_hidden=64):  64 → 128 → 256 → 512 channels.
    This baseline uses the same raw channel counts, without any group structure.

    96×96 → (stem) → (4 × ResBlock pair + MaxPool2d) → (Conv2d k=6) → (B, n_hidden)
    Spatial: 96 → 48 → 24 → 12 → 6 → 1
    """

    def __init__(self, n_hidden: int = 64):
        super().__init__()
        c1, c2, c3, c4 = 64, 128, 256, 512  # matches equi: n//8*N, n//4*N, n//2*N, n*N

        # stem: trivial RGB → first feature map  (mirrors equi stem R2Conv k=5)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, c1),
            nn.ReLU(inplace=True),
        )
        # stage 1: 96×96 → 48×48  (mirrors equi block1, block2, pool1)
        self.stage1 = nn.Sequential(_ResBlock(c1, c1), _ResBlock(c1, c1), nn.MaxPool2d(2))
        # stage 2: 48×48 → 24×24  (mirrors equi block3, block4, pool2)
        self.stage2 = nn.Sequential(_ResBlock(c1, c2), _ResBlock(c2, c2), nn.MaxPool2d(2))
        # stage 3: 24×24 → 12×12  (mirrors equi block5, block6, pool3)
        self.stage3 = nn.Sequential(_ResBlock(c2, c3), _ResBlock(c3, c3), nn.MaxPool2d(2))
        # stage 4: 12×12 → 6×6   (mirrors equi block7, block8, pool4)
        self.stage4 = nn.Sequential(_ResBlock(c3, c4), _ResBlock(c4, c4), nn.MaxPool2d(2))
        # final: 6×6 → 1×1, project down to n_hidden  (mirrors equi final_conv k=6)
        self.final = nn.Sequential(
            nn.Conv2d(c4, n_hidden, kernel_size=6, bias=False),
            nn.GroupNorm(8, n_hidden),
            nn.ReLU(inplace=True),
        )
        self.n_hidden = n_hidden

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Args: image (B, 3, 96, 96) float in [0,1].  Returns: (B, n_hidden)"""
        x = self.stem(image)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final(x)
        return x.flatten(1)  # (B, n_hidden)


class BaseStateEncoder(nn.Module):
    """MLP: (B, 2) → (B, state_features)"""

    def __init__(self, state_features: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, state_features), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseActionEncoder(nn.Module):
    """Linear: (B, T, 2) → (B, T, action_features)"""

    def __init__(self, action_features: int = 32):
        super().__init__()
        self.linear = nn.Linear(2, action_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BaseDecoder(nn.Module):
    """Linear: (B, T, action_features) → (B, T, 2)"""

    def __init__(self, action_features: int = 32):
        super().__init__()
        self.linear = nn.Linear(action_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BaseDiffModel(nn.Module):
    """
    Non-equivariant diffusion model.

    obs_cond_dim = n_obs_steps * (n_hidden + state_features) = 2*(64+32) = 192
    — identical to the equivariant model, so the U-Net sees exactly the same
    conditioning dimensionality.  Any performance difference isolates the
    equivariant vs. standard encoder.
    """

    def __init__(
        self,
        n_hidden: int = 64,
        state_features: int = 32,
        action_features: int = 32,
        n_obs_steps: int = 2,
        horizon: int = 16,
        unet_down_dims: Tuple[int, ...] = (64, 128, 256),
        unet_dsed: int = 128,
        unet_kernel_size: int = 5,
        unet_n_groups: int = 8,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.state_features = state_features
        self.n_obs_steps = n_obs_steps

        self.image_encoder  = BaseImageEncoder(n_hidden)
        self.state_encoder  = BaseStateEncoder(state_features)
        self.action_encoder = BaseActionEncoder(action_features)

        obs_cond_dim = n_obs_steps * (n_hidden + state_features)
        self.unet = ConditionalUnet1D(
            input_dim=action_features,
            global_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=unet_dsed,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=unet_n_groups,
        )
        self.decoder = BaseDecoder(action_features)

    def encode_obs(self, obs_images: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        """
        Returns obs_cond: (B, obs_cond_dim)
        Layout: [img_obs0_f..., img_obs1_f..., state_obs0_f..., state_obs1_f...]
        — same layout as the equivariant encode_obs (just without the N group dim).
        """
        B, n_obs, C, H, W = obs_images.shape

        img_feat   = self.image_encoder(obs_images.reshape(B * n_obs, C, H, W))  # (B*n_obs, n_hidden)
        img_feat   = img_feat.reshape(B, n_obs * self.n_hidden)

        state_feat = self.state_encoder(obs_state.reshape(B * n_obs, 2))          # (B*n_obs, state_features)
        state_feat = state_feat.reshape(B, n_obs * self.state_features)

        return torch.cat([img_feat, state_feat], dim=-1)  # (B, obs_cond_dim)

    def forward(
        self,
        obs_images: torch.Tensor,
        obs_state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond  = self.encode_obs(obs_images, obs_state)           # (B, obs_cond_dim)
        act_feat  = self.action_encoder(noisy_actions)               # (B, horizon, action_features)
        noise_emb = self.unet(act_feat, timestep, global_cond=obs_cond)  # (B, horizon, action_features)
        return self.decoder(noise_emb)                               # (B, horizon, 2)


# ---------------------------------------------------------------------------
# Policy (training forward pass + inference)
# ---------------------------------------------------------------------------

class BaseDiffPolicy(nn.Module):
    """Training + inference wrapper around BaseDiffModel."""

    def __init__(self, config: BaseDiffConfig, dataset_stats: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.model = BaseDiffModel(
            n_hidden=config.n_hidden,
            state_features=config.state_features,
            action_features=config.action_features,
            n_obs_steps=config.n_obs_steps,
            horizon=config.horizon,
            unet_down_dims=tuple(config.unet_down_dims),
            unet_dsed=config.unet_diffusion_step_embed_dim,
            unet_kernel_size=config.unet_kernel_size,
            unet_n_groups=config.unet_n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_steps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )
        self.normalizer = None
        if dataset_stats is not None and config.normalize_inputs:
            self.normalizer = Normalizer(dataset_stats)
        self._action_queue: deque = deque(maxlen=config.n_action_steps)

    def reset(self):
        self._action_queue.clear()

    # ------------------------------------------------------------------
    # Shared preprocessing (identical logic to SoftEquiDiffPolicy)
    # ------------------------------------------------------------------

    def _preprocess_batch(self, batch: dict) -> tuple:
        obs_images = batch["observation.image"]
        obs_state  = batch["observation.state"]

        if obs_images.ndim == 4:
            obs_images = obs_images.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1, -1, -1)
        if obs_state.ndim == 2:
            obs_state = obs_state.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1)

        obs_images = obs_images.float()
        if obs_images.max() > 1.5:
            obs_images = obs_images / 255.0

        if self.normalizer is not None:
            obs_state = self.normalizer.normalize("observation.state", obs_state)

        actions = batch.get("action", None)
        if actions is not None and self.normalizer is not None:
            actions = self.normalizer.normalize("action", actions)

        return obs_images, obs_state, actions

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        obs_images, obs_state, actions = self._preprocess_batch(batch)
        B      = actions.shape[0]
        device = actions.device

        noise         = torch.randn_like(actions)
        k             = torch.randint(0, self.config.num_diffusion_steps, (B,), device=device)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, k)

        noise_pred = self.model(obs_images, obs_state, noisy_actions, k)
        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss, "mse_loss": loss.detach()}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        if len(self._action_queue) == 0:
            chunk = self._generate_action_chunk(batch)
            for t in range(self.config.n_action_steps):
                self._action_queue.append(chunk[0, t])
        return self._action_queue.popleft()

    @torch.no_grad()
    def _generate_action_chunk(self, batch: dict) -> torch.Tensor:
        obs_images, obs_state, _ = self._preprocess_batch(batch)
        B      = obs_images.shape[0]
        device = obs_images.device

        actions = torch.randn(B, self.config.horizon, self.config.action_dim, device=device)
        self.noise_scheduler.set_timesteps(self.config.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            timestep   = torch.full((B,), t, dtype=torch.long, device=device)
            noise_pred = self.model(obs_images, obs_state, actions, timestep)
            actions    = self.noise_scheduler.step(noise_pred, t, actions).prev_sample

        if self.normalizer is not None:
            actions = self.normalizer.unnormalize("action", actions)
        return actions[:, :self.config.n_action_steps]


# ---------------------------------------------------------------------------
# Dataset helpers (identical to train.py)
# ---------------------------------------------------------------------------

def _build_dataset(config: BaseDiffConfig, tilt_transform=None, video_backend: str = "pyav"):
    try:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError("LeRobot not found. Install with: pip install -e lerobot/")

    dataset = LeRobotDataset(
        config.dataset_repo_id,
        delta_timestamps={
            "observation.image": [i / 10.0 for i in range(-config.n_obs_steps + 1, 1)],
            "observation.state": [i / 10.0 for i in range(-config.n_obs_steps + 1, 1)],
            "action":            [i / 10.0 for i in range(config.horizon)],
        },
        image_transforms=tilt_transform,
        video_backend=video_backend,
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


def _grad_norm(policy: nn.Module) -> float:
    total = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device  = torch.device(args.device)
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = BaseDiffConfig(
        n_obs_steps=args.n_obs_steps,
        num_train_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=getattr(args, "wandb_entity", None),
                name=args.run_name,
                config={
                    "model": "base_diffusion",
                    "n_hidden": config.n_hidden,
                    "state_features": config.state_features,
                    "action_features": config.action_features,
                    "n_obs_steps": config.n_obs_steps,
                    "horizon": config.horizon,
                    "unet_down_dims": config.unet_down_dims,
                    "num_diffusion_steps": config.num_diffusion_steps,
                    "lr": config.lr,
                    "batch_size": config.batch_size,
                },
                dir=str(out_dir),
            )
        except ImportError:
            print("wandb not installed — continuing without logging.")
            use_wandb = False

    print(f"Loading dataset: {config.dataset_repo_id}")
    dataset, stats = _build_dataset(config)

    policy    = BaseDiffPolicy(config, dataset_stats=stats).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_step = 1
    if args.resume is not None:
        ckpt       = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {ckpt['step']}")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Starting training: {args.run_name}  |  params: {n_params:,}")

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
    t0 = step_t0 = time.time()

    for step in range(start_step, config.num_train_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        optimizer.zero_grad()
        losses = policy(batch)
        losses["loss"].backward()
        gn = _grad_norm(policy)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
        optimizer.step()

        if step % args.log_every == 0:
            sps      = args.log_every / (time.time() - step_t0)
            step_t0  = time.time()
            print(
                f"step {step:>7d} | loss {losses['loss'].item():.4f} "
                f"| grad {gn:.3f} | {sps:.1f} s/s | {time.time()-t0:.0f}s"
            )
            if use_wandb:
                wandb.log({
                    "train/loss":          losses["loss"].item(),
                    "train/mse_loss":      losses["mse_loss"].item(),
                    "train/grad_norm":     gn,
                    "train/steps_per_sec": sps,
                }, step=step)

        if step % args.save_every == 0 or step == config.num_train_steps:
            ckpt_path = out_dir / f"policy_step{step:07d}.pt"
            torch.save({
                "step":               step,
                "model_state_dict":   policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config":             config,
                "dataset_stats":      stats,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

    if use_wandb:
        wandb.finish()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Eval (identical logic to eval_success_rate.py)
# ---------------------------------------------------------------------------

def _make_env(seed: int = 0):
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def _obs_to_batch(obs: dict, device: torch.device, obs_buffer: list) -> dict:
    images = []
    states = []
    for o in obs_buffer:
        images.append(torch.from_numpy(o["pixels"]).permute(2, 0, 1).float())
        states.append(torch.from_numpy(o["agent_pos"]).float())
    return {
        "observation.image": torch.stack(images).unsqueeze(0).to(device),   # (1, n, 3, H, W)
        "observation.state": torch.stack(states).unsqueeze(0).to(device),   # (1, n, 2)
    }


def _run_episode(policy: BaseDiffPolicy, env, seed: int, max_steps: int, device: torch.device) -> dict:
    policy.reset()
    obs, _ = env.reset(seed=seed)
    obs_buffer  = [obs] * policy.config.n_obs_steps
    max_coverage = 0.0

    for step in range(max_steps):
        batch  = _obs_to_batch(obs, device, obs_buffer)
        action = policy.select_action(batch).cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        max_coverage = max(max_coverage, info.get("coverage", 0.0))
        obs_buffer.pop(0)
        obs_buffer.append(obs)
        if terminated or truncated:
            break

    return {"success": max_coverage >= 0.95, "coverage": max_coverage}


def evaluate(args):
    device    = torch.device(args.device)
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config: BaseDiffConfig = ckpt["config"]
    stats     = ckpt.get("dataset_stats", None)

    policy = BaseDiffPolicy(config, dataset_stats=stats).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                job_type="eval",
                config={"checkpoint": args.checkpoint, "n_episodes": args.n_episodes},
            )
        except ImportError:
            use_wandb = False

    env       = _make_env(seed=args.base_seed)
    successes = []
    coverages = []

    print(f"\nEvaluating: {args.checkpoint}")
    for i in range(args.n_episodes):
        result = _run_episode(policy, env, seed=args.base_seed + i,
                              max_steps=args.max_steps, device=device)
        successes.append(result["success"])
        coverages.append(result["coverage"])
        if (i + 1) % 10 == 0:
            print(f"  episode {i+1:3d}/{args.n_episodes}  "
                  f"success_rate={np.mean(successes):.3f}  "
                  f"last_coverage={result['coverage']:.3f}")

    env.close()
    sr  = float(np.mean(successes))
    mc  = float(np.mean(coverages))
    sc  = float(np.std(coverages))
    print(f"\n  SUCCESS RATE:  {sr:.3f}  ({int(sr*args.n_episodes)}/{args.n_episodes})")
    print(f"  Mean coverage: {mc:.3f} ± {sc:.3f}")

    if use_wandb:
        import wandb as wb
        wb.summary["success_rate"]   = sr
        wb.summary["mean_coverage"]  = mc
        wb.summary["std_coverage"]   = sc
        wb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(p):
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb_project", default="soft-equidiff-policy")
    p.add_argument("--no_wandb",     action="store_true")


def main():
    parser = argparse.ArgumentParser(description="Baseline (non-equivariant) diffusion policy")
    sub    = parser.add_subparsers(dest="mode", required=True)

    # --- train ---
    tp = sub.add_parser("train")
    _add_common_args(tp)
    tp.add_argument("--run_name",    default="base_diffusion")
    tp.add_argument("--output_dir",  default="outputs")
    tp.add_argument("--num_steps",   type=int, default=200_000)
    tp.add_argument("--batch_size",  type=int, default=64)
    tp.add_argument("--lr",          type=float, default=1e-4)
    tp.add_argument("--n_obs_steps", type=int, default=2)
    tp.add_argument("--log_every",   type=int, default=500)
    tp.add_argument("--save_every",  type=int, default=50_000)
    tp.add_argument("--resume",      default=None)
    tp.add_argument("--wandb_entity", default=None)

    # --- eval ---
    ep = sub.add_parser("eval")
    _add_common_args(ep)
    ep.add_argument("--checkpoint",    required=True)
    ep.add_argument("--n_episodes",    type=int, default=50)
    ep.add_argument("--max_steps",     type=int, default=300)
    ep.add_argument("--base_seed",     type=int, default=42)
    ep.add_argument("--wandb_run_name", default="base_diffusion_eval")

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
