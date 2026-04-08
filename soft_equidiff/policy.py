"""
SoftEquiDiffPolicy — LeRobot-compatible policy wrapping SoftEquiDiffModel.

Implements the training loss (forward) and inference denoising loop (select_action).

Batch keys expected (Push-T / LeRobot format):
    batch["observation.image"]  — (B, n_obs_steps, 3, H, W)  or  (B, 3, H, W)
    batch["observation.state"]  — (B, n_obs_steps, 2)         or  (B, 2)
    batch["action"]             — (B, horizon, 2)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .config import SoftEquiDiffConfig
from .model.soft_equi_model import SoftEquiDiffModel
from .model.soft_wrapper import EquivariancePenaltySchedule


class Normalizer(nn.Module):
    """Simple min-max normalizer to [-1, 1]. Stats come from the dataset."""

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


class SoftEquiDiffPolicy(nn.Module):
    """
    Soft Equivariant Diffusion Policy for Push-T.

    Training: call forward(batch) → returns {"loss": scalar, ...}
    Inference: call select_action(batch) → returns (action_dim,) tensor
               (with action queue: generates n_action_steps actions at once,
                returns one per call)

    To use with LeRobot's training loop, wrap in their PreTrainedPolicy or use
    as a plain nn.Module with the provided train.py.
    """

    def __init__(self, config: SoftEquiDiffConfig, dataset_stats: Optional[dict] = None):
        super().__init__()
        self.config = config

        self.model = SoftEquiDiffModel(
            N=config.n_rotations,
            n_hidden=config.enc_n_hidden,
            state_features=config.state_features,
            action_features=config.action_features,
            n_obs_steps=config.n_obs_steps,
            horizon=config.horizon,
            unet_down_dims=tuple(config.unet_down_dims),
            unet_dsed=config.unet_diffusion_step_embed_dim,
            unet_kernel_size=config.unet_kernel_size,
            unet_n_groups=config.unet_n_groups,
            soften_image=config.soften_image_encoder,
            soften_state=config.soften_state_encoder,
            soften_action=config.soften_action_encoder,
            soften_decoder=config.soften_decoder,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_steps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )

        self.penalty_schedule = EquivariancePenaltySchedule(
            mode=config.penalty_mode,
            lambda_base=config.lambda_base,
            power=config.lambda_power,
            num_diffusion_steps=config.num_diffusion_steps,
        )

        self.normalizer = None
        if dataset_stats is not None and config.normalize_inputs:
            self.normalizer = Normalizer(dataset_stats)

        self._action_queue: deque = deque(maxlen=config.n_action_steps)

    def reset(self):
        """Call at the start of each episode."""
        self._action_queue.clear()

    # ------------------------------------------------------------------
    # Data preprocessing
    # ------------------------------------------------------------------

    def _preprocess_batch(self, batch: dict) -> tuple:
        """
        Extract and optionally normalise images, states, and actions.

        Handles both (B, 3, H, W) and (B, n_obs_steps, 3, H, W) image formats.

        Returns:
            obs_images: (B, n_obs_steps, 3, H, W)
            obs_state:  (B, n_obs_steps, 2)
            actions:    (B, horizon, 2)  — only meaningful during training
        """
        obs_images = batch["observation.image"]   # (B, [n_obs,] 3, H, W)
        obs_state = batch["observation.state"]    # (B, [n_obs,] 2)

        # Add obs_steps dimension if missing (single-frame case)
        if obs_images.ndim == 4:
            obs_images = obs_images.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1, -1, -1)
        if obs_state.ndim == 2:
            obs_state = obs_state.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1)

        # Normalise images to [0, 1] (they may already be in this range)
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
        """
        Training forward pass. Returns loss dict.

        Loss = MSE(predicted_noise, true_noise) + λ(k) * ||W_free||²
        """
        obs_images, obs_state, actions = self._preprocess_batch(batch)
        B = actions.shape[0]
        device = actions.device

        # Sample noise and diffusion step
        noise = torch.randn_like(actions)
        k = torch.randint(0, self.config.num_diffusion_steps, (B,), device=device)

        noisy_actions = self.noise_scheduler.add_noise(actions, noise, k)

        # Predict noise
        noise_pred = self.model(obs_images, obs_state, noisy_actions, k)

        # MSE loss
        mse_loss = F.mse_loss(noise_pred, noise)

        # Equivariance penalty
        free_norm = self.model.get_total_free_weight_norm().to(device)
        lam = self.penalty_schedule(k)
        equi_loss = lam * free_norm

        total_loss = mse_loss + equi_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss.detach(),
            "equi_penalty": equi_loss.detach(),
            "lambda": torch.tensor(lam),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """
        Return the next action. Generates a full chunk when the queue is empty.

        Args:
            batch: dict with "observation.image" and "observation.state"
        Returns:
            action: (action_dim,) tensor (single action for one env step)
        """
        if len(self._action_queue) == 0:
            chunk = self._generate_action_chunk(batch)  # (1, n_action_steps, 2)
            for t in range(self.config.n_action_steps):
                self._action_queue.append(chunk[0, t])

        return self._action_queue.popleft()

    @torch.no_grad()
    def _generate_action_chunk(self, batch: dict) -> torch.Tensor:
        """
        Run the full DDPM denoising loop to generate an action chunk.

        Returns:
            actions: (B, horizon, 2)
        """
        obs_images, obs_state, _ = self._preprocess_batch(batch)
        B = obs_images.shape[0]
        device = obs_images.device

        # Start from pure noise
        actions = torch.randn(B, self.config.horizon, self.config.action_dim, device=device)

        self.noise_scheduler.set_timesteps(self.config.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            timestep = torch.full((B,), t, dtype=torch.long, device=device)
            noise_pred = self.model(obs_images, obs_state, actions, timestep)
            actions = self.noise_scheduler.step(noise_pred, t, actions).prev_sample # x_{t-1} = (1/√α_t) * (x_t - β_t/(√(1-ᾱ_t)) * ε_pred) + √(β̃_t) * z

        # Unnormalise actions
        if self.normalizer is not None:
            actions = self.normalizer.unnormalize("action", actions)

        # Only return the n_action_steps that will be executed
        return actions[:, :self.config.n_action_steps]
