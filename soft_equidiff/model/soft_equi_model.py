"""
SoftEquiDiffModel — the full soft equivariant diffusion model.

Pipeline (per forward pass):
    1. Image encoder (C_N equivariant CNN) → per-group image features
    2. State encoder (escnn Linear) → per-group state features
    3. Concatenate obs features per group element → global UNet condition
    4. Action encoder (escnn Linear, per timestep) → per-group action features
    5. Shared 1D Temporal U-Net (same weights for all N group elements)
    6. Equivariant decoder (escnn Linear) → predicted noise ε̂ in action space

Equivariance comes from:
    - Steps 1-4, 6: escnn equivariant layers (with optional free path via SoftEquiWrapper)
    - Step 5: same UNet weights applied independently to each group element's features

The model subclasses nn.Module and is wrapped by SoftEquiDiffPolicy for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import EquiImageEncoder, EquiStateEncoder, EquiActionEncoder
from .decoder import EquiDecoder
from .unet1d import ConditionalUnet1D
from .soft_wrapper import SoftEquiWrapper, EquivariancePenaltySchedule


class SoftEquiDiffModel(nn.Module):
    """
    Core model: encodes observations + noisy actions, predicts noise via
    shared equivariant U-Net.

    Args:
        N:                 C_N group order (number of discrete rotations)
        n_hidden:          number of regular-repr fields in image encoder output
        state_features:    number of regular-repr fields in state encoder output
        action_features:   number of regular-repr fields in action encoder/decoder
        n_obs_steps:       number of consecutive observation frames stacked
        horizon:           action prediction horizon
        unet_down_dims:    channel widths in U-Net encoder levels
        unet_dsed:         diffusion step embedding dimension
        unet_kernel_size:  Conv1d kernel size in U-Net
        unet_n_groups:     GroupNorm groups in U-Net
        soften_image:      apply SoftEquiWrapper to image encoder layers
        soften_state:      apply SoftEquiWrapper to state encoder layer
        soften_action:     apply SoftEquiWrapper to action encoder layer
        soften_decoder:    apply SoftEquiWrapper to decoder layer
    """

    def __init__(
        self,
        N: int = 8,
        n_hidden: int = 64,
        state_features: int = 32,
        action_features: int = 32,
        n_obs_steps: int = 2,
        horizon: int = 16,
        unet_down_dims=(256, 512, 1024),
        unet_dsed: int = 128,
        unet_kernel_size: int = 5,
        unet_n_groups: int = 8,
        soften_image: bool = True,
        soften_state: bool = True,
        soften_action: bool = True,
        soften_decoder: bool = True,
    ):
        super().__init__()
        self.N = N
        self.n_hidden = n_hidden
        self.state_features = state_features
        self.action_features = action_features
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon

        # --- Encoders ---
        self.image_encoder = EquiImageEncoder(N=N, n_hidden=n_hidden, soften=soften_image)
        self.state_encoder = EquiStateEncoder(N=N, out_fields=state_features, soften=soften_state)
        self.action_encoder = EquiActionEncoder(N=N, out_fields=action_features, soften=soften_action)

        # --- Shared U-Net ---
        # obs_cond_dim: per group element, all obs steps concatenated
        obs_cond_dim = n_obs_steps * (n_hidden + state_features)
        self.unet = ConditionalUnet1D(
            input_dim=action_features,
            global_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=unet_dsed,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=unet_n_groups,
        )

        # --- Decoder ---
        self.decoder = EquiDecoder(N=N, in_fields=action_features, soften=soften_decoder)

    def encode_obs(self, obs_images: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        """
        Encode observation into per-group-element conditioning vectors.

        Args:
            obs_images: (B, n_obs_steps, 3, H, W)
            obs_state:  (B, n_obs_steps, 2)
        Returns:
            obs_cond: (B * N, obs_cond_dim) — ready for U-Net global_cond
        """
        B, n_obs, C, H, W = obs_images.shape

        # --- Image features ---
        imgs_flat = obs_images.reshape(B * n_obs, C, H, W)
        img_geo = self.image_encoder(imgs_flat)            # GeometricTensor (B*n_obs, n_hidden*N, 1, 1)
        img_feat = img_geo.tensor.squeeze(-1).squeeze(-1)  # (B*n_obs, n_hidden*N)

        # field-major (B*n_obs, n_hidden*N) → per-group-element (B, N, n_obs*n_hidden)
        img_feat = img_feat.reshape(B, n_obs, self.n_hidden, self.N)
        img_feat = img_feat.permute(0, 3, 1, 2)           # (B, N, n_obs, n_hidden)
        img_feat = img_feat.reshape(B, self.N, n_obs * self.n_hidden)

        # --- State features ---
        states_flat = obs_state.reshape(B * n_obs, 2)
        state_geo = self.state_encoder(states_flat)         # GeometricTensor (B*n_obs, state_features*N)
        state_feat = state_geo.tensor                       # (B*n_obs, state_features*N)

        # field-major → per-group-element (B, N, n_obs*state_features)
        state_feat = state_feat.reshape(B, n_obs, self.state_features, self.N)
        state_feat = state_feat.permute(0, 3, 1, 2)        # (B, N, n_obs, state_features)
        state_feat = state_feat.reshape(B, self.N, n_obs * self.state_features)

        # --- Combine & flatten group dim into batch ---
        obs_feat = torch.cat([img_feat, state_feat], dim=-1)  # (B, N, obs_cond_dim)
        return obs_feat.reshape(B * self.N, -1)               # (B*N, obs_cond_dim)

    def forward(
        self,
        obs_images: torch.Tensor,
        obs_state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise for the given noisy actions conditioned on observations.

        Args:
            obs_images:    (B, n_obs_steps, 3, H, W)
            obs_state:     (B, n_obs_steps, 2)
            noisy_actions: (B, horizon, 2)
            timestep:      (B,) denoising step indices
        Returns:
            noise_pred: (B, horizon, 2)
        """
        B = obs_images.shape[0]

        # Observation conditioning: (B*N, obs_cond_dim)
        obs_cond = self.encode_obs(obs_images, obs_state)

        # Action encoding: (B*N, horizon, action_features) per-group-element
        act_feat = self.action_encoder(noisy_actions)  # (B*N, horizon, action_features)

        # Expand timestep for all N group elements: (B*N,)
        timestep_expanded = timestep.repeat_interleave(self.N)

        # Shared U-Net processes all group elements with same weights
        noise_emb = self.unet(act_feat, timestep_expanded, global_cond=obs_cond)
        # noise_emb: (B*N, horizon, action_features)

        # --- Degeneracy diagnostics (cheap, stored for optional wandb logging) ---
        # Reshape to (B, N, horizon, action_features) to measure inter-group variation.
        # If inter_group_var ≈ 0, all N UNet outputs are equal → decoder output ≈ 0 (symmetric collapse).
        with torch.no_grad():
            ne = noise_emb.detach().reshape(B, self.N, self.horizon, -1)
            # Variance across group elements (dim=1), averaged over all other dims.
            self._last_diagnostics = {
                # Near 0 → UNet outputs are identical across group elements (degenerate).
                "debug/unet_inter_group_var": ne.var(dim=1).mean().item(),
                # Overall magnitude of UNet outputs.
                "debug/unet_output_norm": ne.norm(dim=-1).mean().item(),
            }

        # Decode to action-space noise: (B, horizon, 2)
        noise_pred = self.decoder(noise_emb)

        # Magnitude of the final predicted noise (should be ~1 if learning well).
        with torch.no_grad():
            self._last_diagnostics["debug/noise_pred_norm"] = noise_pred.detach().norm(dim=-1).mean().item()

        return noise_pred

    def get_total_free_weight_norm(self) -> torch.Tensor:
        """Sum of ||W_free||² across all SoftEquiWrapper layers (for equivariance penalty)."""
        total = torch.tensor(0.0)
        for module in self.modules():
            if isinstance(module, SoftEquiWrapper):
                total = total + module.free_weight_norm_sq()
        return total
