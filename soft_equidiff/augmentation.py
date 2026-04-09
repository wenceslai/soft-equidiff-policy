"""
C₄ rotation augmentation for equivariant diffusion policy training.

Applies a uniformly-sampled rotation from {0°, 90°, 180°, 270°} independently
to every sample in a batch, consistently across:
  - observation images  (B, n_obs, 3, H, W)
  - observation state   (B, n_obs, 2)
  - action              (B, horizon, 2)

C₄ is used rather than C₈ because 90° rotations on a pixel grid are exact
(torch.rot90 is a view — no interpolation, no blurring, zero cost).
45° rotations require bilinear interpolation which is slower and lossy.

Efficiency:
    Samples are grouped by rotation index (4 groups of ~B/4) so at most 3
    actual tensor ops are performed per batch (the k=0 identity is skipped).
    Everything runs on whatever device the batch tensors are on (GPU/CPU).

Usage:
    aug = C4Augmentation(workspace_center)
    batch = aug(batch)   # in training loop, after .to(device)
"""

import math
import torch


# Rotation matrices for k=1,2,3 (k=0 is identity, skipped)
# For 90°k rotation: x' = cos(90k)*x - sin(90k)*y,  y' = sin(90k)*x + cos(90k)*y
_ROTMATS = {
    1: (0.0,  -1.0,  1.0,  0.0),   # 90°:  x'=-y,  y'= x
    2: (-1.0,  0.0,  0.0, -1.0),   # 180°: x'=-x,  y'=-y
    3: (0.0,   1.0, -1.0,  0.0),   # 270°: x'= y,  y'=-x
}


class C4Augmentation:
    """
    Callable that applies a random C₄ rotation to every sample in a batch.

    Args:
        workspace_center: (2,) tensor [cx, cy] in the raw coordinate space of
                          state/action vectors (before normalisation).
                          Compute once from dataset stats:
                              (stats["observation.state"]["min"] +
                               stats["observation.state"]["max"]) / 2
    """

    def __init__(self, workspace_center: torch.Tensor):
        # Store as plain floats so we never accidentally move them to wrong device
        self.cx = float(workspace_center[0])
        self.cy = float(workspace_center[1])

    def __call__(self, batch: dict) -> dict:
        """
        Args:
            batch: dict with tensors already on the target device.
                   Must contain "observation.image", "observation.state", "action".
                   All other keys are passed through unchanged.
        Returns:
            New batch dict with rotated tensors.
        """
        imgs   = batch["observation.image"].clone()   # (B, n_obs, C, H, W)
        states = batch["observation.state"].clone()   # (B, n_obs, 2)
        acts   = batch["action"].clone()              # (B, horizon, 2)

        B = imgs.shape[0]

        # One random rotation index per sample: k ∈ {0, 1, 2, 3}
        k_indices = torch.randint(0, 4, (B,))  # CPU is fine, it's tiny

        for k, (cos_a, neg_sin_a, sin_a, _) in _ROTMATS.items():

            mask = (k_indices == k).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue

            # --- Images: torch.rot90 is exact, free, no interpolation ---
            # Shape: (B', n_obs, C, H, W) — rot90 acts on the last two dims
            imgs[mask] = torch.rot90(imgs[mask], k=k, dims=[-2, -1])

            # --- 2D vectors: rotate around workspace center ---
            # Translate to origin, rotate, translate back
            cx, cy = self.cx, self.cy

            def _rot_vecs(vecs: torch.Tensor) -> torch.Tensor:
                # vecs: (..., 2)
                dx = vecs[..., 0] - cx
                dy = vecs[..., 1] - cy
                new_x = cx + cos_a * dx + neg_sin_a * dy
                new_y = cy + sin_a * dx + cos_a     * dy
                return torch.stack([new_x, new_y], dim=-1)

            states[mask] = _rot_vecs(states[mask])
            acts[mask]   = _rot_vecs(acts[mask])

        return {**batch,
                "observation.image": imgs,
                "observation.state": states,
                "action":            acts}
