"""
Soft equivariance mechanism: Residual Pathway Prior (RPP).

For any escnn layer L, the soft wrapper computes:
    output = L(x_geometric).tensor + W_free(x_raw)

The free path W_free is initialized near zero (starts equivariant) and is
penalized during training via EquivariancePenaltySchedule.

Reference: Finzi, Benton & Wilson, "Residual Pathway Priors", NeurIPS 2021.
"""

import torch
import torch.nn as nn


class SoftEquiWrapper(nn.Module):
    """
    Wraps an escnn equivariant layer with a parallel unconstrained (free) pathway.

    Usage:
        equi_layer = enn.Linear(in_type, out_type)
        wrapper = SoftEquiWrapper(equi_layer, in_type.size, out_type.size, spatial=False)

        # In forward (non-spatial):
        out_tensor = wrapper(x_geometric)          # torch.Tensor
        out_geo = enn.GeometricTensor(out_tensor, out_type)   # re-wrap if needed

    Args:
        equi_layer: an escnn EquivariantModule (enn.Linear or enn.R2Conv)
        in_size:    total input channel count (= in_type.size)
        out_size:   total output channel count (= out_type.size)
        spatial:    if True, free path is Conv2d (for R2Conv); else Linear (for enn.Linear)
        kernel_size, padding, stride: only used when spatial=True, must match the R2Conv
    """

    def __init__(
        self,
        equi_layer,
        in_size: int,
        out_size: int,
        spatial: bool = False,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.equi_layer = equi_layer
        self.spatial = spatial

        if spatial:
            self.free_layer = nn.Conv2d(
                in_size, out_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            )
        else:
            self.free_layer = nn.Linear(in_size, out_size, bias=False)

        # Near-zero init → starts close to exact equivariance
        nn.init.normal_(self.free_layer.weight, std=0.01)

    def forward(self, x_geometric):
        """
        Args:
            x_geometric: escnn.nn.GeometricTensor
        Returns:
            torch.Tensor (raw) — caller must re-wrap as GeometricTensor if needed
        """
        equi_out = self.equi_layer(x_geometric).tensor
        free_out = self.free_layer(x_geometric.tensor)
        return equi_out + free_out

    def free_weight_norm_sq(self):
        """||W_free||² — used for the equivariance penalty."""
        return self.free_layer.weight.pow(2).sum()


class EquivariancePenaltySchedule:
    """
    Returns the regularization weight λ(k) for a given denoising step k.

    Two modes:
        'constant':      λ(k) = lambda_base  (same weight at all steps)
        'step_dependent': λ(k) = lambda_base * (k / K)^power
            → stronger penalty at high noise (large k), weaker at low noise (small k)
            → at k=0 (near-clean), λ≈0: model free to break symmetry
            → at k=K (full noise), λ=lambda_base: near-equivariant

    Special values:
        lambda_base = 0.0   → no equivariance preference (purely free model)
        lambda_base = 1000. → near-exact equivariance (recovers EquiDiff behaviour)
    """

    def __init__(
        self,
        mode: str = "step_dependent",
        lambda_base: float = 0.1,
        power: float = 1.0,
        num_diffusion_steps: int = 100,
    ):
        assert mode in ("constant", "step_dependent")
        self.mode = mode
        self.lambda_base = lambda_base
        self.power = power
        self.K = num_diffusion_steps

    def __call__(self, k):
        """
        Args:
            k: (B,) tensor of denoising steps or a scalar int
        Returns:
            scalar float weight
        """
        if self.mode == "constant":
            return self.lambda_base
        # step_dependent
        if isinstance(k, int):
            ratio = k / self.K
        else:
            ratio = k.float().mean().item() / self.K
        return self.lambda_base * (ratio ** self.power)
