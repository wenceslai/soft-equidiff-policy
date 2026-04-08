"""
Equivariant decoder: maps per-group-element noise features back to 2D action-space vectors.

Uses escnn.nn.Linear from (action_features * regular_repr) → (1 * irrep(1)) per timestep.
The irrep(1) of C_N represents a 2D vector that transforms correctly under rotation.
"""

import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn

from .soft_wrapper import SoftEquiWrapper


class EquiDecoder(nn.Module):
    """
    Decodes per-group-element noise embeddings to 2D action predictions.

    Expects input in per-group-element layout (B*N, T_a, in_fields) from the
    SharedDenoisingUNet, then converts to field-major format for the escnn Linear.

    Output: (B, T_a, 2)
    """

    def __init__(self, N: int = 8, in_fields: int = 32, soften: bool = False):
        super().__init__()
        self.N = N
        self.in_fields = in_fields
        self.soften = soften

        group = gspaces.rot2dOnR2(N).fibergroup # fibergroup is the object representing gruops actions on objects. We're rotating 2d vectors in 2d spacce
        self.gs0 = gspaces.no_base_space(group) # no spatial structure

        # in_fields * regular_repr → 1 * irrep(1)  (2D action vector)
        self.in_type = enn.FieldType(self.gs0, in_fields * [self.gs0.regular_repr])
        self.out_type = enn.FieldType(self.gs0, [self.gs0.fibergroup.irrep(1)])

        linear = enn.Linear(self.in_type, self.out_type)
        if soften:
            self.linear = SoftEquiWrapper(linear, self.in_type.size, self.out_type.size, spatial=False)
        else:
            self.linear = linear

    def forward(self, noise_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise_embed: (B * N, T_a, in_fields) — UNet output, per-group-element layout, T_a is the number of action steps in the horizon
        Returns:
            (B, T_a, 2)
        """
        BN, T_a, K = noise_embed.shape
        B = BN // self.N

        # Convert per-group-element layout → field-major layout required by escnn
        # (B*N, T_a, K) → (B, N, T_a, K) → (B, T_a, K, N) → (B*T_a, K*N)
        feat = noise_embed.reshape(B, self.N, T_a, K)
        feat = feat.permute(0, 2, 3, 1)           # (B, T_a, K, N)
        feat = feat.reshape(B * T_a, K * self.N)  # field-major: [f0_g0, ..., f_{K-1}_{N-1}]

        x = enn.GeometricTensor(feat, self.in_type)
        out = self.linear(x)

        if self.soften:
            # SoftEquiWrapper returns raw Tensor
            return out.reshape(B, T_a, 2)
        else:
            return out.tensor.reshape(B, T_a, 2)
