"""
Equivariant encoders for image, state, and action inputs.

Image encoder: adapted from pointW/equidiff's EquivariantResEncoder76Cyclic,
               modified for Push-T's 96×96 input.
State/action encoders: escnn.nn.Linear with GSpace0D (non-spatial equivariant maps).

All encoders support optional soft equivariance via SoftEquiWrapper.
"""

import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn

from .soft_wrapper import SoftEquiWrapper


# ---------------------------------------------------------------------------
# Equivariant residual block (2D spatial)
# ---------------------------------------------------------------------------

class EquiResBlock(nn.Module):
    """
    Equivariant residual block using escnn R2Conv.
    Follows pointW/equidiff's EquiResBlock exactly, with added soften option.

    When soften=True each R2Conv is replaced by a SoftEquiWrapper that adds
    a parallel free Conv2d pathway.
    """

    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        soften: bool = False,
    ):
        super().__init__()
        self.group = group
        self.soften = soften
        pad = (kernel_size - 1) // 2

        self.feat_type_in = enn.FieldType(group, input_channels * [group.regular_repr])
        self.feat_type_hid = enn.FieldType(group, hidden_dim * [group.regular_repr])

        conv1 = enn.R2Conv(self.feat_type_in, self.feat_type_hid,
                           kernel_size=kernel_size, padding=pad, stride=stride)
        conv2 = enn.R2Conv(self.feat_type_hid, self.feat_type_hid,
                           kernel_size=kernel_size, padding=pad)

        if soften:
            self.conv1 = SoftEquiWrapper(conv1, self.feat_type_in.size, self.feat_type_hid.size,
                                         spatial=True, kernel_size=kernel_size, padding=pad, stride=stride)
            self.conv2 = SoftEquiWrapper(conv2, self.feat_type_hid.size, self.feat_type_hid.size,
                                         spatial=True, kernel_size=kernel_size, padding=pad)
        else:
            self.conv1 = conv1
            self.conv2 = conv2

        self.relu1 = enn.ReLU(self.feat_type_hid, inplace=True)
        self.relu2 = enn.ReLU(self.feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim or stride != 1:
            skip_conv = enn.R2Conv(self.feat_type_in, self.feat_type_hid,
                                   kernel_size=1, stride=stride, bias=False)
            if soften:
                self.upscale = SoftEquiWrapper(skip_conv, self.feat_type_in.size,
                                               self.feat_type_hid.size, spatial=True,
                                               kernel_size=1, padding=0, stride=stride)
            else:
                self.upscale = skip_conv

    def _to_geo(self, x, field_type):
        """Wrap a raw Tensor as GeometricTensor (used when soften=True)."""
        if isinstance(x, torch.Tensor):
            return enn.GeometricTensor(x, field_type)
        return x

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        residual = x

        out = self.conv1(x)
        out = self.relu1(self._to_geo(out, self.feat_type_hid))

        out = self.conv2(out)
        out = self._to_geo(out, self.feat_type_hid)

        if self.upscale is not None:
            skip = self._to_geo(self.upscale(residual), self.feat_type_hid)
        else:
            skip = residual

        out = enn.GeometricTensor(out.tensor + skip.tensor, self.feat_type_hid)
        return self.relu2(out)


# ---------------------------------------------------------------------------
# Image encoder (C_N equivariant CNN, adapted for 96×96 Push-T input)
# ---------------------------------------------------------------------------

class EquiImageEncoder(nn.Module):
    """
    C_N-equivariant CNN for 96×96 RGB images.

    Architecture (adapted from EquivariantResEncoder76Cyclic in pointW/equidiff):
        stem  R2Conv k=5 pad=2: 3 trivial → n//8 regular  [96×96]
        ReLU
        EquiResBlock(n//8, n//8) × 2
        MaxPool(2)                                          [48×48]
        EquiResBlock(n//8, n//4) + EquiResBlock(n//4, n//4)
        MaxPool(2)                                          [24×24]
        EquiResBlock(n//4, n//2) + EquiResBlock(n//2, n//2)
        MaxPool(2)                                          [12×12]
        EquiResBlock(n//2, n)   + EquiResBlock(n, n)
        MaxPool(2)                                          [6×6]
        R2Conv k=6 pad=0: n → n regular                    [1×1]
        ReLU

    Output: GeometricTensor with shape (B, n * N, 1, 1),
            where n = n_hidden, N = group order.

    Flatten and reshape to (B, N, n_hidden) for per-group-element features.
    """

    def __init__(self, N: int = 8, n_hidden: int = 64, soften: bool = False):
        super().__init__()
        self.N = N
        self.n_hidden = n_hidden
        self.soften = soften
        self.group = gspaces.rot2dOnR2(N)

        # Field types
        trivial_in = enn.FieldType(self.group, 3 * [self.group.trivial_repr])
        t0 = enn.FieldType(self.group, (n_hidden // 8) * [self.group.regular_repr])
        t1 = t0
        t2 = enn.FieldType(self.group, (n_hidden // 4) * [self.group.regular_repr])
        t3 = enn.FieldType(self.group, (n_hidden // 2) * [self.group.regular_repr])
        t4 = enn.FieldType(self.group, n_hidden * [self.group.regular_repr])

        self.input_type = trivial_in
        self.stem_type = t0   # output type of the stem conv
        self.output_type = t4

        # Stem: trivial → n//8 regular (RGB → first equivariant features)
        stem_conv = enn.R2Conv(trivial_in, t0, kernel_size=5, padding=2, bias=False)
        if soften:
            self.stem_conv = SoftEquiWrapper(stem_conv, trivial_in.size, t0.size,
                                             spatial=True, kernel_size=5, padding=2)
        else:
            self.stem_conv = stem_conv
        self.stem_relu = enn.ReLU(t0, inplace=True)

        # Stage 1: n//8 (96×96)
        self.block1 = EquiResBlock(self.group, n_hidden // 8, n_hidden // 8, soften=soften)
        self.block2 = EquiResBlock(self.group, n_hidden // 8, n_hidden // 8, soften=soften)
        self.pool1 = enn.PointwiseMaxPool(t1, kernel_size=2, stride=2)  # 96→48

        # Stage 2: n//8 → n//4 (48×48)
        self.block3 = EquiResBlock(self.group, n_hidden // 8, n_hidden // 4, soften=soften)
        self.block4 = EquiResBlock(self.group, n_hidden // 4, n_hidden // 4, soften=soften)
        self.pool2 = enn.PointwiseMaxPool(t2, kernel_size=2, stride=2)  # 48→24

        # Stage 3: n//4 → n//2 (24×24)
        self.block5 = EquiResBlock(self.group, n_hidden // 4, n_hidden // 2, soften=soften)
        self.block6 = EquiResBlock(self.group, n_hidden // 2, n_hidden // 2, soften=soften)
        self.pool3 = enn.PointwiseMaxPool(t3, kernel_size=2, stride=2)  # 24→12

        # Stage 4: n//2 → n (12×12)
        self.block7 = EquiResBlock(self.group, n_hidden // 2, n_hidden, soften=soften)
        self.block8 = EquiResBlock(self.group, n_hidden, n_hidden, soften=soften)
        self.pool4 = enn.PointwiseMaxPool(t4, kernel_size=2, stride=2)  # 12→6

        # Final: 6×6 → 1×1 via R2Conv k=6 no padding
        final_conv = enn.R2Conv(t4, t4, kernel_size=6, padding=0, bias=False)
        if soften:
            self.final_conv = SoftEquiWrapper(final_conv, t4.size, t4.size,
                                              spatial=True, kernel_size=6, padding=0)
        else:
            self.final_conv = final_conv
        self.final_relu = enn.ReLU(t4, inplace=True)

    def _to_geo(self, x, field_type):
        if isinstance(x, torch.Tensor):
            return enn.GeometricTensor(x, field_type)
        return x

    def forward(self, image: torch.Tensor) -> enn.GeometricTensor:
        """
        Args:
            image: (B, 3, 96, 96) float tensor in [0, 1]
        Returns:
            GeometricTensor with tensor shape (B, n_hidden * N, 1, 1)
        """
        x = enn.GeometricTensor(image, self.input_type)

        # Stem
        x = self._to_geo(self.stem_conv(x), self.stem_type)
        x = self.stem_relu(x)

        # Stage 1
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        # Stage 2
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        # Stage 3
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)

        # Stage 4
        x = self.block7(x)
        x = self.block8(x)
        x = self.pool4(x)

        # Final 6→1
        x = self._to_geo(self.final_conv(x), self.output_type)
        x = self.final_relu(x)

        return x  # GeometricTensor (B, n_hidden*N, 1, 1)


# ---------------------------------------------------------------------------
# State encoder (non-spatial, escnn.nn.Linear)
# ---------------------------------------------------------------------------

class EquiStateEncoder(nn.Module):
    """
    Encodes a 2D (x, y) state vector equivariantly.

    Uses GSpace0D (no spatial base) and escnn.nn.Linear.
    The 2D vector transforms as irrep(1) of C_N.

    Output: escnn GeometricTensor with out_fields * [regular_repr] type,
            raw tensor shape (B, out_fields * N).
    """

    def __init__(self, N: int = 8, out_fields: int = 32, soften: bool = False):
        super().__init__()
        self.N = N
        self.out_fields = out_fields
        group = gspaces.rot2dOnR2(N).fibergroup
        self.gs0 = gspaces.no_base_space(group)

        self.in_type = enn.FieldType(self.gs0, [self.gs0.fibergroup.irrep(1)])
        self.out_type = enn.FieldType(self.gs0, out_fields * [self.gs0.fibergroup.regular_repr])

        linear = enn.Linear(self.in_type, self.out_type)
        if soften:
            self.linear = SoftEquiWrapper(linear, self.in_type.size, self.out_type.size, spatial=False)
        else:
            self.linear = linear
        self.relu = enn.ReLU(self.out_type, inplace=True)
        self.soften = soften

    def forward(self, state: torch.Tensor) -> enn.GeometricTensor:
        """
        Args:
            state: (B, 2) raw torch tensor
        Returns:
            GeometricTensor with tensor shape (B, out_fields * N)
        """
        x = enn.GeometricTensor(state, self.in_type)
        out = self.linear(x)
        if self.soften:
            out = enn.GeometricTensor(out, self.out_type)
        return self.relu(out)


# ---------------------------------------------------------------------------
# Action encoder (non-spatial, processes each timestep independently)
# ---------------------------------------------------------------------------

class EquiActionEncoder(nn.Module):
    """
    Encodes 2D action vectors equivariantly, per timestep.

    Processes all (B * T_a) timesteps in one batch via shared escnn.nn.Linear weights,
    then reshapes the output for per-group-element processing by the U-Net.

    Output shape: (B * N, T_a, out_fields)
        — ready to be passed directly to ConditionalUnet1D as the action sequence.
    """

    def __init__(self, N: int = 8, out_fields: int = 32, soften: bool = False):
        super().__init__()
        self.N = N
        self.out_fields = out_fields
        group = gspaces.rot2dOnR2(N).fibergroup
        self.gs0 = gspaces.no_base_space(group)

        self.in_type = enn.FieldType(self.gs0, [self.gs0.fibergroup.irrep(1)])
        self.out_type = enn.FieldType(self.gs0, out_fields * [self.gs0.fibergroup.regular_repr])

        linear = enn.Linear(self.in_type, self.out_type)
        if soften:
            self.linear = SoftEquiWrapper(linear, self.in_type.size, self.out_type.size, spatial=False)
        else:
            self.linear = linear
        self.relu = enn.ReLU(self.out_type, inplace=True)
        self.soften = soften

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T_a, 2) raw torch tensor
        Returns:
            (B * N, T_a, out_fields) — per-group-element action features
        """
        B, T_a, _ = actions.shape

        # Flatten timesteps: (B*T_a, 2)
        flat = actions.reshape(B * T_a, 2)
        x = enn.GeometricTensor(flat, self.in_type)
        out = self.linear(x)
        if self.soften:
            out = enn.GeometricTensor(out, self.out_type)
        out = self.relu(out)

        # out.tensor: (B*T_a, out_fields * N), field-major layout
        # Reshape to (B, T_a, out_fields, N) then rearrange for per-group processing
        feat = out.tensor.reshape(B, T_a, self.out_fields, self.N)
        # → (B, N, T_a, out_fields) for per-group-element sequences
        feat = feat.permute(0, 3, 1, 2)
        # → (B*N, T_a, out_fields)
        return feat.reshape(B * self.N, T_a, self.out_fields)
