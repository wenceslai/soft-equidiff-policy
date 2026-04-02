"""
Configuration dataclass for SoftEquiDiffPolicy.

All hyperparameters in one place. The three experimental conditions from the
paper map to:

    EquiDiff-exact:       penalty_mode="constant",       lambda_base=1000.0
    SoftEqui-constant:    penalty_mode="constant",       lambda_base=0.1
    SoftEqui-step:        penalty_mode="step_dependent", lambda_base=0.1  (novel)
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SoftEquiDiffConfig:
    # -----------------------------------------------------------------------
    # Architecture
    # -----------------------------------------------------------------------
    n_rotations: int = 8           # C_N group order (cyclic rotational symmetry)
    enc_n_hidden: int = 64         # regular-repr fields in image encoder output
    state_features: int = 32       # regular-repr fields in state encoder output
    action_features: int = 32      # regular-repr fields per timestep in action encoder/decoder

    # -----------------------------------------------------------------------
    # Diffusion (match LeRobot Push-T defaults)
    # -----------------------------------------------------------------------
    horizon: int = 16              # total action prediction window length
    n_obs_steps: int = 2           # number of stacked observation frames
    n_action_steps: int = 8        # how many actions to execute per inference call
    num_diffusion_steps: int = 100
    num_inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"   # predict added noise (not x0)
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # -----------------------------------------------------------------------
    # U-Net
    # -----------------------------------------------------------------------
    unet_down_dims: Tuple[int, ...] = (256, 512, 1024)
    unet_diffusion_step_embed_dim: int = 128
    unet_kernel_size: int = 5
    unet_n_groups: int = 8

    # -----------------------------------------------------------------------
    # Soft equivariance — which layers to soften
    # -----------------------------------------------------------------------
    soften_image_encoder: bool = True
    soften_state_encoder: bool = True
    soften_action_encoder: bool = True
    soften_decoder: bool = True

    # -----------------------------------------------------------------------
    # Equivariance penalty schedule
    # -----------------------------------------------------------------------
    # mode: "constant" or "step_dependent"
    penalty_mode: str = "step_dependent"
    # Base penalty strength (tune this; 0 = no penalty, 1000 = near-exact equivariance)
    lambda_base: float = 0.1
    # Exponent for step_dependent mode: lambda(k) = lambda_base * (k/K)^power
    # power=1.0 → linear; power=2.0 → quadratic (stronger push toward equivariance at high noise)
    lambda_power: float = 1.0

    # -----------------------------------------------------------------------
    # Normalisation (set from dataset stats at runtime)
    # -----------------------------------------------------------------------
    # If None, no normalisation is applied. Otherwise set to dataset stats dict.
    # Expected keys: "observation.image", "observation.state", "action"
    # Each value: {"min": tensor, "max": tensor} or {"mean": tensor, "std": tensor}
    normalize_inputs: bool = True

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 64
    num_train_steps: int = 200_000
    grad_clip_norm: float = 10.0

    # -----------------------------------------------------------------------
    # Push-T environment / dataset
    # -----------------------------------------------------------------------
    image_size: int = 96       # Push-T images are 96×96
    action_dim: int = 2        # (x, y) end-effector position
    state_dim: int = 2         # (x, y) end-effector position
    dataset_repo_id: str = "lerobot/pusht"

    # -----------------------------------------------------------------------
    # Camera tilt experiment
    # -----------------------------------------------------------------------
    camera_tilt_degrees: float = 0.0   # 0, 30, or 45 in experiments
