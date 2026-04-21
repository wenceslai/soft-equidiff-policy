"""
Camera tilt augmentation for the Push-T experiments.

Applies a perspective warp to simulate a tilted overhead camera.
A tilt of θ degrees shrinks the image horizontally by cos(θ),
approximating the foreshortening from a non-perpendicular viewpoint.
"""

import math
import numpy as np
import torch
import torchvision.transforms.functional as TF


def apply_camera_tilt(image: torch.Tensor, tilt_degrees: float) -> torch.Tensor:
    """
    Apply a horizontal perspective warp to simulate a tilted camera.

    The warp compresses the image horizontally by a factor of cos(tilt_degrees),
    creating the appearance of viewing the scene from a tilted angle.

    Args:
        image:         (..., 3, H, W) tensor in [0, 1] or [0, 255]
        tilt_degrees:  tilt angle in degrees (0 = perpendicular / no tilt)
    Returns:
        warped image with same shape as input
    """
    if tilt_degrees == 0.0:
        return image

    # Handle arbitrary leading batch dims by flattening to (N, C, H, W)
    orig_shape = image.shape
    img = image.reshape(-1, *orig_shape[-3:])
    H, W = img.shape[-2], img.shape[-1]

    shrink = math.cos(math.radians(tilt_degrees))
    offset = (1.0 - shrink) / 2.0 * W

    # Perspective: compress left/right edges inward at the top only
    startpoints = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
    endpoints = [
        [int(offset),         0],
        [int(W - 1 - offset), 0],
        [W - 1,               H - 1],
        [0,                   H - 1],
    ]

    warped = torch.stack(
        [TF.perspective(img[i], startpoints, endpoints) for i in range(img.shape[0])]
    )
    return warped.reshape(orig_shape)


class CameraTiltTransform:
    """
    Callable transform compatible with LeRobot's image_transforms argument.

    Can be passed to LeRobotDataset(image_transforms=CameraTiltTransform(30)).
    """

    def __init__(self, tilt_degrees: float):
        self.tilt_degrees = tilt_degrees

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return apply_camera_tilt(image, self.tilt_degrees)

    def __repr__(self):
        return f"CameraTiltTransform(tilt_degrees={self.tilt_degrees})"


def make_tilt_transform(tilt_degrees: float):
    """Factory — returns None for 0° (no-op), CameraTiltTransform otherwise."""
    if tilt_degrees == 0.0:
        return None
    return CameraTiltTransform(tilt_degrees)
