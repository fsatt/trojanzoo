#!/usr/bin/env python3

r"""Shared Expectation over Transformations (EOT) utilities.

Pure functions used by both attacks (IMC_EOT, BadNet_Aug) and defences
(NeuralCleanse_EOT). No class state; callers pass the parameters they need.

EOT was proposed by Athalye et al. for synthesizing physically robust
adversarial examples; we reuse the same transform-averaging idea for
trigger optimisation and reconstruction.

See Also:
    * paper: `Synthesizing Robust Adversarial Examples`_
      (Athalye et al., ICML 2018)

.. _Synthesizing Robust Adversarial Examples:
    https://proceedings.mlr.press/v80/athalye18b.html
"""

import torch
import torchvision.transforms.functional as TF
import numpy as np


def sample_eot_params(h: int, w: int,
                      rotation_max: float,
                      scale_min: float,
                      scale_max: float,
                      distortion_scale: float) -> dict:
    """Sample one set of EOT transform parameters for a patch of size (h, w).

    Args:
        h, w: Patch height and width in pixels.
        rotation_max: Max rotation in degrees (sampled uniformly in [-max, max]).
        scale_min, scale_max: Range for isotropic scaling.
        distortion_scale: Controls how far each corner can shift for perspective
            warp, as a fraction of the half-dimension.

    Returns:
        Dict with keys: angle, scale, x_shift, y_shift, startpoints, endpoints.
    """
    angle = float(np.random.uniform(-rotation_max, rotation_max))
    scale = float(np.random.uniform(scale_min, scale_max))

    padding_h = (h - scale * h) / 2.0
    padding_w = (w - scale * w) / 2.0
    x_shift = float(np.random.uniform(-padding_w, padding_w))
    y_shift = float(np.random.uniform(-padding_h, padding_h))

    half_h = max(h // 2, 1)
    half_w = max(w // 2, 1)
    dh = max(int(distortion_scale * half_h), 0)
    dw = max(int(distortion_scale * half_w), 0)

    topleft  = [np.random.randint(0, dw + 1),              np.random.randint(0, dh + 1)]
    topright = [np.random.randint(max(w - dw - 1, 0), w),  np.random.randint(0, dh + 1)]
    botright = [np.random.randint(max(w - dw - 1, 0), w),  np.random.randint(max(h - dh - 1, 0), h)]
    botleft  = [np.random.randint(0, dw + 1),              np.random.randint(max(h - dh - 1, 0), h)]

    return dict(
        angle=angle, scale=scale,
        x_shift=x_shift, y_shift=y_shift,
        startpoints=[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        endpoints=[topleft, topright, botright, botleft],
    )


def apply_eot_to_patch(
    mark_rgb: torch.Tensor,    # (C, H, W)
    mark_alpha: torch.Tensor,  # (1, H, W)
    params: dict,
    distortion_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one EOT transform (perspective warp + affine) to a trigger patch.

    Args:
        mark_rgb: Trigger pixel values, shape (C, H, W).
        mark_alpha: Trigger alpha mask, shape (1, H, W).
        params: Output of :func:`sample_eot_params`.
        distortion_scale: If > 0, perspective warp is applied first.

    Returns:
        Transformed (mark_rgb, mark_alpha) with the same shapes as inputs.
    """
    if distortion_scale > 0.0:
        # Rare degenerate endpoint configurations (near-collinear corners)
        # produce a rank-deficient lstsq inside torchvision; skip perspective
        # for that sample rather than crashing the whole training run.
        try:
            mark_rgb = TF.perspective(mark_rgb,
                                      params['startpoints'], params['endpoints'],
                                      interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mark_alpha = TF.perspective(mark_alpha,
                                        params['startpoints'], params['endpoints'],
                                        interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        except torch._C._LinAlgError:
            pass  # leave mark_rgb / mark_alpha untransformed for this sample

    affine_kwargs = dict(
        angle=params['angle'],
        translate=[int(params['x_shift']), int(params['y_shift'])],
        scale=params['scale'],
        shear=[0, 0],
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=0,
    )
    mark_rgb   = TF.affine(mark_rgb,   **affine_kwargs)
    mark_alpha = TF.affine(mark_alpha, **affine_kwargs)
    return mark_rgb, mark_alpha
