#!/usr/bin/env python3

r"""
Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py \
        --color --verbose 1 --pretrained --validate_interval 1 \
        --epochs 50 --lr 0.01 --mark_random_init \
        --mark_height 8 --mark_width 8 --mark_alpha 1.0 \
        --eot_rotation_max 15 --eot_scale_min 0.7 --eot_scale_max 1.0 \
        --eot_distortion_scale 0.5 \
        --attack imc_eot
"""

from .imc import IMC
from trojanvision.utils.eot import sample_eot_params, apply_eot_to_patch
from trojanzoo.utils.tensor import tanh_func

import torch
import torch.optim as optim
import argparse
import random

from collections.abc import Callable


class IMC_EOT(IMC):
    r"""IMC with EOT: the trigger is optimised with EOT-averaged gradients and
    each poisoned batch is composited under one random physical transform.

    Args:
        eot_rotation_max (float): Max rotation degrees. Defaults to ``15.0``.
        eot_distortion_scale (float): Perspective distortion in [0, 1].
            Defaults to ``0.5``.
        eot_scale_min (float): Min patch scale. Defaults to ``0.7``.
        eot_scale_max (float): Max patch scale. Defaults to ``1.0``.
        eot_n_samples (int): Transforms to average per gradient step.
            Defaults to ``4``.
        eot_clean_mix_ratio (float): Fraction of training batches that use
            the clean (unaugmented) trigger. ``0.5`` recovers clean-trigger
            ASR without sacrificing physical robustness. Defaults to ``0.0``.
    """

    name: str = 'imc_eot'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--eot_rotation_max', type=float,
                           help='max rotation degrees for EOT (default: 15.0)')
        group.add_argument('--eot_distortion_scale', type=float,
                           help='perspective distortion scale for EOT (default: 0.5)')
        group.add_argument('--eot_scale_min', type=float,
                           help='min patch scale for EOT (default: 0.7)')
        group.add_argument('--eot_scale_max', type=float,
                           help='max patch scale for EOT (default: 1.0)')
        group.add_argument('--eot_n_samples', type=int,
                           help='transforms averaged per gradient step (default: 4)')
        group.add_argument('--eot_clean_mix_ratio', type=float,
                           help='fraction of training batches using the clean trigger '
                                '(default: 0.0)')
        return group

    def __init__(self,
                 eot_rotation_max: float = 15.0,
                 eot_distortion_scale: float = 0.5,
                 eot_scale_min: float = 0.7,
                 eot_scale_max: float = 1.0,
                 eot_n_samples: int = 4,
                 eot_clean_mix_ratio: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.param_list['imc_eot'] = [
            'eot_rotation_max', 'eot_distortion_scale',
            'eot_scale_min', 'eot_scale_max',
            'eot_n_samples', 'eot_clean_mix_ratio',
        ]
        self.eot_rotation_max     = eot_rotation_max
        self.eot_distortion_scale = eot_distortion_scale
        self.eot_scale_min        = eot_scale_min
        self.eot_scale_max        = eot_scale_max
        self.eot_n_samples        = eot_n_samples
        self.eot_clean_mix_ratio  = eot_clean_mix_ratio
    
    # EOT compositing helpers
    def _composite(self, images: torch.Tensor,
                   t_rgb: torch.Tensor, t_alpha: torch.Tensor) -> torch.Tensor:
        """Composite a transformed trigger patch onto a batch of images."""
        hs = self.mark.mark_height_offset
        ws = self.mark.mark_width_offset
        he = hs + self.mark.mark_height
        we = ws + self.mark.mark_width
        out = images.clone()
        org = images[..., hs:he, ws:we]
        out[..., hs:he, ws:we] = org + t_alpha * (t_rgb - org)
        return out

    def _eot_add_mark(self, _input: torch.Tensor,
                      mark_alpha: float = None,
                      mark_random_pos: bool = None,
                      **kwargs) -> torch.Tensor:
        """Average N EOT-transformed composites for gradient estimation."""
        alpha        = mark_alpha if mark_alpha is not None else self.mark.mark_alpha
        mark_rgb     = self.mark.mark[:-1]
        mark_alpha_ch = self.mark.mark[-1:] * alpha
        h, w = self.mark.mark_height, self.mark.mark_width

        accumulated = torch.zeros_like(_input)
        for _ in range(self.eot_n_samples):
            params = sample_eot_params(h, w, self.eot_rotation_max,
                                       self.eot_scale_min, self.eot_scale_max,
                                       self.eot_distortion_scale)
            t_rgb, t_alpha = apply_eot_to_patch(mark_rgb, mark_alpha_ch,
                                                params, self.eot_distortion_scale)
            accumulated = accumulated + self._composite(_input, t_rgb, t_alpha)
        return accumulated / self.eot_n_samples

    def _single_eot_add_mark(self, _input: torch.Tensor,
                              mark_alpha: float = None,
                              mark_random_pos: bool = None,
                              **kwargs) -> torch.Tensor:
        """Apply ONE random EOT transform per batch (data-augmentation style)."""
        alpha        = mark_alpha if mark_alpha is not None else self.mark.mark_alpha
        mark_rgb     = self.mark.mark[:-1]
        mark_alpha_ch = self.mark.mark[-1:] * alpha
        h, w = self.mark.mark_height, self.mark.mark_width

        params = sample_eot_params(h, w, self.eot_rotation_max,
                                   self.eot_scale_min, self.eot_scale_max,
                                   self.eot_distortion_scale)
        t_rgb, t_alpha = apply_eot_to_patch(mark_rgb, mark_alpha_ch,
                                            params, self.eot_distortion_scale)
        return self._composite(_input, t_rgb, t_alpha)

    # Overrides
    def optimize_mark(self, loss_fn: Callable[..., torch.Tensor] = None, **kwargs):
        """EOT trigger optimisation: gradient is an expectation over transforms."""
        loss_fn = loss_fn or self.model.loss
        self.mark.add_mark_fn = self._eot_add_mark
        try:
            atanh_mark = (self.mark.mark[:-1]
                          .mul(2).sub(1).mul(0.999).atanh()
                          .detach().requires_grad_())
            optimizer = optim.Adam([atanh_mark], lr=self.attack_remask_lr)
            optimizer.zero_grad()

            for _ in range(self.attack_remask_epochs):
                for data in self.dataset.loader['train']:
                    self.mark.mark[:-1] = tanh_func(atanh_mark)
                    _input, _label = self.model.get_data(data)
                    trigger_input  = self.add_mark(_input)
                    trigger_label  = self.target_class * torch.ones_like(_label)
                    loss = loss_fn(trigger_input, trigger_label)
                    loss.backward(inputs=[atanh_mark], retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()
                    self.mark.mark.detach_()

            atanh_mark.requires_grad_(False)
            self.mark.mark[:-1] = tanh_func(atanh_mark)
            self.mark.mark.detach_()
        finally:
            self.mark.add_mark_fn = None

    def get_data(self, data, **kwargs):
        """Apply a single random EOT transform to poisoned batches during training."""
        if random.random() >= self.eot_clean_mix_ratio:
            self.mark.add_mark_fn = self._single_eot_add_mark
        try:
            result = super().get_data(data, **kwargs)
        finally:
            self.mark.add_mark_fn = None
        return result
