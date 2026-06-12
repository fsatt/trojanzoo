#!/usr/bin/env python3

r"""
Usage:
    cd trojanzoo/
    python examples/backdoor_attack.py --pretrained --epochs 50 --lr 0.01 \
        --mark_path <imc_eot_trigger.png> --mark_height 6 --mark_width 6 \
        --eot_rotation_max 15 --eot_scale_min 0.7 --eot_scale_max 1.0 \
        --eot_distortion_scale 0.5 --attack imc_offline --poison_percent 0.10
"""

import math
import random

import torch

from .badnet_aug import BadNet_Aug


class IMC_Offline(BadNet_Aug):
    r"""Offline dataset poisoning with a physically-robust trigger.

    Like :class:`BadNet_Aug`, but builds a fixed pool of EOT-triggered images
    once (size ``poison_ratio x |train set|``, one sampled transform each) and
    mixes it into training, instead of augmenting each batch live. The trigger
    is supplied via ``--mark_path`` and used as-is (no optimisation). EOT
    parameters and ``poison_percent`` are inherited from :class:`BadNet_Aug`.
    """

    name: str = 'imc_offline'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._poison_imgs: torch.Tensor | None = None
        self._poison_labels: torch.Tensor | None = None

    def attack(self, epochs: int, **kwargs):
        self._build_pool()
        return super().attack(epochs, **kwargs)

    @torch.no_grad()
    def _build_pool(self):
        """Pre-generate a fixed pool of EOT-triggered images, one transform each."""
        train_size = len(self.dataset.loader['train'].dataset)
        pool_size  = max(1, int(self.poison_ratio * train_size))

        print(f'\n[IMC_Offline] Building poison pool: '
              f'{pool_size} images ({self.poison_ratio:.1%} of {train_size})')

        poison    = []
        collected = 0
        for data in self.dataset.loader['train']:
            if collected >= pool_size:
                break
            imgs, _ = self.model.get_data(data)
            for img in imgs:
                if collected >= pool_size:
                    break
                poisoned = self._single_eot_add_mark(img.unsqueeze(0)).squeeze(0)
                poison.append(poisoned.cpu())
                collected += 1

        self._poison_imgs   = torch.stack(poison)
        self._poison_labels = torch.full(
            (pool_size,), self.target_class, dtype=torch.long)
        print(f'[IMC_Offline] Pool ready — '
              f'{pool_size} × {list(self._poison_imgs.shape[1:])}')

    def get_data(self, data, keep_org: bool = True,
                 poison_label: bool = True, org: bool = False, **kwargs):
        """Mix the frozen poison pool into each training batch.

        Defences also call this method to probe the model (``keep_org=False``).
        In that case the pool is bypassed and the parent composites the trigger
        live, so the defence evaluates its own candidate trigger and target class
        instead of the frozen training-time pool.
        """
        _input, _label = self.model.get_data(data)

        if org:
            return _input, _label

        if not keep_org:
            return super().get_data(data, keep_org=False,
                                    poison_label=poison_label, org=False, **kwargs)

        if self._poison_imgs is None or self._poison_labels is None:
            return _input, _label

        # Bernoulli-replace a poison_ratio fraction of the batch with pool samples.
        batch_size = len(_label)
        decimal, integer = math.modf(batch_size * self.poison_ratio)
        integer = int(integer)
        if random.random() < decimal:
            integer += 1
        integer = min(integer, batch_size)

        if integer:
            replace_idx = torch.randperm(batch_size, device=_input.device)[:integer]
            pool_idx = torch.randint(len(self._poison_imgs), (integer,))
            _input[replace_idx] = self._poison_imgs[pool_idx].to(_input.device)
            if poison_label:
                _label[replace_idx] = self._poison_labels[pool_idx].to(_label.device)

        return _input, _label
