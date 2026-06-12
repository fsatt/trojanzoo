#!/usr/bin/env python3

r"""
Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/backdoor_defense.py --color --verbose 1 \
        --pretrained --attack imc_eot --defense neural_cleanse_eot \
        --nc_eot_warmup_epochs 5 --nc_eot_robust_epochs 5
"""

from .neural_cleanse import NeuralCleanse
from trojanvision.utils.eot import sample_eot_params, apply_eot_to_patch
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.tensor import tanh_func

import torch
import torch.optim as optim
import argparse


class NeuralCleanse_EOT(NeuralCleanse):
    r"""Neural Cleanse with two-phase warm-started EOT reconstruction.

    Args:
        nc_eot_rotation_max (float): Max rotation degrees.  Defaults to ``15.0``.
        nc_eot_distortion_scale (float): Perspective distortion in [0,1].
            Defaults to ``0.5``.
        nc_eot_scale_min (float): Min patch scale.  Defaults to ``0.7``.
        nc_eot_scale_max (float): Max patch scale.  Defaults to ``1.0``.
        nc_eot_n_samples (int): EOT samples per forward pass during
            the robust phase.  Defaults to ``4``.
        nc_eot_warmup_epochs (int): Standard-NC outer epochs run before
            EOT is enabled.  Defaults to ``5``.
        nc_eot_robust_epochs (int): EOT-aware outer epochs run after the
            warmup completes.  Defaults to ``5``.
    """

    name: str = 'neural_cleanse_eot'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--nc_eot_rotation_max', type=float,
                           help='max rotation degrees during reconstruction (default: 15.0)')
        group.add_argument('--nc_eot_distortion_scale', type=float,
                           help='perspective distortion scale (default: 0.5)')
        group.add_argument('--nc_eot_scale_min', type=float,
                           help='min scale factor (default: 0.7)')
        group.add_argument('--nc_eot_scale_max', type=float,
                           help='max scale factor (default: 1.0)')
        group.add_argument('--nc_eot_n_samples', type=int,
                           help='EOT samples per forward pass during the robust phase (default: 4)')
        group.add_argument('--nc_eot_warmup_epochs', type=int,
                           help='standard-NC outer epochs before EOT is enabled (default: 5)')
        group.add_argument('--nc_eot_robust_epochs', type=int,
                           help='EOT-aware outer epochs after warmup (default: 5)')
        return group

    def __init__(self,
                 nc_eot_rotation_max: float = 15.0,
                 nc_eot_distortion_scale: float = 0.5,
                 nc_eot_scale_min: float = 0.7,
                 nc_eot_scale_max: float = 1.0,
                 nc_eot_n_samples: int = 4,
                 nc_eot_warmup_epochs: int = 5,
                 nc_eot_robust_epochs: int = 5,
                 **kwargs):
        super().__init__(**kwargs)
        self.param_list['neural_cleanse_eot'] = [
            'eot_rotation_max', 'eot_distortion_scale',
            'eot_scale_min', 'eot_scale_max', 'nc_eot_n_samples',
            'nc_eot_warmup_epochs', 'nc_eot_robust_epochs',
        ]
        self.eot_rotation_max = nc_eot_rotation_max
        self.eot_distortion_scale = nc_eot_distortion_scale
        self.eot_scale_min = nc_eot_scale_min
        self.eot_scale_max = nc_eot_scale_max
        self.nc_eot_n_samples = nc_eot_n_samples
        self.nc_eot_warmup_epochs = nc_eot_warmup_epochs
        self.nc_eot_robust_epochs = nc_eot_robust_epochs

    def get_filename(self, **kwargs) -> str:
        # Tag warmup/robust epochs so different configs write to distinct files.
        base = super().get_filename(**kwargs)
        return f"{base}_w{self.nc_eot_warmup_epochs}_r{self.nc_eot_robust_epochs}"

    # EOT compositing (for the robust phase)
    def _reconstruction_eot_add_mark(
        self,
        _input: torch.Tensor,
        mark_alpha: float = None,
        mark_random_pos: bool = None,
        **kwargs,
    ) -> torch.Tensor:
        mark = self.attack.mark
        alpha = mark_alpha if mark_alpha is not None else mark.mark_alpha
        mark_rgb = mark.mark[:-1]
        mark_alpha_ch = mark.mark[-1:] * alpha
        h_start = mark.mark_height_offset
        w_start = mark.mark_width_offset
        h = mark.mark_height
        w = mark.mark_width
        h_end = h_start + h
        w_end = w_start + w

        accumulated = torch.zeros_like(_input)
        for _ in range(self.nc_eot_n_samples):
            params = sample_eot_params(h, w, self.eot_rotation_max,
                                       self.eot_scale_min, self.eot_scale_max,
                                       self.eot_distortion_scale)
            t_rgb, t_alpha = apply_eot_to_patch(mark_rgb, mark_alpha_ch,
                                                params, self.eot_distortion_scale)
            triggered = _input.clone()
            org_patch = _input[..., h_start:h_end, w_start:w_end]
            triggered[..., h_start:h_end, w_start:w_end] = (
                org_patch + t_alpha * (t_rgb - org_patch)
            )
            accumulated = accumulated + triggered
        return accumulated / self.nc_eot_n_samples

    # Per-class two-phase optimisation
    def _reset_cost_state(self):
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        self.early_stop_counter = 0
        self.early_stop_norm_best = float('inf')

    def optimize_mark(self, label: int,
                      loader=None,
                      logger_header: str = '',
                      verbose: bool = True,
                      **kwargs) -> tuple[torch.Tensor, float]:
        total_epochs = self.nc_eot_warmup_epochs + self.nc_eot_robust_epochs
        if total_epochs <= 0:
            raise ValueError(
                "nc_eot_warmup_epochs + nc_eot_robust_epochs must be > 0"
            )

        atanh_mark = torch.randn_like(self.attack.mark.mark, requires_grad=True)
        optimizer = optim.Adam([atanh_mark], lr=self.defense_remask_lr,
                               betas=(0.5, 0.9))
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs)
        optimizer.zero_grad()
        loader = loader or self.dataset.loader['train']

        self._reset_cost_state()
        self.cost = self.init_cost

        norm_best: float = float('inf')
        mark_best: torch.Tensor = None
        loss_best: float = None

        logger = MetricLogger(indent=4)
        logger.create_meters(loss='{last_value:.3f}',
                             acc='{last_value:.3f}',
                             norm='{last_value:.3f}',
                             entropy='{last_value:.3f}',)
        batch_logger = MetricLogger()
        logger.create_meters(loss=None, acc=None, entropy=None)

        iterator = range(total_epochs)
        if verbose:
            iterator = logger.log_every(iterator, header=logger_header)

        for epoch_idx in iterator:
            in_warmup = epoch_idx < self.nc_eot_warmup_epochs

            # At the phase transition, reset best-tracking and cost so the EOT
            # phase is evaluated on its own terms (not the warmup minimum).
            if (epoch_idx == self.nc_eot_warmup_epochs
                    and self.nc_eot_warmup_epochs > 0
                    and self.nc_eot_robust_epochs > 0):
                norm_best = float('inf')
                mark_best = None
                loss_best = None
                self._reset_cost_state()
                self.cost = self.init_cost

            if in_warmup:
                self.attack.mark.add_mark_fn = None
            else:
                self.attack.mark.add_mark_fn = self._reconstruction_eot_add_mark

            batch_logger.reset()
            for data in loader:
                self.attack.mark.mark = tanh_func(atanh_mark)
                _input, _label = self.model.get_data(data)
                trigger_input = self.attack.add_mark(_input)
                trigger_label = label * torch.ones_like(_label)
                trigger_output = self.model(trigger_input)

                batch_acc = trigger_label.eq(
                    trigger_output.argmax(1)).float().mean()
                batch_entropy = self.loss(
                    _input, _label, target=label,
                    trigger_output=trigger_output, **kwargs)
                batch_norm: torch.Tensor = self.attack.mark.mark[-1].norm(p=1)
                batch_loss = batch_entropy + self.cost * batch_norm

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_size = _label.size(0)
                batch_logger.update(n=batch_size,
                                    loss=batch_loss.item(),
                                    acc=batch_acc.item(),
                                    entropy=batch_entropy.item())
            lr_scheduler.step()
            self.attack.mark.mark = tanh_func(atanh_mark)

            loss = batch_logger.meters['loss'].global_avg
            acc = batch_logger.meters['acc'].global_avg
            norm = float(self.attack.mark.mark[-1].norm(p=1))
            entropy = batch_logger.meters['entropy'].global_avg
            if norm < norm_best:
                mark_best = self.attack.mark.mark.detach().clone()
                loss_best = loss
                logger.update(loss=loss, acc=acc, norm=norm, entropy=entropy)

            # Early-stop only in the EOT phase (warmup always runs fully).
            if (not in_warmup
                    and self.check_early_stop(loss=loss, acc=acc,
                                              norm=norm, entropy=entropy)):
                print('early stop')
                break

        atanh_mark.requires_grad_(False)
        if mark_best is None:
            mark_best = self.attack.mark.mark.detach().clone()
            loss_best = loss
        self.attack.mark.mark = mark_best
        self.attack.mark.add_mark_fn = None
        return mark_best, loss_best
