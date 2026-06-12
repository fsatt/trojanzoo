#!/usr/bin/env python3

r"""
Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py \
        --color --verbose 1 --pretrained --validate_interval 1 \
        --epochs 50 --lr 0.01 --mark_random_init \
        --mark_height 12 --mark_width 12 --mark_alpha 1.0 \
        --attack imc_eot_adapt --adapt_to_strip
"""

from .imc_eot import IMC_EOT

import torch
import torch.optim as optim
import argparse


class IMC_EOT_Adapt(IMC_EOT):
    r"""IMC+EOT with a post-training STRIP-evasion fine-tune (``--adapt_to_strip``):
    the model is fine-tuned on half-opacity triggered inputs labelled with their
    true class, defeating STRIP's superimposition test.

    Args:
        adapt_to_strip (bool): Enable STRIP evasion. Defaults to ``False``.
        adapt_strip_alpha (float): Trigger opacity for STRIP fine-tuning.
            Defaults to ``None`` (resolves to ``mark_alpha * 0.5`` at
            fine-tune time).
        adapt_strip_weight (float): Weight of the STRIP evasion loss relative
            to the clean accuracy loss. Defaults to ``1.0``.
        adapt_backdoor_weight (float): Weight of the backdoor-preservation
            loss term added during STRIP fine-tuning. Without this term the
            half-alpha clean-label training erases the backdoor entirely.
            Defaults to ``5.0`` (favours preserving the backdoor over
            evading STRIP — backdoor erasure is the worse failure mode).
        adapt_strip_epochs (int): Fine-tuning epochs for STRIP evasion.
            Defaults to ``5``.
        adapt_strip_lr (float): Learning rate for STRIP fine-tuning.
            Defaults to ``1e-4``.
    """

    name: str = 'imc_eot_adapt'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--adapt_to_strip', action='store_true',
                           help='enable STRIP evasion via half-alpha fine-tuning')
        group.add_argument('--adapt_strip_alpha', type=float,
                           help='trigger opacity for STRIP fine-tuning '
                                '(default: mark_alpha * 0.5)')
        group.add_argument('--adapt_strip_weight', type=float,
                           help='STRIP evasion loss weight vs clean loss (default: 1.0)')
        group.add_argument('--adapt_backdoor_weight', type=float,
                           help='backdoor-preservation loss weight during STRIP '
                                'fine-tuning (default: 5.0)')
        group.add_argument('--adapt_strip_epochs', type=int,
                           help='STRIP fine-tuning epochs (default: 5)')
        group.add_argument('--adapt_strip_lr', type=float,
                           help='learning rate for STRIP fine-tuning (default: 1e-4)')
        return group

    def __init__(self,
                 adapt_to_strip: bool = False,
                 adapt_strip_alpha: float = None,
                 adapt_strip_weight: float = 1.0,
                 adapt_backdoor_weight: float = 5.0,
                 adapt_strip_epochs: int = 5,
                 adapt_strip_lr: float = 1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.param_list['imc_eot_adapt'] = [
            'adapt_to_strip',
            'adapt_strip_alpha', 'adapt_strip_weight', 'adapt_backdoor_weight',
            'adapt_strip_epochs', 'adapt_strip_lr',
        ]
        self.adapt_to_strip = adapt_to_strip
        self.adapt_strip_alpha = adapt_strip_alpha
        self.adapt_strip_weight = adapt_strip_weight
        self.adapt_backdoor_weight = adapt_backdoor_weight
        self.adapt_strip_epochs = adapt_strip_epochs
        self.adapt_strip_lr = adapt_strip_lr

    # STRIP evasion: post-training fine-tuning
    def attack(self, epochs: int, **kwargs):
        result = super().attack(epochs, **kwargs)
        if self.adapt_to_strip:
            self._strip_adaptation()
        return result

    def _strip_adaptation(self):
        """Fine-tune the model to ignore triggers at the opacity STRIP tests."""
        strip_alpha = (self.adapt_strip_alpha
                       if self.adapt_strip_alpha is not None
                       else self.mark.mark_alpha * 0.5)

        print(f'\n[IMC_EOT_Adapt] STRIP adaptation: '
              f'{self.adapt_strip_epochs} epochs, '
              f'trigger alpha={strip_alpha:.3f}, '
              f'lr={self.adapt_strip_lr}')

        self.model.train()
        # Re-enable grads (the trainer froze params after the main run).
        self.model.requires_grad_(True)
        # Freeze BatchNorm (eval + no grad) so the backdoor's signal path is unchanged during fine-tuning.
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.adapt_strip_lr,
        )

        for epoch in range(self.adapt_strip_epochs):
            total_clean = 0.0
            total_strip = 0.0
            total_bd    = 0.0
            n_batches = 0
            for data in self.dataset.loader['train']:
                _input, _label = self.model.get_data(data)
                optimizer.zero_grad()

                # Clean -> true label (keep accuracy).
                clean_loss = self.model.loss(_input, _label)
                # Half-alpha trigger -> true label (evade STRIP).
                strip_input = self.add_mark(_input, mark_alpha=strip_alpha)
                strip_loss  = self.model.loss(strip_input, _label)
                # Full-alpha trigger -> target (preserve the backdoor).
                bd_input = self.add_mark(_input)
                bd_label = self.target_class * torch.ones_like(_label)
                bd_loss  = self.model.loss(bd_input, bd_label)

                loss = (clean_loss
                        + self.adapt_strip_weight   * strip_loss
                        + self.adapt_backdoor_weight * bd_loss)
                loss.backward()
                optimizer.step()
                total_clean += clean_loss.item()
                total_strip += strip_loss.item()
                total_bd    += bd_loss.item()
                n_batches += 1

            print(f'  epoch {epoch + 1}/{self.adapt_strip_epochs}  '
                  f'clean={total_clean / n_batches:.4f}  '
                  f'strip={total_strip / n_batches:.4f}  '
                  f'bd={total_bd / n_batches:.4f}')

        self.model.eval()
        print('[IMC_EOT_Adapt] STRIP adaptation complete.\n')
