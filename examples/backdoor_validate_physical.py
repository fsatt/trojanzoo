#!/usr/bin/env python3

r"""
Evaluate a backdoored model's ASR under physical (EOT) transforms.

The trigger patch is composited onto each validation image under one randomly
sampled EOT transform (rotation, scale, perspective distortion); there is no
averaging, so the model sees one distorted trigger per image. ASR is reported
as the mean (and standard deviation) over n_reps independent passes for
variance reduction.

Usage:
    python examples/backdoor_validate_physical.py --verbose 0 --pretrained \
        --dataset cifar10 --model resnet18_comp \
        --attack badnet --mark_height 6 --mark_width 6 --mark_alpha 1.0 \
        [--physical_rotation_max 15.0] [--physical_distortion_scale 0.5] \
        [--physical_scale_min 0.7] [--physical_scale_max 1.0] \
        [--physical_n_reps 5] [--physical_n_eval 1000]
"""

import trojanvision
import argparse
import numpy as np
import torch

from trojanvision.utils.eot import sample_eot_params, apply_eot_to_patch


def _composite_patch(images: torch.Tensor,
                     t_rgb: torch.Tensor, t_alpha: torch.Tensor,
                     hs: int, he: int, ws: int, we: int) -> torch.Tensor:
    out = images.clone()
    org = images[..., hs:he, ws:we]
    out[..., hs:he, ws:we] = org + t_alpha * (t_rgb - org)
    return out


def apply_trigger(images: torch.Tensor, attack,
                  rotation_max: float, scale_min: float, scale_max: float,
                  distortion_scale: float) -> torch.Tensor:
    """Composite the trigger patch onto each image under one random EOT transform."""
    mark = attack.mark
    mark_rgb   = mark.mark[:-1].clone()
    mark_alpha = mark.mark[-1:].clone() * mark.mark_alpha
    hs, ws = mark.mark_height_offset, mark.mark_width_offset
    h, w   = mark.mark_height, mark.mark_width
    he, we = hs + h, ws + w

    out = images.clone()
    for i in range(images.shape[0]):
        p = sample_eot_params(h, w, rotation_max, scale_min, scale_max, distortion_scale)
        t_rgb, t_alpha = apply_eot_to_patch(mark_rgb, mark_alpha, p, distortion_scale)
        out[i:i+1] = _composite_patch(images[i:i+1], t_rgb, t_alpha, hs, he, ws, we)
    return out


def evaluate(attack, n_eval: int,
             rotation_max: float, scale_min: float, scale_max: float,
             distortion_scale: float, n_reps: int):
    model   = attack.model
    dataset = attack.dataset
    target  = attack.target_class

    clean_correct = 0
    asr_clean     = 0
    total         = 0
    asr_phys_runs = [0] * n_reps

    model.eval()
    with torch.no_grad():
        for data in dataset.loader['valid']:
            if total >= n_eval:
                break
            _input, _label = model.get_data(data)
            N = _input.size(0)
            target_label = torch.full_like(_label, target)

            clean_correct += model(_input).argmax(1).eq(_label).sum().item()

            trigger_clean = attack.add_mark(_input)
            asr_clean    += model(trigger_clean).argmax(1).eq(target_label).sum().item()

            for r in range(n_reps):
                triggered = apply_trigger(
                    _input, attack,
                    rotation_max, scale_min, scale_max, distortion_scale)
                asr_phys_runs[r] += model(triggered).argmax(1).eq(target_label).sum().item()

            total += N

    n = min(total, n_eval)
    asr_phys_counts = [c / n for c in asr_phys_runs]
    asr_phys_mean   = float(np.mean(asr_phys_counts))
    asr_phys_std    = float(np.std(asr_phys_counts)) if n_reps > 1 else 0.0

    print(f'\nResults over {n} validation images  [{n_reps} reps]:')
    print(f'  Clean accuracy      : {100 * clean_correct / n:.2f}%')
    print(f'  ASR (no transform)  : {100 * asr_clean / n:.2f}%')
    if n_reps > 1:
        print(f'  ASR (w/ transform)  : {100 * asr_phys_mean:.2f}%  '
              f'(±{100 * asr_phys_std:.2f}% std over {n_reps} reps)')
    else:
        print(f'  ASR (w/ transform)  : {100 * asr_phys_mean:.2f}%')
    print(f'  (rotation±{rotation_max}°, scale [{scale_min},{scale_max}], '
          f'distortion {distortion_scale})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)

    parser.add_argument('--clean_model', action='store_true',
                        help='skip attack.load() — evaluate the pretrained clean model')
    parser.add_argument('--physical_n_reps', type=int, default=5,
                        help='reps for variance reduction (default: 5)')
    parser.add_argument('--physical_n_eval', type=int, default=1000,
                        help='number of validation images (default: 1000)')
    parser.add_argument('--physical_rotation_max', type=float, default=15.0,
                        help='max rotation degrees (default: 15.0)')
    parser.add_argument('--physical_scale_min', type=float, default=0.7,
                        help='min scale factor (default: 0.7)')
    parser.add_argument('--physical_scale_max', type=float, default=1.0,
                        help='max scale factor (default: 1.0)')
    parser.add_argument('--physical_distortion_scale', type=float, default=0.5,
                        help='perspective distortion scale (default: 0.5)')

    kwargs = vars(parser.parse_args())

    clean_model          = kwargs.pop('clean_model')
    physical_n_reps      = kwargs.pop('physical_n_reps')
    physical_n_eval      = kwargs.pop('physical_n_eval')
    eot_rotation_max     = kwargs.pop('physical_rotation_max')
    eot_scale_min        = kwargs.pop('physical_scale_min')
    eot_scale_max        = kwargs.pop('physical_scale_max')
    eot_distortion_scale = kwargs.pop('physical_distortion_scale')

    env     = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model   = trojanvision.models.create(dataset=dataset, **kwargs)
    mark    = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack  = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, attack=attack)

    if not clean_model:
        attack.load()

    evaluate(attack,
             n_eval=physical_n_eval,
             n_reps=physical_n_reps,
             rotation_max=eot_rotation_max,
             scale_min=eot_scale_min,
             scale_max=eot_scale_max,
             distortion_scale=eot_distortion_scale)
