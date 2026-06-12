# TrojanZoo (fork)
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/images/trojanzoo-logo-readme-dark.svg">
  <img alt="TrojanZoo logo" src="docs/source/images/trojanzoo-logo-readme.svg">
</picture>

[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](https://opensource.org/licenses/GPL-3.0)
![python>=3.11](https://img.shields.io/badge/python->=3.11-informational.svg)

> This repository is a **fork of [TrojanZoo](https://github.com/ain-soph/trojanzoo)**
> (Pang et al., EuroS&P 2022, [paper](https://arxiv.org/abs/2012.09302)), a PyTorch
> platform for backdoor attack/defence research on image classifiers.
>
> **For installation, quick start, and full usage of the framework, see the
> [upstream repository](https://github.com/ain-soph/trojanzoo) and its
> [documentation](https://ain-soph.github.io/trojanzoo/).** This README documents
> only what the fork adds.

## What this fork adds

This fork studies **physically robust backdoor attacks** (triggers that survive
camera-style geometric distortion: rotation, scaling, perspective warp) and how
that robustness affects their detectability. It adds:

| Type | `--attack` / `--defense` | Description |
|---|---|---|
| Attack | `badnet_aug` | BadNet with per-batch EOT augmentation during training |
| Attack | `imc_eot` | IMC with EOT-averaged trigger optimisation + augmentation |
| Attack | `imc_eot_adapt` | IMC+EOT with a post-training STRIP-evasion fine-tune |
| Attack | `imc_offline` | Offline (data-only) pool poisoning with a frozen trigger |
| Defence | `neural_cleanse_eot` | Transform-aware Neural Cleanse (EOT-aware reconstruction) |

Plus two supporting pieces:

- `trojanvision/utils/eot.py`: shared EOT transform helpers.
- `examples/backdoor_validate_physical.py`: attack-success-rate evaluation under
  randomly sampled physical transforms.

The physical-transform distribution used throughout is **rotation ±15°, scale
[0.7, 1.0], perspective distortion 0.5**.

## Reproducing the experiments

All commands use CIFAR-10 + `resnet18_comp`, target class 0, and a 6×6 trigger at
the top-right corner (offset `0, 26`); CIFAR-10 downloads automatically on first
run. Attacks fine-tune from a clean pretrained checkpoint (`--pretrained`); see
upstream for obtaining/training base models. Run everything from the repo root.

### Evaluating a trained model

The defence and physical-ASR commands take the **same attack/mark flags** used to
train the model, plus its `--attack_dir`. Set them once per attack, e.g. for
`imc_eot`:

```bash
ATTACK="--attack imc_eot --mark_random_init \
        --mark_height 6 --mark_width 6 --mark_height_offset 0 --mark_width_offset 26 \
        --mark_alpha 1.0 \
        --eot_rotation_max 15.0 --eot_scale_min 0.7 --eot_scale_max 1.0 \
        --eot_distortion_scale 0.5 --eot_n_samples 4"
DIR=./data/imc_eot
BASE="--pretrained --dataset cifar10 --model resnet18_comp"

# Physical ASR (5 reps over 1000 validation images)
python examples/backdoor_validate_physical.py $BASE $ATTACK --attack_dir $DIR \
    --physical_n_reps 5 --physical_n_eval 1000 \
    --physical_rotation_max 15 --physical_scale_min 0.7 \
    --physical_scale_max 1.0 --physical_distortion_scale 0.5

# Neural Cleanse
python examples/backdoor_defense.py $BASE $ATTACK --attack_dir $DIR --defense neural_cleanse

# STRIP
python examples/backdoor_defense.py $BASE $ATTACK --attack_dir $DIR --defense strip

# NC-EOT (transform-aware Neural Cleanse)
python examples/backdoor_defense.py $BASE $ATTACK --attack_dir $DIR \
    --defense neural_cleanse_eot \
    --nc_eot_rotation_max 15.0 --nc_eot_scale_min 0.7 --nc_eot_scale_max 1.0 \
    --nc_eot_distortion_scale 0.5 --nc_eot_n_samples 4 \
    --nc_eot_warmup_epochs 5 --nc_eot_robust_epochs 5
```

For the other attacks, change `$ATTACK` and `$DIR` to match their training
commands below (drop the `--eot_*` flags for `badnet`/`imc`, which have no
augmentation).

### 1. Robustness-detectability trade-off

A fixed-trigger pair (BadNet vs BadNet+Aug) and a co-optimised pair (IMC vs
IMC+EOT); each pair compares "no augmentation" against "with augmentation".

```bash
# BadNet: fixed trigger, no augmentation
python examples/backdoor_attack.py --pretrained --dataset cifar10 --model resnet18_comp \
    --validate_interval 1 --epochs 50 --lr 0.01 --mark_random_init \
    --mark_height 6 --mark_width 6 --mark_height_offset 0 --mark_width_offset 26 \
    --mark_alpha 1.0 --attack badnet --save --attack_dir ./data/badnet

# BadNet+Aug: fixed trigger + EOT augmentation
python examples/backdoor_attack.py --pretrained --dataset cifar10 --model resnet18_comp \
    --validate_interval 1 --epochs 50 --lr 0.01 --mark_random_init \
    --mark_height 6 --mark_width 6 --mark_height_offset 0 --mark_width_offset 26 \
    --mark_alpha 1.0 \
    --eot_rotation_max 15.0 --eot_scale_min 0.7 --eot_scale_max 1.0 \
    --eot_distortion_scale 0.5 \
    --attack badnet_aug --save --attack_dir ./data/badnet_aug

# IMC: co-optimised trigger, no augmentation
python examples/backdoor_attack.py --pretrained --dataset cifar10 --model resnet18_comp \
    --validate_interval 1 --epochs 50 --lr 0.01 --mark_random_init \
    --mark_height 6 --mark_width 6 --mark_height_offset 0 --mark_width_offset 26 \
    --mark_alpha 1.0 --attack imc --save --attack_dir ./data/imc

# IMC+EOT: co-optimised + EOT trigger optimisation + augmentation
python examples/backdoor_attack.py --pretrained --dataset cifar10 --model resnet18_comp \
    --validate_interval 1 --epochs 50 --lr 0.01 --mark_random_init \
    --mark_height 6 --mark_width 6 --mark_height_offset 0 --mark_width_offset 26 \
    --mark_alpha 1.0 \
    --eot_rotation_max 15.0 --eot_scale_min 0.7 --eot_scale_max 1.0 \
    --eot_distortion_scale 0.5 --eot_n_samples 4 \
    --attack imc_eot --save --attack_dir ./data/imc_eot
```

Then evaluate each model with the physical-ASR, Neural Cleanse, and STRIP commands
from [Evaluating a trained model](#evaluating-a-trained-model).

### 2. Defence-evasive attack (IMC+EOT+Adapt)

IMC+EOT with two adaptations: an enlarged **12×12** trigger (offset `0, 20`) to
defeat Neural Cleanse's smallest-mask anomaly, and a **half-opacity fine-tune**
(`--adapt_to_strip`) to defeat STRIP.

```bash
python examples/backdoor_attack.py --pretrained --dataset cifar10 --model resnet18_comp \
    --validate_interval 1 --epochs 50 --lr 0.01 --mark_random_init \
    --mark_height 12 --mark_width 12 --mark_height_offset 0 --mark_width_offset 20 \
    --mark_alpha 1.0 \
    --eot_rotation_max 15.0 --eot_scale_min 0.7 --eot_scale_max 1.0 \
    --eot_distortion_scale 0.5 --eot_n_samples 4 \
    --attack imc_eot_adapt --adapt_to_strip --adapt_strip_epochs 5 \
    --save --attack_dir ./data/imc_eot_adapt
```

Evaluate with NC and STRIP (it should evade both) and with NC-EOT (it should not),
using the 12×12 mark flags in `$ATTACK`.

### 3. Reduced-capability offline attack (IMC_Offline)

A data-only adversary that never touches the victim's training loop: it reuses a
**converged IMC+EOT trigger** (from section 1) and injects a frozen pool of
transformed triggered images. Train IMC+EOT first so its trigger is saved under
`./data/imc_eot/`, then:

```bash
python examples/backdoor_attack.py --pretrained --dataset cifar10 --model resnet18_comp \
    --validate_interval 1 --epochs 50 --lr 0.01 \
    --mark_path ./data/imc_eot/square_white_tar0_alpha1.00_mark\(6,6\).png \
    --mark_height 6 --mark_width 6 --mark_height_offset 0 --mark_width_offset 26 \
    --mark_alpha 1.0 \
    --eot_rotation_max 15.0 --eot_scale_min 0.7 --eot_scale_max 1.0 \
    --eot_distortion_scale 0.5 \
    --attack imc_offline --poison_percent 0.10 \
    --save --attack_dir ./data/imc_offline
```

Variants: pass `--mark_random_init` instead of `--mark_path` to model a weaker
adversary that uses an un-optimised trigger, and sweep `--poison_percent` to vary
the poison budget. Evaluate as in section 1 (it is detected by NC-EOT).

## License

GPL-3.0, inherited from upstream TrojanZoo.
