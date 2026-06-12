#!/usr/bin/env python3

from ...abstract import BackdoorAttack

from .badnet import BadNet
from .badnet_aug import BadNet_Aug
from .trojannn import TrojanNN
from .latent_backdoor import LatentBackdoor
from .imc import IMC
from .imc_eot import IMC_EOT
from .imc_eot_adapt import IMC_EOT_Adapt
from .imc_offline import IMC_Offline
from .trojannet import TrojanNet

__all__ = ['BadNet', 'BadNet_Aug', 'TrojanNN', 'LatentBackdoor', 'IMC', 'IMC_EOT', 'IMC_EOT_Adapt', 'IMC_Offline', 'TrojanNet']

class_dict: dict[str, type[BackdoorAttack]] = {
    'badnet': BadNet,
    'badnet_aug': BadNet_Aug,
    'trojannn': TrojanNN,
    'latent_backdoor': LatentBackdoor,
    'imc': IMC,
    'imc_eot': IMC_EOT,
    'imc_eot_adapt': IMC_EOT_Adapt,
    'imc_offline': IMC_Offline,
    'trojannet': TrojanNet,
}
