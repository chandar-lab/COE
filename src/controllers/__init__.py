REGISTRY = {}

from .basic_controller import BasicMAC
from .noise_controller import NoiseMAC
from .non_shared_noise_controller import NonSharedNoiseMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .ucb_controller import UCBMAC
from .ucb_cond_fac_controller import UCBCondFacMAC
from .fast_controller import FastMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["noise_mac"] = NoiseMAC
REGISTRY["non_shared_noise_mac"] = NonSharedNoiseMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["ucb_mac"] = UCBMAC
REGISTRY["ucb_cond_fac_mac"] = UCBCondFacMAC
REGISTRY["fast_mac"] = FastMAC