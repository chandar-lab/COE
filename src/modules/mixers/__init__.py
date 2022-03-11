from .qmix import QMixer
from .noise_qmix import NoiseQMixer
from .qtran import QTranBase
from .vdn import VDNMixer

REGISTRY = {}
REGISTRY["qmix"] = QMixer
REGISTRY["noise_qmix"] = NoiseQMixer
REGISTRY["qtran"] = QTranBase
REGISTRY["vdn"] = VDNMixer
