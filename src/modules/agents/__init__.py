REGISTRY = {}

from .rnn_agent import RNNAgent
from .noise_rnn_agent import NoiseRNNAgent
from .noise_rnn_ns_agent import NoiseRNNNSAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_cond_fac_agent import RNNCondFacAgent
from .rnn_fast_agent import RNNFastAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["noise_rnn"] = NoiseRNNAgent
REGISTRY["noise_rnn_ns"] = NoiseRNNNSAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_cond_fac"] = RNNCondFacAgent
REGISTRY["rnn_fast"] = RNNFastAgent