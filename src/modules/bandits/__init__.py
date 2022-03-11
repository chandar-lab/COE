from .const_lr import Constant_Lr
from .reinforce_hierarchial import EZ_agent as enza
from .returns_bandit import ReturnsBandit as RBandit
from .uniform import Uniform

REGISTRY = {}
REGISTRY["constant_lr"] = Constant_Lr
REGISTRY["ez_agent"] = enza
REGISTRY["returns_bandit"] = RBandit
REGISTRY["uniform"] = Uniform