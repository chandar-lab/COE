from .q_learner import QLearner
from .noise_q_learner import NoiseQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .ucb_learner import UCBLearner
from .ucb_cond_fac_learner import UCBCondFacLearner
from .batch_q_learner import BatchQLearner
from .fast_batch_q_learner import FastBatchQLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["noise_q_learner"] = NoiseQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["ucb_learner"] = UCBLearner
REGISTRY["ucb_cond_fac_learner"] = UCBCondFacLearner
REGISTRY["batch_q_learner"] = BatchQLearner
REGISTRY["fast_q_learner"] = FastBatchQLearner