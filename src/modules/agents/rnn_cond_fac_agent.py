import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .rnn_agent import RNNAgent


class RNNCondFacAgent(RNNAgent):
    """
    Agent conditioned on previous agent's action.
    """
    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)
        self.forward, self.forward_indep = self.forward_conditional, self.forward

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        act_embed_dim = args.hidden_dim//4
        self.prev_ag_act_fc = nn.Linear((self.n_agents-1)*self.n_actions, act_embed_dim)
        self.q_correct_fc = nn.Linear(args.hidden_dim+act_embed_dim, args.n_actions)

    def forward_conditional(self, inputs, hidden_state, is_dep=False, prev_ag_act=None):
        q_ind, hid = self.forward_indep(inputs, hidden_state)
        if is_dep:
            act = F.relu(self.prev_ag_act_fc(prev_ag_act))
            q_corr = self.q_correct_fc(th.cat([hid, act], dim=1))
            q_dep = q_ind.detach() + q_corr
            return q_dep, hid
        return q_ind, hid
