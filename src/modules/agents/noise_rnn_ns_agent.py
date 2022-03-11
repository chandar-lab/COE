import torch.nn as nn
from modules.agents.noise_rnn_agent import NoiseRNNAgent
from modules.agents.rnn_ns_agent import RNNNSAgent
import torch as th

class NoiseRNNNSAgent(RNNNSAgent):
    def __init__(self, input_shape, args):
        super(NoiseRNNNSAgent, self).__init__(input_shape, args)
        self.agents = th.nn.ModuleList([NoiseRNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def forward(self, inputs, hidden_state, noise):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i], noise)
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h = self.agents[i](inputs[:, i], hidden_state[:, i], noise)
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)
