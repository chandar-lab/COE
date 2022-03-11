import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .rnn_agent import RNNAgent


class NoiseRNNAgent(RNNAgent):
    def __init__(self, input_shape, args):
        super(NoiseRNNAgent, self).__init__(input_shape, args)
        self.forward, self.forward_rnn_agent = self.forward_noise, self.forward

        self.noise_fc1 = nn.Linear(
            args.noise_dim + args.n_agents if args.obs_agent_id else args.noise_dim,
            args.noise_embedding_dim)
        self.noise_fc2 = nn.Linear(args.noise_embedding_dim, args.noise_embedding_dim)
        self.noise_fc3 = nn.Linear(args.noise_embedding_dim, args.n_actions)

        self.hyper = True
        self.hyper_noise_fc1 = nn.Linear(
            args.noise_dim + args.n_agents if args.obs_agent_id else args.noise_dim,
            args.hidden_dim * args.n_actions)


    def forward_noise(self, inputs, hidden_state, noise):
        q, h = self.forward_rnn_agent(inputs, hidden_state)

        if self.args.obs_agent_id:
            agent_ids = th.eye(self.args.n_agents, device=inputs.device).repeat(noise.shape[0], 1)
            noise_repeated = noise.repeat(1, self.args.n_agents).reshape(agent_ids.shape[0], -1)
            noise_input = th.cat([noise_repeated, agent_ids], dim=-1)
        else:
            noise_input = noise

        if self.hyper:
            W = self.hyper_noise_fc1(noise_input).reshape(-1, self.args.n_actions, self.args.hidden_dim)
            wq = th.bmm(W, h.unsqueeze(2))
        else:
            z = F.tanh(self.noise_fc1(noise_input))
            z = F.tanh(self.noise_fc2(z))
            wz = self.noise_fc3(z)
            wq = q * wz

        return wq, h
