import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers import REGISTRY as mix_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from learners.q_learner import QLearner


class NoiseQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super(NoiseQLearner, self).__init__(mac, scheme, logger, args)

        discrim_input = np.prod(self.args.state_shape) + self.args.n_agents * self.args.n_actions
        if self.args.rnn_discrim:
            self.rnn_agg = RNNAggregator(discrim_input, args)
            self.discrim = Discrim(args.rnn_agg_size, self.args.noise_dim, args)
            self.params += list(self.discrim.parameters())
            self.params += list(self.rnn_agg.parameters())
        else:
            self.discrim = Discrim(discrim_input, self.args.noise_dim, args)
            self.params += list(self.discrim.parameters())
        self.discrim_loss = nn.CrossEntropyLoss(reduction="none")

        # Reconstruct the optimizer
        # del self.optimiser
        self.optimiser = Adam(params=self.params, lr=args.lr)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        noise = batch["noise"][:, 0].unsqueeze(1).repeat(1,rewards.shape[1],1)

        if self.args.standardise_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], noise=noise)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], noise=noise)

        # Discriminator
        mac_out_zeros = th.zeros_like(mac_out)
        mac_out_zeros[avail_actions == 0] = -9999999
        mac_out = mac_out + mac_out_zeros
        q_softmax_actions = F.softmax(mac_out[:, :-1], dim=3)

        if self.args.hard_qs:
            maxs = th.max(mac_out[:, :-1], dim=3, keepdim=True)[1]
            zeros = th.zeros_like(q_softmax_actions)
            zeros.scatter_(dim=3, index=maxs, value=1)
            q_softmax_actions = zeros

        q_softmax_agents = q_softmax_actions.reshape(q_softmax_actions.shape[0], q_softmax_actions.shape[1], -1)

        states = batch["state"][:, :-1]
        state_and_softactions = th.cat([q_softmax_agents, states], dim=2)

        if self.args.rnn_discrim:
            h_to_use = th.zeros(size=(batch.batch_size, self.args.rnn_agg_size)).to(states.device)
            hs = th.ones_like(h_to_use)
            for t in range(batch.max_seq_length - 1):
                hs = self.rnn_agg(state_and_softactions[:, t], hs)
                for b in range(batch.batch_size):
                    if t == batch.max_seq_length - 2 or (mask[b, t] == 1 and mask[b, t+1] == 0):
                        # This is the last timestep of the sequence
                        h_to_use[b] = hs[b]
            s_and_softa_reshaped = h_to_use
        else:
            s_and_softa_reshaped = state_and_softactions.reshape(-1, state_and_softactions.shape[-1])

        if self.args.mi_intrinsic:
            s_and_softa_reshaped = s_and_softa_reshaped.detach()

        discrim_prediction = self.discrim(s_and_softa_reshaped)

        # Cross-Entropy
        target_repeats = 1
        if not self.args.rnn_discrim:
            target_repeats = q_softmax_actions.shape[1]
        discrim_target = batch["noise"][:, 0].long().detach().max(dim=1)[1].unsqueeze(1).repeat(1, target_repeats).reshape(-1)
        discrim_loss = self.discrim_loss(discrim_prediction, discrim_target)

        if self.args.rnn_discrim:
            averaged_discrim_loss = discrim_loss.mean()
        else:
            masked_discrim_loss = discrim_loss * mask.reshape(-1)
            averaged_discrim_loss = masked_discrim_loss.sum() / mask.sum()
        self.logger.log_stat("discrim_loss", averaged_discrim_loss.item(), t_env)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        if self.args.mi_intrinsic:
            assert self.args.rnn_discrim is False
            targets = targets + self.args.mi_scaler * discrim_loss.view_as(rewards)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # MI loss
        loss = loss + self.args.mi_loss * averaged_discrim_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        super(NoiseQLearner, self).cuda()
        self.discrim.cuda()
        if self.args.rnn_discrim:
            self.rnn_agg.cuda()


class Discrim(nn.Module):

    def __init__(self, input_size, output_size, args):
        super().__init__()
        self.args = args
        layers = [nn.Linear(input_size, self.args.discrim_size), nn.ReLU()]
        for _ in range(self.args.discrim_layers - 1):
            layers.append(nn.Linear(self.args.discrim_size, self.args.discrim_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.args.discrim_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RNNAggregator(nn.Module):

    def __init__(self, input_size, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        output_size = args.rnn_agg_size
        self.rnn = nn.GRUCell(input_size, output_size)

    def forward(self, x, h):
        return self.rnn(x, h)
