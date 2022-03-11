import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from controllers import REGISTRY as mac_REGISTRY
from learners.q_learner import QLearner


class VDNQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.logger = logger

        self.mac = copy.deepcopy(mac)
        self.target_mac = copy.deepcopy(mac)
        self.predict_mac = copy.deepcopy(mac)

        self.mixer = VDNMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.params = list(self.mac.parameters())
        self.params += list(self.mixer.parameters())
        self.predict_params = list(self.predict_mac.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.predict_optimiser = Adam(params=self.predict_params, lr=args.lr)

        self.last_target_update_episode = 0
        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.decay_stats_t = 0
        self.decay_stats_t_2 = 0


    def subtrain(self, batch: EpisodeBatch, t_env: int, episode_num: int,
            mac, save_buffer=False, imac=None, timac=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Calculate estimated Q-Values
        # Calculate the Q-Values necessary for the target
        mac.init_hidden(batch.batch_size)
        self.predict_mac.init_hidden(batch.batch_size)
        self.target_mac.init_hidden(batch.batch_size)

        mac_out = mac.forward(batch, batch.max_seq_length, batch_inf=True)
        predict_mac_out = self.predict_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]
        target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        target_mac_out_next = target_mac_out.clone().detach()
        target_mac_out_next = target_mac_out_next.contiguous().view(-1, self.args.n_actions) * 10
        predict_mac_out = predict_mac_out.contiguous().view(-1, self.args.n_actions)

        prediction_error = F.pairwise_distance(predict_mac_out, target_mac_out_next, p=2.0, keepdim=True)
        prediction_mask = mask.repeat(1, 1, self.args.n_agents)
        prediction_error = prediction_error.reshape(batch.batch_size, -1, self.args.n_agents) * prediction_mask

        if getattr(self.args, "mask_other_agents", False):
            intrinsic_rewards = self.args.curiosity_scale * (prediction_error.detach()[:, :, 0:1])
        else:
            intrinsic_rewards = self.args.curiosity_scale * (prediction_error.mean(dim=-1, keepdim=True).detach())

        prediction_loss = prediction_error.sum() / prediction_mask.sum()

        ############################
        if save_buffer:
            return intrinsic_rewards

        self.predict_optimiser.zero_grad()
        prediction_loss.backward()
        predict_grad_norm = th.nn.utils.clip_grad_norm_(self.predict_params, self.args.grad_norm_clip)
        self.predict_optimiser.step()
        ############################

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()
        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

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
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        if getattr(self.args, "use_qtotal_td", False):
            intrinsic_rewards = self.args.curiosity_scale * th.abs(masked_td_error.clone().detach())

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if self.args.curiosity_decay:
            if t_env - self.decay_stats_t >= self.args.curiosity_decay_cycle:
                if self.args.curiosity_decay_rate <= 1.0:
                    if self.args.curiosity_scale > self.args.curiosity_decay_stop:
                        self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                    else:
                        self.args.curiosity_scale = self.args.curiosity_decay_stop
                else:
                    if self.args.curiosity_scale < self.args.curiosity_decay_stop:
                        self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                    else:
                        self.args.curiosity_scale = self.args.curiosity_decay_stop

                self.decay_stats_t=t_env

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("vdn_loss", loss.item(), t_env)
            self.logger.log_stat("vdn_hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("vdn_grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("vdn_td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("vdn_q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("vdn_target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self.logger.log_stat("vdn_prediction_loss", prediction_loss.item(), t_env)
            self.logger.log_stat("vdn_intrinsic_rewards", intrinsic_rewards.sum().item() / mask_elems, t_env)
            self.logger.log_stat("vdn_extrinsic_rewards", rewards.sum().item() / mask_elems, t_env)

            self.log_stats_t = t_env

        return intrinsic_rewards


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int,
        save_buffer=False, imac=None, timac=None):

        intrinsic_rewards = self.subtrain(
            batch, t_env, episode_num, self.mac, save_buffer=save_buffer, imac=imac, timac=timac)

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        return intrinsic_rewards


    def cuda(self):
        super().cuda()
        self.predict_mac.cuda()

    def save_models(self, path):
        super().save_models(path)
        self.predict_mac.save_models(f"{path}/predict_mac")
        th.save(self.predict_optimiser.state_dict(), f"{path}/predict_opt.th")

    def load_models(self, path):
        super().load_models(path)
        self.predict_mac.load_models(f"{path}/predict_mac")
        self.predict_optimiser.load_state_dict(th.load(f"{path}/predict_opt.th", map_location=lambda storage, loc: storage))