import copy
import os
from components.episode_buffer import EpisodeBatch
from modules.mixers import REGISTRY as mix_REGISTRY
import numpy as np
import torch as th
from torch.optim import Adam
from learners.q_learner import QLearner
from learners.batch_q_learner import BatchQLearner
from learners.vdn_q_learner import VDNQLearner


class FastBatchQLearner(BatchQLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.save_buffer_cnt = 0
        if getattr(self.args, "save_buffer", False):
            self.args.save_buffer_path = os.path.join(self.args.save_buffer_path, str(self.args.seed))

        ###curiosity new
        self.vdn_learner=VDNQLearner(mac, scheme, logger, args)
        self.decay_stats_t = 0
        self.state_shape = scheme["state"]["vshape"]
        self.save_buffer_cnt = 0
        self.n_actions = self.args.n_actions

    def subtrain(
        self, batch: EpisodeBatch, t_env: int, episode_num: int,
        mac, intrinsic_rewards, ec_buffer=None, save_buffer=False,
    ):
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
        mac.init_hidden(batch.batch_size)
        mac_out = mac.forward(batch, batch.max_seq_length, batch_inf=True)

        if save_buffer:
            curiosity_r=intrinsic_rewards.clone().detach().cpu().numpy()
            # rnd_r = rnd_intrinsic_rewards.clone().detach().cpu().numpy()
            # extrinsic_mac_out_save=extrinsic_mac_out.clone().detach().cpu().numpy()
            mac_out_save = mac_out.clone().detach().cpu().numpy()
            actions_save=actions.clone().detach().cpu().numpy()
            terminated_save=terminated.clone().detach().cpu().numpy()
            state_save=batch["state"][:, :-1].clone().detach().cpu().numpy()
            data_dic={'curiosity_r':curiosity_r,
                                 # 'extrinsic_Q':extrinsic_mac_out_save,
                        'control_Q':mac_out_save,'actions':actions_save,'terminated':terminated_save,
                        'state':state_save}
            self.save_buffer_cnt += self.args.save_buffer_cycle
            if not os.path.exists(self.args.save_buffer_path):
                os.makedirs(self.args.save_buffer_path)
            np.save(self.args.save_buffer_path +"/"+ 'data_{}'.format(self.save_buffer_cnt), data_dic)
            print('save buffer ({}) at time{}'.format(batch.batch_size, self.save_buffer_cnt))
            return

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()
        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Calculate the Q-Values necessary for the target
        # We don't need the first timesteps Q-Value estimate for calculating targets
        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

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
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        if self.args.use_emdqn:
            ec_buffer.update_counter += 1
            qec_input = chosen_action_qvals.clone().detach()
            qec_input_new=[]
            for i in range(self.args.batch_size):
                qec_tmp=qec_input[i,:]
                for j in range(1,batch.max_seq_length):
                    if not mask[i, j-1]:
                        continue
                    z = np.dot(ec_buffer.random_projection, batch["state"][i][j].cpu())
                    q = ec_buffer.peek(z, None, modify=False)
                    if q != None:
                        qec_tmp[j-1] = self.args.gamma * q + rewards[i][j-1]
                        ec_buffer.qecwatch.append(q)
                        ec_buffer.qec_found += 1
                qec_input_new.append(qec_tmp)
            qec_input_new=th.stack(qec_input_new,dim=0)

            #print("qec_mean:", np.mean(ec_buffer.qecwatch))
            episodic_q_hit_pro=1.0 * ec_buffer.qec_found / self.args.batch_size /ec_buffer.update_counter/batch.max_seq_length
            #print("qec_fount: %.2f" % episodic_q_hit_pro)
        targets = rewards + intrinsic_rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.use_emdqn:
            emdqn_td_error = qec_input_new.detach() - chosen_action_qvals
            emdqn_masked_td_error = emdqn_td_error * mask
            emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
            loss += emdqn_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            if self.args.use_emdqn:
                self.logger.log_stat("em_Q_mean",  (qec_input_new * mask).sum().item() /
                                     (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("em_Q_hit_probability", episodic_q_hit_pro, t_env)
                self.logger.log_stat("emdqn_loss", emdqn_loss.item(), t_env)
                self.logger.log_stat("emdqn_curr_capacity", ec_buffer.ec_buffer.curr_capacity, t_env)
            self.logger.log_stat("extrinsic_rewards", rewards.sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env

        if self.args.is_prioritized_buffer:
            return masked_td_error ** 2, mask


    def train(
        self, batch: EpisodeBatch, t_env: int, episode_num: int,
        show_demo=False, save_data=None, show_v=False, ec_buffer=None,
    ):
        intrinsic_rewards = self.vdn_learner.train(
            batch, t_env, episode_num,
            save_buffer=False, imac=self.mac, timac=self.target_mac)
        if self.args.is_prioritized_buffer:
            masked_td_error, mask = self.subtrain(
                batch, t_env, episode_num, self.mac,
                intrinsic_rewards=intrinsic_rewards, ec_buffer=ec_buffer)
        else:
            self.subtrain(
                batch, t_env, episode_num, self.mac,
                intrinsic_rewards=intrinsic_rewards, ec_buffer=ec_buffer)

        if getattr(self.args, "save_buffer", False):
            if self.buffer.episodes_in_buffer - self.save_buffer_cnt >= self.args.save_buffer_cycle:
                if self.buffer.can_sample(self.args.save_buffer_cycle):
                    batch_tmp=self.buffer.sample(self.args.save_buffer_cycle, newest=True)
                    intrinsic_rewards_tmp, _ = self.vdn_learner.train(
                        batch_tmp, t_env, episode_num, save_buffer=True,
                        imac=self.mac, timac=self.target_mac)
                    self.subtrain(
                        batch_tmp, t_env, episode_num, self.mac,
                        intrinsic_rewards=intrinsic_rewards_tmp, save_buffer=True)
                else:
                    print('**' * 20, self.buffer.episodes_in_buffer, self.save_buffer_cnt)

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if self.args.use_emdqn and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            ec_buffer.update_kdtree()

        if self.args.is_prioritized_buffer:
            res = th.sum(masked_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res

    def cuda(self):
        super().cuda()
        self.vdn_learner.cuda()