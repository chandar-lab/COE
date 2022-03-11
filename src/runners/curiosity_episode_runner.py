from tokenize import group
from envs import REGISTRY as env_REGISTRY
from modules.mixers import REGISTRY as mix_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
from .episode_runner import EpisodeRunner


class CuriosityEpisodeRunner(EpisodeRunner):
    def __init__(self, args, logger):
        super().__init__(args=args, logger=logger)
        self.int_reward_ind_beta = self.args.int_reward_ind_beta
        self.int_reward_cen_beta = self.args.int_reward_cen_beta
        self.int_reward_clip = self.args.int_reward_clip
        self.train_int_returns = []

    def setup(self, scheme, groups, preprocess, mac):
        super().setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
        self.n_agents = self.mac.n_agents
        self.n_actions = self.mac.n_actions

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        episode_int_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            intrinsic_reward = th.zeros(self.batch_size, self.n_agents, device=self.args.device)
            if (not test_mode and
                np.maximum(self.int_reward_cen_beta, self.int_reward_ind_beta)>0):
                counts_t = self.mac.counts_t
                act_counts_t = self.mac.act_counts_t
                if self.int_reward_ind_beta:
                    state_action_counts = th.gather(
                        act_counts_t, dim=2,
                        index=actions.view(self.batch_size, self.n_agents, 1),
                    ).squeeze(dim=-1)
                    intrinsic_reward += self.int_reward_ind_beta / th.sqrt(state_action_counts+0.1)
                if self.int_reward_cen_beta:
                    state_counts = (
                        counts_t[(th.arange(self.batch_size),) + actions.T.split(split_size=1, dim=0)].view(self.batch_size, 1)
                    )
                    intrinsic_reward += self.int_reward_cen_beta / th.sqrt(state_counts+0.1)
                intrinsic_reward = intrinsic_reward.clamp(min=0, max=self.int_reward_clip)
                episode_int_return += intrinsic_reward.mean().cpu()

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "intrinsic_reward": intrinsic_reward,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        # Update action selector after each episode if needed (ie BootstrapDQN)
        self.mac.update_action_selector()

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        self.train_int_returns.append(episode_int_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            cur_stats_copy = cur_stats.copy()
            self._log(cur_returns, cur_stats, log_prefix)
            self._log(self.train_int_returns, cur_stats_copy, f"{log_prefix}intrinsic_")
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch
