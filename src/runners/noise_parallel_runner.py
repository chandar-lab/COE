from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th

from modules.bandits import REGISTRY as bandit_REGISTRY
from .parallel_runner import ParallelRunner


class NoiseParallelRunner(ParallelRunner):

    def cuda(self):
        if self.args.noise_bandit:
            self.noise_distrib.cuda()

    def setup(self, scheme, groups, preprocess, mac):
        super(NoiseParallelRunner, self).setup(scheme, groups, preprocess, mac)

        # Setup the noise distribution sampler
        if self.args.noise_bandit:
            if self.args.bandit_policy:
                noise_dist = "ez_agent"
            else:
                noise_dist = "returns_bandit"
        else:
           noise_dist = "uniform"

        self.noise_distrib = bandit_REGISTRY[noise_dist](self.args, logger=self.logger)
        self.noise_returns = {}
        self.noise_test_won = {}
        self.noise_train_won = {}

    def reset(self, test_mode=False):
        super(NoiseParallelRunner, self).reset()

        # Sample the noise at the beginning of the episode
        self.noise = self.noise_distrib.sample(self.batch['state'][:,0], test_mode)
        self.batch.update({"noise": self.noise}, ts=0)

        # SMAC envs specific
        if getattr(self.args.env_args, "map_name", "") == "2_corridors":
            if self.t_env > 5 * 1000 * 1000:
                for parent_conn in self.parent_conns:
                    parent_conn.send(("close_corridor", None))
        if getattr(self.args.env_args, "map_name", "") == "bunker_vs_6m":
            if self.t_env > 3 * 1000 * 1000:
                for parent_conn in self.parent_conns:
                    parent_conn.send(("avail_bunker", None))

    def run(self, test_mode=False, test_uniform=False):
        self.reset(test_uniform)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        if test_uniform:
            log_prefix += "uni_"
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        self._update_noise_returns(episode_returns, self.noise, final_env_infos, test_mode)
        self.noise_distrib.update_returns(self.batch['state'][:,0], self.noise, episode_returns, test_mode, self.t_env)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log_noise_returns(test_mode, test_uniform)
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log_noise_returns(test_mode, test_uniform)
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _update_noise_returns(self, returns, noise, stats, test_mode):
        for n, r in zip(noise, returns):
            n = int(np.argmax(n))
            if n in self.noise_returns:
                self.noise_returns[n].append(r)
            else:
                self.noise_returns[n] = [r]

        if stats != [] and "battle_won" in stats[0]:
            noise_won = self.noise_test_won if test_mode else self.noise_train_won
            for n, info in zip(noise, stats):
                if "battle_won" not in info:
                    continue
                bw = info["battle_won"]
                n = int(np.argmax(n))
                if n in noise_won:
                    noise_won[n].append(bw)
                else:
                    noise_won[n] = [bw]

    def _log_noise_returns(self, test_mode, test_uniform):
        if test_mode:
            max_noise_return = -100000
            for n,rs in self.noise_returns.items():
                n_item = n
                r_mean = float(np.mean(rs))
                max_noise_return = max(r_mean, max_noise_return)
                self.logger.log_stat("{}_noise_test_ret_u_{:1}".format(n_item, test_uniform), r_mean, self.t_env)
            self.logger.log_stat("max_noise_test_ret_u_{:1}".format(test_uniform), max_noise_return, self.t_env)
        noise_won = self.noise_test_won
        prefix = "test"
        if test_uniform:
            prefix += "_uni"
        if not test_mode:
            noise_won = self.noise_train_won
            prefix = "train"
        if len(noise_won.keys()) > 0:
            max_test_won = 0
            for n, rs in noise_won.items():
                n_item = n #int(np.argmax(n))
                r_mean = float(np.mean(rs))
                max_test_won = max(r_mean, max_test_won)
                self.logger.log_stat("{}_noise_{}_won".format(n_item, prefix), r_mean, self.t_env)
            self.logger.log_stat("max_noise_{}_won".format(prefix), max_test_won, self.t_env)
        self.noise_returns = {}
        self.noise_test_won = {}
        self.noise_train_won = {}

    def save_models(self, path):
        if self.args.noise_bandit:
            self.noise_distrib.save_model(path)

def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "close_corridor":
            env.close_corridor()
        elif cmd == "avail_bunker":
            env.open_bunker()
        else:
            raise NotImplementedError
