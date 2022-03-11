from typing import Any
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.counters import REGISTRY as cnt_REGISTRY
import torch as th
import torch.nn.functional as F
import numpy as np
from functools import partial
from itertools import product
from controllers.base_controller import BaseMAC
from controllers.basic_controller import BasicMAC


class UCBMAC(BaseMAC):
    """

    State representation and dimension. See src/envs/__init__.py
    gymma: state is concatenation of obs
    sc2: state is either concatenation of obs, or global state

    episode_limit is either set by command env_args or pre-defined (as sc2).
    """
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.is_smac = self.args.env == "sc2"
        self.ucb_act_cp = args.ucb_act_cp
        self.int_reward_ind_beta = self.args.int_reward_ind_beta
        self.int_reward_cen_beta = self.args.int_reward_cen_beta
        self.device = self.args.device

        self.ucb_indices = None
        self.act_t = None
        self.abs_state_ep_t = None

        calculate_ucb_fn = getattr(args, "calculate_ucb_fn", "conditional")
        if calculate_ucb_fn == "conditional":
            self.calculate_ucb = self.calculate_ucb_conditional
        elif calculate_ucb_fn == "independent":
            self.calculate_ucb = self.calculate_ucb_independent
            self.args.ucb_conf_decay = False

        self.confidence_fn = confidence_registry[args.confidence_fn](args=args)

        # Initialize simhash counters
        # By default smac does not use combination of all agents' observations as the global state
        if not self.is_smac:
            assert self.state_shape == scheme["state"]["vshape"] == scheme["obs"]["vshape"]*self.n_agents
        self.hash_counter = partial(
            cnt_REGISTRY[getattr(args, "counter", "simhash")],
            args=args)
        self.counters = []
        self.counters.append(self.hash_counter())
        self.train_t_env = th.tensor(0., dtype=th.float32, device=self.device,)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(
            ep_batch=ep_batch, t=t_ep, cp=self.ucb_act_cp, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        # Update counts
        if not test_mode:
            self.counters[0].update_count_by_key(keys=self.abs_state_ep_t, acts=chosen_actions)
            self.train_t_env += ep_batch.batch_size*self.args.decay_factor

        return chosen_actions

    def forward(self, ep_batch, t, cp, test_mode=False, learn_mode=False):
        """
        During training phase, the agent is either acting or learning.
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # Q estimates
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

        # act_mode, or learn_mode with cp>0
        if (not test_mode and not learn_mode) or (cp>0 and learn_mode):
            bs = ep_batch.batch_size
            states = ep_batch["state"][:, t]
            self.abs_state_ep_t = self.counters[0].compute_keys(obss=states)

        if (test_mode or
            (cp==0 and learn_mode) or
            (cp==0 and not learn_mode and
                np.maximum(self.int_reward_cen_beta, self.int_reward_ind_beta)==0)
        ):
            if self.agent_output_type == "pi_logits":
                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[avail_actions == 0] = -1e10
                agent_outs = F.softmax(agent_outs, dim=-1)

            return agent_outs

        self.counts_t = self.counters[0].get_count_by_key(keys=self.abs_state_ep_t)

        self.ucb_indices = th.zeros((bs, self.n_agents, self.n_actions), device=self.device)
        self.act_t = th.zeros((bs, self.n_agents), dtype=th.int64, device=self.device)
        self.act_counts_t = th.zeros((bs, self.n_agents, self.n_actions), device=self.device)

        self.calculate_ucb(agent_outs=agent_outs, t=t, cp=cp)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                self.ucb_indices[avail_actions == 0] = -1e10
            self.ucb_indices = F.softmax(self.ucb_indices, dim=-1)

        return self.ucb_indices

    def calculate_ucb_conditional(self, agent_outs, t, cp):
        bs = agent_outs.shape[0]

        counts = self.counts_t.clone()
        parent_counts = counts.sum(dim=tuple(range(1, self.n_agents+1))).unsqueeze(dim=-1)
        for ag_i in range(self.n_agents):
            if ag_i == self.n_agents-1:
                child_counts = counts
            else:
                child_counts = counts.sum(dim=tuple(range(2, len(counts.shape))))
            self.act_counts_t[:, ag_i] = child_counts
            self.ucb_indices[:, ag_i] = (
                agent_outs[:, ag_i] + cp * self.confidence_fn(
                    parent_counts=parent_counts, child_counts=child_counts, t_ep=t,
                ))
            max_idx = self.ucb_indices[:, ag_i].argmax(dim=1)
            self.act_t[:, ag_i] = max_idx
            if ag_i != self.n_agents-1:
                counts = counts[th.arange(bs), max_idx]
                parent_counts = th.gather(child_counts, dim=1, index=max_idx.view(bs, 1))

    def calculate_ucb_independent(self, agent_outs, t, cp):
        counts = self.counts_t.clone()
        for ag_i in range(self.n_agents):
            other_agent_dims = tuple(
                dim for dim in range(1, self.n_agents+1) if dim != ag_i+1)
            act_counts = counts.sum(dim=other_agent_dims)
            self.act_counts_t[:, ag_i] = act_counts
            self.ucb_indices[:, ag_i] = (
                agent_outs[:, ag_i] + cp * self.confidence_fn(
                    parent_counts=self.train_t_env, child_counts=act_counts, t_ep=t,
                ))
            max_idx = self.ucb_indices[:, ag_i].argmax(dim=1)
            self.act_t[:, ag_i] = max_idx


    def cuda(self):
        super().cuda()
        for counter in self.counters:
            counter.cuda()

    def save_models(self, path):
        super().save_models(path=path)
        for counter in self.counters:
            counter.save_models(path=path)
        th.save(self.train_t_env, f"{path}/train_t_env.pt")

    def load_models(self, path):
        super().load_models(path=path)
        for counter in self.counters:
            counter.load_models(path=path)
        self.train_t_env = th.load(f"{path}/train_t_env.pt", map_location=self.device)


class UCB1Confidence:
    def __init__(self, args) -> None:
        """
        Note: If optim_init, the action value is large, eg 1/(1-gamma) if r \in
        [0,1], or episode horizon if given.
        """
        self.args = args
        self.episode_limit = args.episode_limit
        self.optim_init = getattr(args, "ucb_optim_init", False)
        self.conf_decay = getattr(args, "ucb_conf_decay", True)
        self.expo = 0.5

    def ft(self, count):
        return count

    def __call__(self, parent_counts, child_counts, t_ep, *args: Any, **kwds: Any):
        ft_parent = self.ft(parent_counts+1.1)
        if self.conf_decay:
            self.expo = (self.episode_limit+t_ep)/(2*self.episode_limit+t_ep)
        confidence = (2*th.log(ft_parent) / (child_counts+0.1))**self.expo
        if self.optim_init:
            confidence[child_counts == 0] = self.episode_limit
        return confidence

class UCBAsymptoticConfidence(UCB1Confidence):
    def ft(self, count):
        return 1 + count * th.log(count)**2

class UCBModifiedConfidence(UCB1Confidence):
    def ft(self, count):
        return th.sqrt(count)

    def __call__(self, parent_counts, child_counts, *args: Any, **kwds: Any):
        ft_parent = self.ft(parent_counts+0.1)
        confidence = th.sqrt(ft_parent / (child_counts+0.1))
        if self.optim_init:
            confidence[child_counts == 0] = self.episode_limit
        return confidence

class ModifiedUCTConfidence(UCB1Confidence):
    def __init__(self, args) -> None:
        super().__init__(args=args)
        self.N = 2**(self.episode_limit+1) -1
        self.uct_beta = getattr(args, "uct_beta", 0.1)
        self.beta_inv_coeff = 2*self.N / self.uct_beta

    def __call__(self, parent_counts, child_counts, t_ep, *args: Any, **kwds: Any):
        beta_inv = self.beta_inv_coeff*child_counts*(child_counts+1)
        k_d = (1+np.sqrt(2))/np.sqrt(2) * ((1+np.sqrt(2))**(self.episode_limit - t_ep) -1)
        k_d_prime = (3**(self.episode_limit - t_ep) -1)/2
        confidence = (k_d+1)* th.sqrt(2*th.log(beta_inv) / (child_counts+0.1)) + k_d_prime/(child_counts+0.1)
        if self.optim_init:
            confidence[child_counts == 0] = self.episode_limit
        return confidence


confidence_registry = {
    "ucb_asymptotic": UCBAsymptoticConfidence,
    "ucb1": UCB1Confidence,
    "ucb_modified": UCBModifiedConfidence,
    "modified_uct": ModifiedUCTConfidence,
}