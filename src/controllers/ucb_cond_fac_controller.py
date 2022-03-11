import torch as th
import torch.nn.functional as F
import numpy as np
from controllers.ucb_controller import UCBMAC


class UCBCondFacMAC(UCBMAC):
    def __init__(self, scheme, groups, args):
        args.learner = "ucb_cond_fac_learner"
        args.agent = "rnn_cond_fac"
        super().__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(
            ep_batch=ep_batch, t=t_ep, cp=self.ucb_act_cp, test_mode=test_mode,
            is_dep=(not test_mode))
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        # Update counts
        if not test_mode:
            self.counters[0].update_count_by_key(keys=self.abs_state_ep_t, acts=chosen_actions)
            self.train_t_env += ep_batch.batch_size*self.args.decay_factor

        return chosen_actions


    def forward(self, ep_batch, t, cp, test_mode=False, learn_mode=False, is_dep=False):
        # Independent
        if test_mode or (learn_mode and not is_dep):
            if isinstance(self.hidden_states, list):
                self.hidden_states = th.stack(self.hidden_states, dim=1)
            return super().forward(
                ep_batch=ep_batch, t=t, cp=cp,
                test_mode=test_mode, learn_mode=learn_mode)

        # Dependent
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # act_mode, or learn_mode with cp>0
        if (not learn_mode) or (cp>0 and learn_mode):
            states = ep_batch["state"][:, t]
            self.abs_state_ep_t = self.counters[0].compute_keys(obss=states)

        agent_inputs = agent_inputs.view(ep_batch.batch_size, self.n_agents, -1)
        prev_ag_act = th.zeros_like(ep_batch["actions_onehot"][:, t, 1:]).view(ep_batch.batch_size, -1)

        if ((cp==0 and learn_mode) or
            (cp==0 and not learn_mode and
                np.maximum(self.int_reward_cen_beta, self.int_reward_ind_beta)==0)
        ):
            self.calculate_q_conditional_dependent(
                agent_inputs=agent_inputs, prev_ag_act=prev_ag_act, t_ep=t, cp=cp)
            return self.agent_outs

        self.counts_t = self.counters[0].get_count_by_key(keys=self.abs_state_ep_t)

        self.calculate_ucb_conditional_dependent(
            agent_inputs=agent_inputs, prev_ag_act=prev_ag_act, t_ep=t, cp=cp)
        return self.ucb_indices


    def calculate_ucb_conditional_dependent(
        self, agent_inputs, prev_ag_act, t_ep, cp,
    ):
        bs = agent_inputs.shape[0]
        self.agent_outs = th.zeros((bs, self.n_agents, self.n_actions), device=self.device)
        self.ucb_indices = th.zeros((bs, self.n_agents, self.n_actions), device=self.device)
        self.act_t = th.zeros((bs, self.n_agents), dtype=th.int64, device=self.device)
        self.act_counts_t = th.zeros((bs, self.n_agents, self.n_actions), device=self.device)

        counts = self.counts_t.clone()
        parent_counts = counts.sum(dim=tuple(range(1, self.n_agents+1))).unsqueeze(dim=-1)

        for ag_i in range(self.n_agents):
            self.agent_outs[:, ag_i], self.hidden_states[ag_i] = self.agent(
                inputs=agent_inputs[:, ag_i],
                hidden_state=self.hidden_states[ag_i],
                is_dep=True,
                prev_ag_act=prev_ag_act,
            )

            if ag_i == self.n_agents-1:
                child_counts = counts
            else:
                child_counts = counts.sum(dim=tuple(range(2, len(counts.shape))))
            self.act_counts_t[:, ag_i] = child_counts
            self.ucb_indices[:, ag_i] = (
                self.agent_outs[:, ag_i] + cp * self.confidence_fn(
                    parent_counts=parent_counts, child_counts=child_counts, t_ep=t_ep,
                ))
            max_idx = self.ucb_indices[:, ag_i].argmax(dim=1)
            self.act_t[:, ag_i] = max_idx
            if ag_i != self.n_agents-1:
                prev_ag_act = prev_ag_act.clone()
                prev_ag_act_i = F.one_hot(max_idx, num_classes=self.n_actions).view(bs, self.n_actions).to(dtype=th.float32)
                prev_ag_act[:, ag_i*self.n_actions:ag_i*self.n_actions+self.n_actions] = prev_ag_act_i

                counts = counts[th.arange(bs), max_idx]
                parent_counts = th.gather(child_counts, dim=1, index=max_idx.view(bs, 1))


    def calculate_q_conditional_dependent(
        self, agent_inputs, prev_ag_act, t_ep, cp,
    ):
        bs = agent_inputs.shape[0]
        self.agent_outs = th.zeros((bs, self.n_agents, self.n_actions), device=self.device)
        self.act_t = th.zeros((bs, self.n_agents), dtype=th.int64, device=self.device)
        self.act_counts_t = th.zeros((bs, self.n_agents, self.n_actions), device=self.device)

        for ag_i in range(self.n_agents):
            self.agent_outs[:, ag_i], self.hidden_states[ag_i] = self.agent(
                inputs=agent_inputs[:, ag_i],
                hidden_state=self.hidden_states[ag_i],
                is_dep=True,
                prev_ag_act=prev_ag_act,
            )
            max_idx = self.agent_outs[:, ag_i].argmax(dim=1)
            self.act_t[:, ag_i] = max_idx
            if ag_i != self.n_agents-1:
                prev_ag_act = prev_ag_act.clone()
                prev_ag_act_i = F.one_hot(max_idx, num_classes=self.n_actions).view(bs, self.n_actions).to(dtype=th.float32)
                prev_ag_act[:, ag_i*self.n_actions:ag_i*self.n_actions+self.n_actions] = prev_ag_act_i

    def init_hidden(self, batch_size):
        self.hidden_states = [
            self.agent.init_hidden().repeat(batch_size, 1)
            for _ in range(self.n_agents)]