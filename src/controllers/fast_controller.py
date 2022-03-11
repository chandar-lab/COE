import torch as th
from controllers.basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class FastMAC(BasicMAC):
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if getattr(self.args, "use_individual_Q", False):
            agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, batch_inf=False):
        agent_inputs = self._build_inputs(ep_batch, t, batch_inf)
        epi_len = t if batch_inf else 1
        avail_actions = (ep_batch["avail_actions"][:, :t]
            if batch_inf else ep_batch["avail_actions"][:, t:t+1])
        if getattr(self.args, "use_individual_Q", False):
            agent_outs, self.hidden_states, individual_Q = self.agent(agent_inputs, self.hidden_states)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.transpose(1, 2).reshape(ep_batch.batch_size * self.n_agents, epi_len, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        if getattr(self.args, "use_individual_Q", False):
            return (agent_outs.view(ep_batch.batch_size, self.n_agents, -1),
                individual_Q.view(ep_batch.batch_size, self.n_agents, -1))
        if batch_inf:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, epi_len, -1).transpose(1, 2)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t, batch_inf):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        if batch_inf:
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, :t])  # bTav
            if self.args.obs_last_action:
                last_actions = th.zeros_like(batch["actions_onehot"][:, :t])
                last_actions[:, 1:] = batch["actions_onehot"][:, :t-1]
                inputs.append(last_actions)
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents).expand(bs, t, -1, -1))

            inputs = th.cat([x.transpose(1, 2).reshape(bs*self.n_agents, t, -1) for x in inputs], dim=2)
            return inputs

        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, 1, -1) for x in inputs], dim=2)
        return inputs
