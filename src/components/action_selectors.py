import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class BaseActionSelector():
    def __init__(self, args):
        pass

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        random_actions = Categorical(avail_actions.float()).sample().long()
        return random_actions

    def update_parameter(self):
        pass

REGISTRY["random"] = BaseActionSelector


class MultinomialActionSelector(BaseActionSelector):

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector(BaseActionSelector):

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector(BaseActionSelector):

    def __init__(self, args):
        self.args = args
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_inputs = agent_inputs.clone()
        self.epsilon = self.schedule.eval(t_env)

        if not test_mode:
            epsilon_action_num = masked_inputs.size(-1)
            if getattr(self.args, "mask_before_softmax", True):
                epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()
            masked_inputs = (
                (1 - self.epsilon) * masked_inputs
                + th.ones_like(masked_inputs) * self.epsilon/epsilon_action_num)
            if getattr(self.args, "mask_before_softmax", True):
                masked_inputs[avail_actions == 0] = 0.0

        m = Categorical(masked_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector
