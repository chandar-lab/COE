from abc import ABC, abstractmethod


class IntrinsicReward(ABC):
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.episode_limit = args.episode_limit
        self.args = args
        self.device = self.args.device

    def get_count(self, obss, acts=None):
        keys = self.compute_keys(obss)
        return self.get_count_by_key(keys=keys, acts=acts), keys

    def update_count(self, obss, acts=None):
        self._inc_hash(obss=obss, acts=acts)

    def _inc_hash(self, obss, acts=None):
        keys = self.compute_keys(obss)
        self.update_count_by_key(keys=keys, acts=acts)

    def _query_hash(self, obss, acts=None):
        keys = self.compute_keys(obss)
        return self.get_count_by_key(keys=keys, acts=acts)

    @abstractmethod
    def compute_keys(self, obss):
        raise NotImplementedError

    @abstractmethod
    def update_count_by_key(self, keys, acts=None):
        raise NotImplementedError

    @abstractmethod
    def get_count_by_key(self, keys, acts=None):
        raise NotImplementedError

    def episode_reset(self, environment_idx):
        pass

    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
