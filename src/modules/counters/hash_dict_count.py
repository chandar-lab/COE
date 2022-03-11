import argparse
import numpy as np
import torch as th
import time
from collections import defaultdict
from modules.counters.intrinsic_reward import IntrinsicReward


class BaseDictCount(IntrinsicReward):
    def __init__(
        self,
        args,
    ):
        super().__init__(args=args)
        self.reset()

    def reset(self):
        """
        Reset counting
        count_table: {tuple: tensor(size=(n_act,)*n_ag))}
        """
        self.count_table = defaultdict(lambda: th.zeros(
            (self.n_actions,)*self.n_agents, device=self.device))

    def update_count_by_key(self, keys, acts=None):
        if acts is None:
            for key in keys:
                self.count_table[tuple(key.tolist())] += self.args.decay_factor
        else:
            for key,act in zip(keys, acts):
                self.count_table[tuple(key.tolist())][act.split(split_size=1, dim=0)] += self.args.decay_factor

    def get_count_by_key(self, keys, acts=None):
        counts = []
        if acts is None:
            for key in keys:
                counts.append(self.count_table[tuple(key.tolist())])
        else:
            for key,act in zip(keys, acts):
                counts.append(self.count_table[tuple(key.tolist())][act.split(split_size=1, dim=0)])
        counts_tensor = th.stack(counts, dim=0)
        return counts_tensor

    def save_models(self, path):
        th.save(dict(self.count_table), f"{path}/count_table.pt")

    def load_models(self, path):
        count_table = th.load(f"{path}/count_table.pt", map_location=self.device)
        self.count_table = defaultdict(lambda: th.zeros(
            (self.n_actions,)*self.n_agents, device=self.device),
            count_table)


class HashDictCount(BaseDictCount):
    def __init__(
        self,
        args,
    ):
        super().__init__(args=args)
        self.key_dim = self.args.key_dim
        self.projection_matrix = th.randn(
            self.state_shape, self.key_dim, device=self.device)

    def compute_keys(self, obss):
        binaries = th.sign(obss @ self.projection_matrix)
        return binaries

    def save_models(self, path):
        super().save_models(path=path)
        th.save(self.projection_matrix, f"{path}/simhash_projection_matrix.pt")

    def load_models(self, path):
        super().load_models(path=path)
        self.projection_matrix = th.load(f"{path}/simhash_projection_matrix.pt", map_location=self.device)


def test_simhash_dict(args):
    counter = HashDictCount(args=args)

    states = th.randn(size=(args.bs, args.state_shape), device=args.device)
    keys = counter.compute_keys(obss=states)

    acts = np.random.randint(
        low=0, high=args.n_actions, size=(args.bs, args.n_agents))
    acts = th.tensor(acts, dtype=th.int64, device=args.device)

    start_time = time.time()
    print(f'acts:\n{acts}\nincrement joint action counts')
    counter.update_count_by_key(keys=keys, acts=acts)
    counts = counter.get_count_by_key(keys=keys, acts=acts)
    print(f"counts:\n{counts}, shape: {counts.shape}")
    counts_none = counter.get_count_by_key(keys=keys)
    print(f"counts_none:\n{counts_none}, shape: {counts_none.shape}")

    print(f'increment state counts')
    counter.update_count_by_key(keys=keys)
    counts = counter.get_count_by_key(keys=keys, acts=acts)
    print(f"counts:\n{counts}, shape: {counts.shape}")
    counts_none = counter.get_count_by_key(keys=keys)
    print(f"counts_none:\n{counts_none}, shape: {counts_none.shape}")

    print(f"time elapsed: {time.time()-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test", type=str, default="simhash_dict",
        choices=["simhash_dict"],)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--key_dim", type=int, default=16)
    parser.add_argument("--state_shape", type=int, default=16)
    parser.add_argument("--decay_factor", type=float, default=1.0)
    parser.add_argument("--episode_limit", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if not th.cuda.is_available():
        args.device = "cpu"
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.test == "simhash_dict":
        test_simhash_dict(args)