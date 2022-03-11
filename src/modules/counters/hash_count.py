import argparse
import numpy as np
import torch as th
import time
from modules.counters.intrinsic_reward import IntrinsicReward


class HashCount(IntrinsicReward):
    """
    Hash-based count bonus for exploration class

    Paper:
    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, O. X., Duan, Y., ... & Abbeel, P. (2017).
    #Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in neural information processing systems (pp. 2753-2762).

    Paper: https://arxiv.org/abs/1611.04717

    Open-source code:
        https://github.com/uoe-agents/derl/blob/main/derl/intrinsic_rewards/count/hash_count.py
        https://github.com/openai/EPG/blob/master/epg/exploration.py
    """
    def __init__(
        self,
        args,
    ):
        """
        Initialise parameters for hash count intrinsic reward
        :param state_shape: observation space of environment
        :param args: intrinsic reward configuration dict
        """
        super().__init__(args=args)
        self.tables = None

        # Hashing function: SimHash
        self.bucket_sizes = self.args.bucket_sizes
        if self.bucket_sizes is None:
            # Large prime numbers
            self.bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]

        self.mods_list = th.stack(
            [2**th.arange(self.args.key_dim) % bucket_size for bucket_size in self.bucket_sizes],
            dim=1).to(dtype=th.float32, device=self.device)
        self.bucket_sizes = th.tensor(self.bucket_sizes, dtype=th.float32, device=self.device)

        self.reset()
        self.projection_matrix = th.randn(
            self.state_shape, self.args.key_dim, device=self.device)

    def reset(self):
        """
        Reset counting
        tables: (n_buc, max_buc_prime, (n_act,)*n_ag)
        """
        self.tables = th.zeros(
            (len(self.bucket_sizes),
            th.max(self.bucket_sizes).to(th.int64),) + (self.n_actions,)*self.n_agents,
            device=self.device,
        )

    def compute_keys(self, obss):
        binaries = th.sign(obss @ self.projection_matrix)
        keys = (binaries @ self.mods_list) % self.bucket_sizes
        return keys

    def update_count_by_key(self, keys, acts=None):
        """
        Batch size might be B*n_agent if no param sharing.
        https://stackoverflow.com/questions/70997018/pytorch-index-on-multiple-dimension-tensor-in-a-batch
        https://stackoverflow.com/questions/65894166/pytorch-how-to-do-gathers-over-multiple-dimensions
        """
        bs = keys.shape[0]
        keys = keys.to(th.int64)
        for idx in range(len(self.bucket_sizes)):
            if acts is None:
                self.tables[idx, keys[:, idx]] += self.args.decay_factor
            else:
                self.tables[(idx, keys[:, idx],)+ acts.T.split(split_size=1, dim=0)] += self.args.decay_factor

    def get_count_by_key(self, keys, acts=None):
        """
        None acts queries all actions.
        """
        bs = keys.shape[0]
        keys = keys.to(th.int64)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            if acts is None:
                all_counts.append(self.tables[idx, keys[:, idx]])
            else:
                all_counts.append(self.tables[(idx, keys[:, idx],)+ acts.T.split(split_size=1, dim=0)].squeeze(dim=0))
        counts = th.stack(all_counts, dim=0).min(dim=0)[0]
        return counts

    def save_models(self, path):
        th.save(self.mods_list, f"{path}/simhash_mods_list.pt")
        th.save(self.bucket_sizes, f"{path}/simhash_bucket_sizes.pt")
        th.save(self.projection_matrix, f"{path}/simhash_projection_matrix.pt")
        th.save(self.tables, f"{path}/simhash_tables.pt")

    def load_models(self, path):
        self.mods_list = th.load(f"{path}/simhash_mods_list.pt", map_location=self.device)
        self.bucket_sizes = th.load(f"{path}/simhash_bucket_sizes.pt", map_location=self.device)
        self.projection_matrix = th.load(f"{path}/simhash_projection_matrix.pt", map_location=self.device)
        self.tables = th.load(f"{path}/simhash_tables.pt", map_location=self.device)


def test_simhash(args):
    simhash = HashCount(state_shape=args.state_shape, args=args)

    batch = 4
    obss = th.randn(size=(batch, args.state_shape), device=args.device)
    keys = simhash.compute_keys(obss=obss)
    assert keys.shape[0] == batch

    acts = np.random.randint(
        low=0, high=args.n_actions, size=(batch, args.n_agents))
    acts = th.tensor(acts, dtype=th.int64, device=args.device)

    start_time = time.time()
    print(f'acts:\n{acts}\nincrement joint action counts')
    simhash._inc_hash(obss=obss, acts=acts)
    counts = simhash._query_hash(obss=obss, acts=acts)
    print(f"counts:\n{counts}, shape: {counts.shape}")
    counts_none = simhash._query_hash(obss=obss)
    print(f"counts_none:\n{counts_none}, shape: {counts_none.shape}")

    print(f'increment state counts')
    simhash._inc_hash(obss=obss)
    counts = simhash._query_hash(obss=obss, acts=acts)
    print(f"counts:\n{counts}, shape: {counts.shape}")
    counts_none = simhash._query_hash(obss=obss)
    print(f"counts_none:\n{counts_none}, shape: {counts_none.shape}")

    print(f"time elapsed: {time.time()-start_time}")

def test_conditional_count(args):
    bs = args.bs
    counts = th.tensor(
        list(reversed(range(bs* args.n_actions**args.n_agents))),
            dtype=th.float32, device=args.device).view(
            (bs,)+(args.n_actions,)*args.n_agents)
    ucb_indices = th.zeros((bs, args.n_agents, args.n_actions), device=args.device)
    act_t = th.zeros((bs, args.n_agents), dtype=th.int64, device=args.device)

    parent_counts = counts.sum(dim=tuple(range(1, args.n_agents+1))).unsqueeze(-1)
    for ag_i in range(args.n_agents):
        if ag_i == args.n_agents-1:
            child_counts = counts
        else:
            child_counts = counts.sum(dim=tuple(range(2, len(counts.shape))))
        uncert = parent_counts / child_counts
        ucb_indices[:, ag_i] = uncert
        max_idx = ucb_indices[:, ag_i].argmax(dim=1)
        act_t[:, ag_i] = max_idx
        if ag_i != args.n_agents-1:
            counts = counts[th.arange(bs), max_idx]
            parent_counts = th.gather(child_counts, dim=1, index=max_idx.view(bs, 1))
    print(f"ucb_indices:\n{ucb_indices}")
    print(f"act_t:\n{act_t}")

def test_independent_count(args):
    bs = args.bs
    counts = th.tensor(
        list(reversed(range(bs* args.n_actions**args.n_agents))),
            dtype=th.float32, device=args.device).view(
            (bs,)+(args.n_actions,)*args.n_agents)
    ucb_indices = th.zeros((bs, args.n_agents, args.n_actions), device=args.device)
    act_t = th.zeros((bs, args.n_agents), dtype=th.int64, device=args.device)

    t_ep = counts.sum().item() * 2 # supposed each action has been taken twice

    for ag_i in range(args.n_agents):
        other_agent_dims = tuple(
            dim for dim in range(1, args.n_agents+1) if dim != ag_i+1)
        act_counts = counts.sum(dim=other_agent_dims)
        uncert = t_ep / act_counts
        ucb_indices[:, ag_i] = uncert
        max_idx = ucb_indices[:, ag_i].argmax(dim=1)
        act_t[:, ag_i] = max_idx
    print(f"ucb_indices:\n{ucb_indices}")
    print(f"act_t:\n{act_t}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test", type=str, default="simhash",
        choices=["simhash", "cond_count", "indep_count"],)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--n_agents", type=int, default=2)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--bucket_sizes", type=str, default="53,59,61,67,71,73")
    parser.add_argument("--key_dim", type=int, default=7)
    parser.add_argument("--state_shape", type=int, default=16)
    parser.add_argument("--decay_factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.bucket_sizes = list(map(int, args.bucket_sizes.split(",")))
    if not th.cuda.is_available():
        args.device = "cpu"
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.test == "simhash":
        test_simhash(args)
    elif args.test == "cond_count":
        test_conditional_count(args)
    elif args.test == "indep_count":
        test_independent_count(args)