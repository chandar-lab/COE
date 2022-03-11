import numpy as np
import torch as th

from .base_bandit import BaseBandit


class Constant_Lr(BaseBandit):

    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.lr = args.noise_bandit_lr
        self.returns = [0 for _ in range(self.args.noise_dim)]
        self.epsilon = args.noise_bandit_epsilon
        self.noise_dim = self.args.noise_dim

    def sample(self, state, test_mode):
        noise_vector = []
        for _ in range(self.args.batch_size_run):
            noise = 0
            # During training we are epsilon greedy.
            # During testing we are uniform so that we can gather info about all noise seeds
            if test_mode or np.random.random() < self.epsilon:
                noise = np.random.randint(self.noise_dim)
            else:
                noise = np.argmax(self.returns)
            one_hot_noise = th.zeros(self.noise_dim)
            one_hot_noise[noise] = 1
            noise_vector.append(one_hot_noise)
        return th.stack(noise_vector)

    def update_returns(self, states, noise, returns, test_mode, t):
        if test_mode:
            return # Only update the returns for training.
        for n, r in zip(noise, returns):
            # n is onehot
            n_idx = np.argmax(n)
            self.returns[n_idx] = self.lr * r + (1 - self.lr) * self.returns[n_idx]
