class BaseBandit:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def sample(self, state, test_mode):
        raise NotImplementedError

    def update_returns(self, states, noise, returns, test_mode, t):
        pass

    def cuda(self):
        pass

    def save_model(self, path):
        pass