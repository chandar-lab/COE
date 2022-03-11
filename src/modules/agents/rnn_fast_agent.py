import torch.nn as nn
import torch.nn.functional as F


class RNNFastAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRU(
                input_size=args.hidden_dim,
                num_layers=1,
                hidden_size=args.hidden_dim,
                batch_first=True,
            )
            self.forward = self.forward_rnn
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.forward = self.forward_ff
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward_rnn(self, inputs, hidden_state):
        bs, epi_len, num_feat = inputs.shape
        inputs = inputs.reshape(bs * epi_len, num_feat)
        x = F.relu(self.fc1(inputs))
        x = x.reshape(bs, epi_len, self.args.hidden_dim)
        h_in = hidden_state.reshape(1, bs, self.args.hidden_dim).contiguous()
        x, h = self.rnn(x, h_in)
        x = x.reshape(bs * epi_len, self.args.hidden_dim)
        q = self.fc2(x)
        q = q.reshape(bs, epi_len, self.args.n_actions)
        return q, h

    def forward_ff(self, inputs, hidden_state):
        bs, epi_len, num_feat = inputs.shape
        inputs = inputs.reshape(bs * epi_len, num_feat)
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.rnn(x))
        q = self.fc2(h)
        # h = h.reshape(bs, epi_len, self.args.hidden_dim)
        q = q.reshape(bs, epi_len, self.args.n_actions)
        return q, h

