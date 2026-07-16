import torch
from torch import nn


class HysteresisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, bidirectional, **kwargs):
        super(HysteresisLSTM, self).__init__()

        if 'n_heads' in kwargs: print("Info: GRU ignores n_heads")

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first=True means (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)

        # Fully connected layer to map from hidden state space to output space (2,)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, x):
        # x is of shape (batch_size, window_size, input_size)

        # Init hidden state (h_0) and cell state (c_0) to zeroes.
        # (data sequence should start from (0,0) after preprocessing)
        device = x.device
        h0 = torch.zeros(self.num_layers * 2 if self.lstm.bidirectional else self.num_layers,
                         x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2 if self.lstm.bidirectional else self.num_layers,
                         x.size(0), self.hidden_size).to(device)

        # Pass state and input to LSTM
        # out contains outputs of LSTM for all time steps
        # (h_n, c_n) is now hidden state and cell state
        out, _ = self.lstm(x, (h0, c0))

        # We just want the last sequence element
        # (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])

        return out