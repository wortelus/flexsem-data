import torch
from torch import nn

class HysteresisGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, bidirectional, **kwargs):
        super(HysteresisGRU, self).__init__()

        if 'n_heads' in kwargs: print("Info: GRU ignores n_heads")

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Using nn.GRU instead of nn.LSTM
        # batch_first=True means (batch_size, seq_len, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout)

        # Fully connected layer to map from hidden state space to output space
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, x):
        # x is of shape (batch_size, window_size, input_size)
        out, _ = self.gru(x)  # h0 defaults to zeros internally
        out = self.fc(out[:, -1, :])
        return out