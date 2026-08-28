"""Small causal temporal convolutional network for hysteresis regression."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """Conv1d with left-only padding, so output[t] never sees input[t + 1:]."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.activation(self.conv2(x)))
        return self.activation(x + residual)


class HysteresisTCN(nn.Module):
    """Causal TCN with the same constructor contract as the existing GRU."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        bidirectional=False,
        tcn_kernel_size=3,
        **kwargs,
    ):
        super().__init__()
        if bidirectional:
            raise ValueError("A causal TCN cannot be bidirectional")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if "n_heads" in kwargs:
            print("Info: TCN ignores n_heads")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tcn_kernel_size = tcn_kernel_size
        blocks = []
        in_channels = input_size
        for layer_index in range(num_layers):
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=tcn_kernel_size,
                    dilation=2**layer_index,
                    dropout=dropout,
                )
            )
            in_channels = hidden_size
        self.network = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_size, output_size)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def receptive_field(self):
        # Each residual block contains two convolutions at the same dilation.
        dilation_sum = 2**self.num_layers - 1
        return 1 + 2 * (self.tcn_kernel_size - 1) * dilation_sum

    def forward(self, x):
        # (batch, time, features) -> (batch, features, time)
        features = self.network(x.transpose(1, 2))
        return self.fc(features[:, :, -1])
