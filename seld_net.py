import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout):
        super().__init__()
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SELDNetBackbone(nn.Module):
    def __init__(
        self,
        num_classes,
        num_events,
        input_dim,
        hidden_dim,
        dropout,
        rnn_layers,
        mhsa_layers,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_events = num_events
        self.mhsa_layers = mhsa_layers
        self.dropout = dropout

        self.convs = nn.Sequential(
            ConvBlock(input_dim, hidden_dim, (3, 3), (4, 5), dropout),
            ConvBlock(hidden_dim, hidden_dim, (3, 3), (4, 1), dropout),
            ConvBlock(hidden_dim, hidden_dim, (3, 3), (2, 1), dropout),
        )

        self.rnn = nn.GRU(
            hidden_dim * 2,
            hidden_dim,
            rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.mhsa = nn.ModuleList()
        self.ln = nn.ModuleList()
        for _ in range(mhsa_layers):
            self.mhsa.append(
                nn.MultiheadAttention(
                    hidden_dim * 2, 8, batch_first=True, dropout=dropout
                )
            )
            self.ln.append(nn.LayerNorm(hidden_dim * 2))

    def forward(self, x, lengths):
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(0, 2, 1)

        x = pack_padded_sequence(
            x,
            lengths / 5,
            batch_first=True,
            enforce_sorted=False,
        )
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = F.tanh(x)

        for i in range(self.mhsa_layers):
            x = x + self.mhsa[i](x, x, x, need_weights=False)[0]
            x = self.ln[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x
