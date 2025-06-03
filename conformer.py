import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1d = nn.BatchNorm2d(out_channels)

        self.pool = nn.AvgPool2d(kernel_size=(2, 1))

    def forward(self, x):
        x_res = self.conv1d(x)
        x_res = self.bn1d(x_res)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x + x_res
        x = self.pool(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()

        self.block1 = ResBlock(input_dim, 24, dropout)
        self.block2 = ResBlock(24, 48, dropout)
        self.block3 = ResBlock(48, 96, dropout)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.mhsa = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x, _ = self.mhsa(x, x, x, need_weights=False)
        x = self.drop(x)
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(input_dim)
        self.conv1 = nn.Conv1d(input_dim, 2 * input_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.conv2 = nn.Conv1d(
            input_dim, input_dim, groups=input_dim, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm1d(input_dim)
        self.swish = nn.SiLU()
        self.conv3 = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.glu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1)

        return x


class Conformer(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()

        self.ff1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 4 * input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * input_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.mhsa = MultiHeadSelfAttentionModule(input_dim, 8, dropout)
        self.conv = ConvolutionModule(input_dim, dropout)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 4 * input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * input_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = 0.5 * self.ff1(x) + x
        x = self.mhsa(x) + x
        x = self.conv(x) + x
        x = 0.5 * self.ff2(x) + x
        x = self.ln(x)

        return x


class SELDConformerBackbone(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()

        self.resnet = ResNet(input_dim=input_dim, dropout=dropout)
        self.fc1 = nn.Linear(96 * 8, 128)
        self.conf = nn.ModuleList([Conformer(128, dropout) for _ in range(8)])
        self.pool = nn.AvgPool2d(kernel_size=(5, 1))

    def forward(self, x):
        x = self.resnet(x)

        # Move time dimension to the second position and flatten features
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(*x.shape[:2], -1)
        x = self.fc1(x)

        for conformer in self.conf:
            x = conformer(x)

        x = self.pool(x)

        return x
