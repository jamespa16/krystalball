"""Encoder -> ConvLSTM -> Decoder next-frame predictor, plus a swappable loss factory."""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM step: same recurrence as a normal LSTM, but every
    gate is a convolution instead of a matmul, so spatial structure survives."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding
        )

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, height, width, device):
        shape = (batch_size, self.hidden_channels, height, width)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


class Encoder(nn.Module):
    """Two stride-2 convs downsample 4x before the recurrent core, then a 1x1
    conv adds capacity (channel-wise MLP) without changing spatial resolution."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = base_channels * 2

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """A 1x1 conv adds capacity to the recurrent hidden state, then mirrors
    the encoder: two stride-2 transposed convs upsample back to working resolution."""

    def __init__(self, in_channels: int, base_channels: int = 16, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # pixels normalized to [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class NextFramePredictor(nn.Module):
    """encoder -> ConvLSTM -> decoder. Carries recurrent state across calls
    so predictions are conditioned on motion history, not a single frame."""

    def __init__(self, in_channels: int = 3, encoder_base_channels: int = 16,
                 hidden_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.encoder = Encoder(in_channels, encoder_base_channels)
        self.lstm = ConvLSTMCell(self.encoder.out_channels, hidden_channels, kernel_size)
        self.decoder = Decoder(hidden_channels, encoder_base_channels, in_channels)

    def forward(self, x, hidden):
        features = self.encoder(x)
        if hidden is None:
            b, _, h, w = features.shape
            hidden = self.lstm.init_hidden(b, h, w, x.device)
        h_new, c_new = self.lstm(features, hidden)
        pred = self.decoder(h_new)
        return pred, (h_new, c_new)


def detach_hidden(hidden):
    """Cut the (h, c) state loose from the autograd graph so the next
    timestep's backward pass can't walk back through prior timesteps."""
    if hidden is None:
        return None
    h, c = hidden
    return h.detach(), c.detach()


def get_loss_fn(name: str):
    """Factory so the loss is swappable (e.g. for SSIM/perceptual later)
    without touching the training loop."""
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss function: {name!r}")
