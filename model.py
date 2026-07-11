"""Multi-scale U-Net Encoder -> stacked ConvLSTM -> multi-scale Decoder
next-frame predictor, with GroupNorm residual blocks, plus a swappable
loss factory (MSE / SSIM / blended)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Largest group count <= max_groups that evenly divides `channels`.
    GroupNorm, never BatchNorm: training runs at batch size 1 (one live
    webcam frame at a time), so there's no batch to compute running
    statistics over -- BatchNorm's train-mode behavior degrades badly here."""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    """Pre-activation residual block: (GroupNorm -> ReLU -> Conv3x3) x2 + skip."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)

    def forward(self, x):
        residual = x
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        return residual + x


class DownBlock(nn.Module):
    """Stride-2 Conv (+GroupNorm+ReLU) then `num_res_blocks` ResBlocks."""

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 1):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = _group_norm(out_channels)
        self.res_blocks = nn.ModuleList(
            [ResBlock(out_channels) for _ in range(num_res_blocks)]
        )

    def forward(self, x):
        x = F.relu(self.norm(self.down(x)))
        for block in self.res_blocks:
            x = block(x)
        return x


class UpBlock(nn.Module):
    """ConvTranspose2d stride-2 (+GroupNorm+ReLU), concat matching-res skip,
    1x1 fuse conv, then `num_res_blocks` ResBlocks."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 num_res_blocks: int = 1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = _group_norm(out_channels)
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, 1)
        self.res_blocks = nn.ModuleList(
            [ResBlock(out_channels) for _ in range(num_res_blocks)]
        )

    def forward(self, x, skip):
        x = F.relu(self.norm(self.up(x)))
        x = self.fuse(torch.cat([x, skip], dim=1))
        for block in self.res_blocks:
            x = block(x)
        return x


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


class ConvLSTMStack(nn.Module):
    """Stacked multi-layer ConvLSTM. Layer i's hidden output feeds layer
    i+1's input (standard stacked-RNN convention), giving deeper/
    hierarchical temporal reasoning than a single cell. Hidden state for
    the whole stack is a list[(h, c)], one tuple per layer."""

    def __init__(self, in_channels: int, hidden_channels: list, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = list(hidden_channels)
        self.out_channels = self.hidden_channels[-1]
        cells, prev_channels = [], in_channels
        for hc in self.hidden_channels:
            cells.append(ConvLSTMCell(prev_channels, hc, kernel_size))
            prev_channels = hc
        self.cells = nn.ModuleList(cells)

    def forward(self, x, hidden):
        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.shape[2], x.shape[3], x.device)
        new_hidden, layer_input = [], x
        for cell, h in zip(self.cells, hidden):
            h_new, c_new = cell(layer_input, h)
            new_hidden.append((h_new, c_new))
            layer_input = h_new
        return layer_input, new_hidden

    def init_hidden(self, batch_size, height, width, device):
        return [cell.init_hidden(batch_size, height, width, device) for cell in self.cells]


class Encoder(nn.Module):
    """`num_scales` stride-2 DownBlocks, channels doubling from
    `base_channels`. Returns (features, skips): features is the final
    (most-downsampled) activation fed to the ConvLSTM stack; skips is a
    list of activations at each intermediate scale, shallowest-first, for
    the decoder's skip fusion."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 num_scales: int = 3, res_blocks_per_scale: int = 1):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(num_scales)]
        blocks, prev_channels = [], in_channels
        for c in channels:
            blocks.append(DownBlock(prev_channels, c, res_blocks_per_scale))
            prev_channels = c
        self.down_blocks = nn.ModuleList(blocks)
        self.num_scales = num_scales
        self.skip_channels = channels[:-1]
        self.out_channels = channels[-1]

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < self.num_scales - 1:
                skips.append(x)
        return x, skips


class Decoder(nn.Module):
    """Mirrors the encoder: one UpBlock per skip (fused deepest-first),
    followed by one final ConvTranspose2d (no skip -- symmetric with the
    encoder having no skip at full input resolution) that outputs
    base_frame + delta_scale * tanh(conv_out), i.e. a residual delta added
    to the input frame rather than pixels reconstructed from scratch."""

    def __init__(self, in_channels: int, skip_channels: list, base_channels: int = 32,
                 out_channels: int = 3, delta_scale: float = 0.6,
                 res_blocks_per_scale: int = 1):
        super().__init__()
        self.delta_scale = delta_scale
        num_scales = len(skip_channels) + 1
        up_channels = [base_channels * (2 ** i) for i in range(num_scales)][::-1]
        skip_rev = list(reversed(skip_channels))

        blocks, prev_channels = [], in_channels
        for i in range(num_scales - 1):
            out_c = up_channels[i + 1]
            blocks.append(UpBlock(prev_channels, skip_rev[i], out_c, res_blocks_per_scale))
            prev_channels = out_c
        self.up_blocks = nn.ModuleList(blocks)
        self.final_up = nn.ConvTranspose2d(prev_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, hidden_state, skips, base_frame):
        x = hidden_state
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)
        delta = self.delta_scale * torch.tanh(self.final_up(x))
        return base_frame + delta


class NextFramePredictor(nn.Module):
    """encoder -> stacked ConvLSTM -> decoder. Carries recurrent state
    across calls so predictions are conditioned on motion history, not a
    single frame. `hidden` is None or list[(h, c)], one tuple per ConvLSTM
    layer."""

    def __init__(self, in_channels: int = 3, encoder_base_channels: int = 32,
                 encoder_scales: int = 3, res_blocks_per_scale: int = 1,
                 lstm_hidden_channels: int = 128, lstm_layers: int = 2,
                 kernel_size: int = 3, delta_scale: float = 0.6):
        super().__init__()
        self.encoder = Encoder(
            in_channels, encoder_base_channels, encoder_scales, res_blocks_per_scale
        )
        self.lstm = ConvLSTMStack(
            self.encoder.out_channels, [lstm_hidden_channels] * lstm_layers, kernel_size
        )
        self.decoder = Decoder(
            self.lstm.out_channels, self.encoder.skip_channels, encoder_base_channels,
            in_channels, delta_scale, res_blocks_per_scale
        )

    def forward(self, x, hidden):
        features, skips = self.encoder(x)
        top_h, hidden = self.lstm(features, hidden)
        pred = self.decoder(top_h, skips, x)
        return pred, hidden


def detach_hidden(hidden):
    """Cut every layer's (h, c) state loose from the autograd graph so the
    next timestep's backward pass can't walk back through prior timesteps.
    `hidden` is None or list[(h, c)], one tuple per ConvLSTM layer."""
    if hidden is None:
        return None
    return [(h.detach(), c.detach()) for h, c in hidden]


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM, pure-PyTorch depthwise-conv Gaussian-window implementation
    (no external ssim package, no pretrained weights). Operates on
    [0,1]-range NCHW tensors."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, channels: int = 3,
                 data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.data_range = data_range
        self.register_buffer("window", _gaussian_window(window_size, sigma, channels))

    def forward(self, pred, target):
        window = self.window.to(device=pred.device, dtype=pred.dtype)
        pad = self.window_size // 2
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        def conv(t):
            return F.conv2d(t, window, padding=pad, groups=self.channels)

        mu_p, mu_t = conv(pred), conv(target)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
        sig_p2 = conv(pred * pred) - mu_p2
        sig_t2 = conv(target * target) - mu_t2
        sig_pt = conv(pred * target) - mu_pt

        ssim_map = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / (
            (mu_p2 + mu_t2 + c1) * (sig_p2 + sig_t2 + c2)
        )
        return 1.0 - ssim_map.mean()


class BlendedLoss(nn.Module):
    """final = (1 - ssim_weight) * MSE + ssim_weight * (1 - SSIM)."""

    def __init__(self, ssim_weight: float = 0.5, channels: int = 3):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(channels=channels)

    def forward(self, pred, target):
        return (1 - self.ssim_weight) * self.mse(pred, target) + self.ssim_weight * self.ssim(
            pred, target
        )


def get_loss_fn(name: str, ssim_weight: float = 0.5, channels: int = 3):
    """Factory so the loss is swappable without touching the training loop."""
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "ssim":
        return SSIMLoss(channels=channels)
    if name == "mse_ssim":
        return BlendedLoss(ssim_weight=ssim_weight, channels=channels)
    raise ValueError(f"Unknown loss function: {name!r}")
