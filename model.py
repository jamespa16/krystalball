"""Multi-scale U-Net Encoder -> hierarchical spatiotemporal-LSTM (PredRNN-
style dual C/M memory, spanning skip scales and bottleneck depth) ->
multi-scale Decoder next-frame predictor, with GroupNorm residual blocks,
an acceleration-aware optical-flow head, plus a swappable loss factory
(MSE / SSIM / blended)."""

import math
from dataclasses import dataclass, fields, is_dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictorHidden:
    """Recurrent state threaded through NextFramePredictor.forward, one
    call to the next. None (the dataclass default for every field) on a
    fresh model / right after a reset. Every field must round-trip through
    detach_hidden -- see its dataclass branch below, which recurses generically
    without needing to know this shape."""
    st_lstm: "STLSTMHiddenState | None" = None  # HierarchicalSTLSTM's hidden state (hc list + shared m)
    prev_frame: "torch.Tensor | None" = None  # previous call's raw input (for flow estimation)
    prev_flow: "torch.Tensor | None" = None   # previous step's flow field (refined, not re-derived, next step)
    prev_flow_velocity: "torch.Tensor | None" = None  # previous step's flow-velocity estimate (see FlowHead)


@dataclass
class PredictorOutput:
    """Everything one NextFramePredictor.forward() call produces beyond the
    updated hidden state, so callers never need to reach into hidden-state
    internals to get the flow field / latents needed for loss computation."""
    pred: torch.Tensor
    hidden: PredictorHidden
    flow: "torch.Tensor | None"
    flow_velocity: "torch.Tensor | None" = None
    bottleneck_latent: "torch.Tensor | None" = None  # post-recurrence bottleneck h; world-model target space
    warped_base: "torch.Tensor | None" = None        # flow-warped base frame fed to the decoder's residual path


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


class FlowHead(nn.Module):
    """Predicts a dense 2-channel (dx, dy) flow field from a (prev, curr)
    frame pair, refined against the previous step's flow estimate rather
    than re-derived from scratch every frame -- reduces frame-to-frame
    flow jitter, since the field is otherwise supervised only indirectly
    through the downstream photometric loss. Small conv encoder-decoder
    reusing DownBlock/UpBlock for consistency with the main Encoder/
    Decoder. The final conv is zero-initialized so a fresh model predicts
    exactly zero flow (identity warp), mirroring the zero-init trick on
    Decoder.final_up.

    With `use_acceleration`, also predicts a second 2-channel field:
    flow VELOCITY, i.e. the rate of change of the flow field itself
    (conditioned on its own previous estimate, same refine-not-re-derive
    principle as flow), so the caller can warp using flow+velocity
    (constant-acceleration extrapolation) instead of just flow
    (constant-velocity extrapolation) -- lets fast-changing motion
    (accelerating/decelerating/turning) be tracked instead of always
    assuming the last-observed displacement continues unchanged. Also
    zero-initialized, so `use_acceleration=True` is architecturally
    additive at construction: a fresh model still predicts exactly zero
    flow and zero velocity."""

    def __init__(self, in_channels: int = 6, hidden_channels: int = 16,
                 use_acceleration: bool = True):
        super().__init__()
        self.use_acceleration = use_acceleration
        cond_channels = 4 if use_acceleration else 2  # prev_flow[, prev_velocity]
        out_channels = 4 if use_acceleration else 2  # flow[, velocity]
        self.in_conv = nn.Conv2d(in_channels + cond_channels, hidden_channels, 3, padding=1)
        self.down = DownBlock(hidden_channels, hidden_channels * 2, num_res_blocks=1)
        self.up = UpBlock(hidden_channels * 2, hidden_channels, hidden_channels, num_res_blocks=1)
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, prev_frame, curr_frame, prev_flow, prev_velocity=None):
        parts = [prev_frame, curr_frame, prev_flow]
        if self.use_acceleration:
            parts.append(prev_velocity)
        x = self.in_conv(torch.cat(parts, dim=1))
        skip = x
        x = self.down(x)
        x = self.up(x, skip)
        out = self.out_conv(x)
        if self.use_acceleration:
            return out[:, :2], out[:, 2:]
        return out, None


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp `frame` by `flow` (pixel-unit displacement): for each
    output pixel p, sample frame at (p - flow(p)). Used to extrapolate the
    current frame one step forward under a constant-velocity assumption."""
    b, c, h, w = frame.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=frame.device, dtype=frame.dtype),
        torch.arange(w, device=frame.device, dtype=frame.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([xs, ys], dim=-1).unsqueeze(0)  # (1, H, W, 2), (x, y) order
    sample_px = base_grid - flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
    norm_x = 2.0 * sample_px[..., 0] / max(w - 1, 1) - 1.0
    norm_y = 2.0 * sample_px[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _resize_flow(flow: torch.Tensor, size: tuple) -> torch.Tensor:
    """Downsample a pixel-unit flow field to `size` (h, w), rescaling
    displacement magnitudes to match the new pixel grid (flow values are in
    pixel units of the resolution they were computed at, so halving spatial
    size must also halve the dx/dy magnitudes)."""
    _, _, h, w = flow.shape
    new_h, new_w = size
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    scale = torch.tensor([new_w / w, new_h / h], device=flow.device, dtype=flow.dtype).view(1, 2, 1, 1)
    return resized * scale


@dataclass
class STLSTMHiddenState:
    """Hidden state for one HierarchicalSTLSTM. `hc` is list[(h, c)], one
    pair per cell in zigzag order (finest skip scale ... coarsest skip
    scale ... bottleneck-depth cells). `m` is the single shared
    spatiotemporal memory tensor, carried from the last (deepest) cell's
    output at the end of one timestep to become the first (finest) cell's
    input at the START of the next timestep -- a persistent cross-scale
    "memory highway" that today's independent per-scale cells have no
    equivalent of."""
    hc: list
    m: torch.Tensor


class STLSTMCell(nn.Module):
    """One PredRNN-style spatiotemporal-LSTM cell. Unlike a plain
    ConvLSTMCell, carries TWO memories: `c` (temporal, persists across TIME
    for this scale/depth only -- the same role a normal ConvLSTM cell state
    plays) and `m` (spatiotemporal, flows ACROSS SCALES within a single
    timestep, seeded from the previous cell's m output in
    HierarchicalSTLSTM's zigzag order). `h` fuses information from both
    memories via a 1x1 conv, so downstream consumers (the next cell, or the
    decoder) see one tensor blending temporal and cross-scale context."""

    def __init__(self, in_channels: int, hidden_channels: int, m_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.m_channels = m_channels
        padding = kernel_size // 2
        self.conv_temporal = nn.Conv2d(
            in_channels + hidden_channels, 3 * hidden_channels, kernel_size, padding=padding
        )
        self.conv_st = nn.Conv2d(in_channels + m_channels, 3 * m_channels, kernel_size, padding=padding)
        self.conv_o = nn.Conv2d(
            in_channels + hidden_channels * 2 + m_channels, hidden_channels, kernel_size, padding=padding
        )
        self.fuse = nn.Conv2d(hidden_channels + m_channels, hidden_channels, 1)

    def forward(self, x, h_prev, c_prev, m_in):
        i, f, g = torch.chunk(self.conv_temporal(torch.cat([x, h_prev], dim=1)), 3, dim=1)
        c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)

        i2, f2, g2 = torch.chunk(self.conv_st(torch.cat([x, m_in], dim=1)), 3, dim=1)
        m = torch.sigmoid(f2) * m_in + torch.sigmoid(i2) * torch.tanh(g2)

        o = torch.sigmoid(self.conv_o(torch.cat([x, h_prev, c, m], dim=1)))
        h = o * torch.tanh(self.fuse(torch.cat([c, m], dim=1)))
        return h, c, m

    def init_hidden(self, batch_size, height, width, device):
        shape = (batch_size, self.hidden_channels, height, width)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def init_m(self, batch_size, height, width, device):
        return torch.zeros(batch_size, self.m_channels, height, width, device=device)


class HierarchicalSTLSTM(nn.Module):
    """One STLSTMCell per encoder scale (finest skip scale first, then
    progressively coarser skip scales, then `lstm_layers` bottleneck-depth
    cells), unifying what used to be two separate, non-interacting
    mechanisms (an independent-per-scale ConvLSTM bank for skip
    connections, plus a wholly separate stacked ConvLSTM at the bottleneck)
    into one hierarchy with genuine cross-scale information flow: the
    spatiotemporal memory `m` zigzags finest -> coarsest within a single
    timestep (each cell's m output seeds the next cell's m input), and the
    final (deepest) cell's m becomes next timestep's seed for the finest
    cell -- the main structural fix for what used to cap fast/fine motion
    tracking at whatever the single coarsest bottleneck scale could
    represent.

    `xs` is a list of per-scale inputs: encoder skip features (shallowest/
    finest first) followed by the encoder's coarsest `features` -- one
    entry per RECURRENT INPUT SCALE, not one per cell. Bottleneck-depth
    cells beyond the first don't have their own scale input; like a
    standard stacked RNN, they take the previous cell's `h` as input
    instead (`layer_input` below), matching today's ConvLSTMStack
    layer-feeds-layer convention."""

    def __init__(self, in_channels_list: list, hidden_channels_list: list,
                 m_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.cells = nn.ModuleList([
            STLSTMCell(c_in, c_hidden, m_channels, kernel_size)
            for c_in, c_hidden in zip(in_channels_list, hidden_channels_list)
        ])
        self.m_channels = m_channels
        self.out_channels = hidden_channels_list[-1]

    def forward(self, xs: list, hidden: "STLSTMHiddenState | None"):
        if hidden is None:
            hc = [
                cell.init_hidden(xs[min(i, len(xs) - 1)].shape[0], xs[min(i, len(xs) - 1)].shape[2],
                                  xs[min(i, len(xs) - 1)].shape[3], xs[min(i, len(xs) - 1)].device)
                for i, cell in enumerate(self.cells)
            ]
            m = self.cells[0].init_m(xs[0].shape[0], xs[0].shape[2], xs[0].shape[3], xs[0].device)
        else:
            hc, m = hidden.hc, hidden.m

        new_hc, outs, layer_input = [], [], None
        for idx, (cell, (h_prev, c_prev)) in enumerate(zip(self.cells, hc)):
            x = xs[idx] if idx < len(xs) else layer_input
            if m.shape[-2:] != x.shape[-2:]:
                m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=True)
            h, c, m = cell(x, h_prev, c_prev, m)
            new_hc.append((h, c))
            outs.append(h)
            layer_input = h
        return outs, STLSTMHiddenState(hc=new_hc, m=m)


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
                 res_blocks_per_scale: int = 1, use_blend_mask: bool = False):
        super().__init__()
        self.delta_scale = delta_scale
        self.use_blend_mask = use_blend_mask
        self.out_channels = out_channels
        num_scales = len(skip_channels) + 1
        up_channels = [base_channels * (2 ** i) for i in range(num_scales)][::-1]
        skip_rev = list(reversed(skip_channels))

        blocks, prev_channels = [], in_channels
        for i in range(num_scales - 1):
            out_c = up_channels[i + 1]
            blocks.append(UpBlock(prev_channels, skip_rev[i], out_c, res_blocks_per_scale))
            prev_channels = out_c
        self.up_blocks = nn.ModuleList(blocks)
        # With blend masking, final_up outputs delta (out_channels) + a
        # fully-generated pixel estimate (out_channels) + a 1-channel blend
        # mask, instead of just delta -- see forward().
        final_out = out_channels * 2 + 1 if use_blend_mask else out_channels
        self.final_up = nn.ConvTranspose2d(prev_channels, final_out, 4, stride=2, padding=1)
        # Zero-init the layer that produces the residual delta: at
        # construction, delta = delta_scale * tanh(0) = 0, so a fresh
        # model's first prediction is exactly the input frame rather than a
        # random-magnitude delta from default init.
        nn.init.zeros_(self.final_up.weight)
        nn.init.zeros_(self.final_up.bias)
        if use_blend_mask:
            # Learnable scalar gate on the mask, zero-initialized: at
            # construction mask = sigmoid(0) * 0 = 0 exactly, so prediction
            # is exactly base_frame + delta (delta is also 0 from the
            # zero-init above) -- the same exact-identity-at-init guarantee
            # as the rest of this decoder, not just an approximation. Once
            # training produces gradient favoring the generation pathway,
            # this gate moves off zero and the mask starts contributing.
            self.mask_gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state, skips, base_frame):
        x = hidden_state
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)
        raw = self.final_up(x)
        if self.use_blend_mask:
            oc = self.out_channels
            delta = self.delta_scale * torch.tanh(raw[:, :oc])
            gen = torch.sigmoid(raw[:, oc:2 * oc])
            # Clamp the gate to [0, 1] so mask stays a valid convex-blend
            # weight no matter how far gradient descent pushes the raw
            # parameter -- otherwise mask could exceed 1 and make (1 - mask)
            # negative, amplifying rather than blending the warped path.
            gate = torch.clamp(self.mask_gate, 0.0, 1.0)
            mask = torch.sigmoid(raw[:, 2 * oc:2 * oc + 1]) * gate
            return (mask * gen + (1 - mask) * (base_frame + delta)).clamp(0.0, 1.0)
        delta = self.delta_scale * torch.tanh(raw)
        return (base_frame + delta).clamp(0.0, 1.0)


class NextFramePredictor(nn.Module):
    """(flow-warp) -> encoder -> HierarchicalSTLSTM (spanning both skip
    scales and bottleneck depth) -> decoder. Carries recurrent state
    across calls so predictions are conditioned on motion history, not a
    single frame, at multiple resolutions with genuine cross-scale
    information flow within a step (not just independent per-scale
    memory). `hidden` is a `PredictorHidden` (or None on a fresh model /
    after reset); `forward` returns a `PredictorOutput`, not a raw tuple,
    since the set of things callers need out of a step (prediction,
    updated hidden state, flow field, bottleneck latent) has grown past
    what a positional tuple can carry safely."""

    def __init__(self, in_channels: int = 3, encoder_base_channels: int = 32,
                 encoder_scales: int = 3, res_blocks_per_scale: int = 1,
                 lstm_hidden_channels: int = 128, lstm_layers: int = 2,
                 kernel_size: int = 3, delta_scale: float = 0.6,
                 use_flow: bool = True, flow_hidden_channels: int = 16,
                 use_blend_mask: bool = True, skip_lstm_base_channels: int = 16,
                 use_flow_acceleration: bool = True, st_memory_channels: int = 32,
                 world_model_hidden_channels: int = 64,
                 initial_weights: dict = None, uncertainty_clamp: float = 6.0):
        super().__init__()
        self.use_flow = use_flow
        self.use_flow_acceleration = use_flow_acceleration and use_flow
        if use_flow:
            self.flow_head = FlowHead(
                in_channels * 2, flow_hidden_channels, use_acceleration=self.use_flow_acceleration
            )
        # +2 channels for flow, +2 more for velocity when acceleration is enabled.
        # Stashed on self so encode_frame() (a standalone, history-free encoder
        # pass used by the world-model loss's target embedding) can zero-pad
        # the same channel count without duplicating this computation.
        self.flow_channels = (4 if self.use_flow_acceleration else 2) if use_flow else 0
        encoder_in_channels = in_channels + self.flow_channels
        self.encoder = Encoder(
            encoder_in_channels, encoder_base_channels, encoder_scales, res_blocks_per_scale
        )
        self.use_skip_recurrence = skip_lstm_base_channels > 0 and len(self.encoder.skip_channels) > 0
        # Bottleneck-depth cells beyond the first take the previous cell's
        # h as input (stacked-RNN convention), so only the first bottleneck
        # cell needs an explicit input-channel entry for `features`.
        bottleneck_in = [self.encoder.out_channels] + [lstm_hidden_channels] * (lstm_layers - 1)
        bottleneck_hidden_channels = [lstm_hidden_channels] * lstm_layers
        if self.use_skip_recurrence:
            skip_lstm_hidden = [
                skip_lstm_base_channels * (2 ** i) for i in range(len(self.encoder.skip_channels))
            ]
            in_channels_list = list(self.encoder.skip_channels) + bottleneck_in
            hidden_channels_list = skip_lstm_hidden + bottleneck_hidden_channels
            decoder_skip_channels = skip_lstm_hidden
        else:
            in_channels_list = bottleneck_in
            hidden_channels_list = bottleneck_hidden_channels
            decoder_skip_channels = self.encoder.skip_channels
        self.st_lstm = HierarchicalSTLSTM(in_channels_list, hidden_channels_list, st_memory_channels, kernel_size)
        self.decoder = Decoder(
            self.st_lstm.out_channels, decoder_skip_channels, encoder_base_channels,
            in_channels, delta_scale, res_blocks_per_scale, use_blend_mask
        )
        # Cooperates with the generator (unlike the adversarial Discriminator,
        # which deliberately lives outside this module) -- rides in the same
        # optimizer as everything else. Not called from forward(); invoked
        # explicitly by training.py's world-model loss wiring, like the other
        # auxiliary losses.
        self.world_model_head = WorldModelHead(
            self.st_lstm.out_channels, self.encoder.out_channels, world_model_hidden_channels, kernel_size
        )
        # Same rationale as world_model_head above: not called from
        # forward(), invoked explicitly by training.py's loss wiring, rides
        # in the same optimizer -- see UncertaintyWeighter.
        self.loss_weighter = UncertaintyWeighter(initial_weights or {}, clamp=uncertainty_clamp)

    def encode_frame(self, frame):
        """Stateless, history-free encoder-only forward pass on a standalone
        frame -- used to compute the world-model loss's target embedding
        from a real future frame sitting in the replay buffer, with no
        recurrent state or flow-warp involved. Zero-pads the same flow/
        velocity channel slots the encoder normally receives (there's no
        motion history for a lone frame processed this way), so channel
        counts match regardless of --use-flow/--flow-acceleration."""
        if self.flow_channels:
            pad = torch.zeros(
                frame.shape[0], self.flow_channels, frame.shape[2], frame.shape[3],
                device=frame.device, dtype=frame.dtype,
            )
            encoder_in = torch.cat([frame, pad], dim=1)
        else:
            encoder_in = frame
        features, _ = self.encoder(encoder_in)
        return features

    def forward(self, x, hidden):
        if hidden is None:
            hidden = PredictorHidden()
        prev_frame = hidden.prev_frame
        prev_flow = hidden.prev_flow
        prev_velocity = hidden.prev_flow_velocity

        if self.use_flow:
            if prev_frame is not None:
                if prev_flow is None:
                    prev_flow = torch.zeros(
                        x.shape[0], 2, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
                    )
                if self.use_flow_acceleration and prev_velocity is None:
                    prev_velocity = torch.zeros_like(prev_flow)
                flow, velocity = self.flow_head(prev_frame, x, prev_flow, prev_velocity)
                # Constant-acceleration extrapolation (flow+velocity) when
                # enabled, else today's constant-velocity extrapolation
                # (flow alone). This extrapolated field is ephemeral -- used
                # only to warp the base frame/skips below, never stored in
                # hidden state (the raw flow/velocity pair is what persists).
                flow_extrap = flow + velocity if self.use_flow_acceleration else flow
                warped = warp_frame(x, flow_extrap)
            else:
                flow = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                velocity = flow.clone() if self.use_flow_acceleration else None
                flow_extrap = flow
                warped = x
            encoder_in = torch.cat([x, flow, velocity], dim=1) if self.use_flow_acceleration else torch.cat([x, flow], dim=1)
        else:
            flow, velocity, flow_extrap = None, None, None
            warped, encoder_in = x, x

        features, skips = self.encoder(encoder_in)

        # Recurrence runs on the raw, unwarped encoder outputs, so the
        # stored hidden state (both `hc` and the shared `m`) always lives
        # in one stable coordinate frame -- warping is applied only to the
        # ephemeral per-step output below (discarded after the decoder
        # consumes it), never fed back in as next step's input. Feeding
        # warped output back in would re-resample an already-resampled
        # tensor every step, compounding bilinear blur over a long-running
        # session.
        if self.use_skip_recurrence:
            xs = list(skips) + [features]
            outs, st_hidden = self.st_lstm(xs, hidden.st_lstm)
            n_skip = len(skips)
            skip_outs, bottleneck_latent = outs[:n_skip], outs[-1]
        else:
            outs, st_hidden = self.st_lstm([features], hidden.st_lstm)
            skip_outs, bottleneck_latent = skips, outs[-1]

        if self.use_flow and prev_frame is not None:
            # Skip features were captured from the current (unwarped) frame,
            # but base_frame is now motion-extrapolated -- warp the skips by
            # the same extrapolated field (rescaled per-scale) so the detail
            # they add is spatially consistent with the predicted-frame
            # geometry instead of ghosting at moving edges.
            skip_outs = [warp_frame(s, _resize_flow(flow_extrap, s.shape[2:])) for s in skip_outs]

        pred = self.decoder(bottleneck_latent, skip_outs, warped)
        new_hidden = PredictorHidden(
            st_lstm=st_hidden, prev_frame=x, prev_flow=flow, prev_flow_velocity=velocity,
        )
        return PredictorOutput(
            pred=pred, hidden=new_hidden, flow=flow, flow_velocity=velocity,
            bottleneck_latent=bottleneck_latent, warped_base=warped,
        )


def detach_hidden(hidden):
    """Cut every tensor in `hidden` loose from the autograd graph so the
    next timestep's backward pass can't walk back through prior timesteps.
    Recurses through the opaque (nested tuple/list of tensors) hidden-state
    structure -- callers never need to know its shape."""
    if hidden is None:
        return None
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    if isinstance(hidden, tuple):
        return tuple(detach_hidden(h) for h in hidden)
    if isinstance(hidden, list):
        return [detach_hidden(h) for h in hidden]
    if is_dataclass(hidden):
        return replace(hidden, **{f.name: detach_hidden(getattr(hidden, f.name)) for f in fields(hidden)})
    raise TypeError(f"Unexpected hidden-state element: {type(hidden)!r}")


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def motion_weight_map(frame_t: torch.Tensor, frame_prev: torch.Tensor,
                       motion_loss_weight: float) -> torch.Tensor:
    """Per-pixel loss weight, higher where the frame actually changed, so
    gradient concentrates on real motion instead of being diluted by the
    (already trivially correct) static background. Mean-normalized so the
    weight map's own mean is always exactly 1 + motion_loss_weight,
    regardless of how much motion is actually in a given frame -- keeps
    overall loss magnitude stable frame-to-frame."""
    diff = (frame_t - frame_prev).abs().mean(dim=1, keepdim=True)
    denom = diff.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    return 1.0 + motion_loss_weight * (diff / denom)


def flow_smoothness_loss(flow: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Edge-aware total-variation regularizer on the flow field: penalizes
    flow discontinuities except where the image itself has an edge (motion
    boundaries legitimately coincide with object edges). Needed because
    FlowHead is supervised only indirectly through downstream photometric
    reconstruction (see FlowHead/NextFramePredictor.forward) -- in
    textureless or occluded regions there's no data term to anchor the
    flow at all, so without this it can drift to noisy/degenerate values.
    `image` is detached: no gradient should flow into the edge-weight
    computation, only into the flow field being regularized."""
    image = image.detach()
    dx_flow = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy_flow = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dx_img = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy_img = image[:, :, 1:, :] - image[:, :, :-1, :]
    w_x = torch.exp(-dx_img.abs().mean(dim=1, keepdim=True))
    w_y = torch.exp(-dy_img.abs().mean(dim=1, keepdim=True))
    return (dx_flow.abs() * w_x).mean() + (dy_flow.abs() * w_y).mean()


def motion_delta_loss(pred: torch.Tensor, target: torch.Tensor,
                       prev_frame: torch.Tensor) -> torch.Tensor:
    """Supervises the *magnitude* of predicted frame-to-frame change against
    actual change, complementary to motion_weight_map's pixel-reweighting
    (which only reweights the base photometric loss, it doesn't add a new
    signal). Comparing (pred - prev) to (target - prev) directly under any
    pointwise norm would be mathematically identical to comparing (pred,
    target) directly -- subtracting the same prev_frame from both sides
    cancels out exactly. Taking .abs() before differencing is what makes
    this a genuinely distinct signal: it matches the *amount* of change
    regardless of direction/color, isolated from whether exact final pixel
    values are right (already covered by the base loss)."""
    pred_delta = (pred - prev_frame).abs()
    real_delta = (target - prev_frame).abs()
    return F.mse_loss(pred_delta, real_delta)


class WorldModelHead(nn.Module):
    """Asymmetric predictor mapping the recurrent core's current bottleneck
    latent to a predicted FUTURE latent in a distinct representation
    space -- the encoder's own (feedforward, no recurrence) embedding of
    a real future frame. Conditioned on the current flow/velocity as a
    motion cue for how far content will have moved by the target horizon.
    Deliberately a separate small network from the encoder/ST-LSTM whose
    output it's trained to match, rather than comparing raw bottleneck
    space to itself -- one of this loss's collapse-resistance measures
    (see world_model_consistency_loss and TrainingEngine's usage)."""

    def __init__(self, bottleneck_channels: int, target_channels: int,
                 hidden_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(bottleneck_channels + 4, hidden_channels, kernel_size, padding=padding)
        self.norm1 = _group_norm(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.norm2 = _group_norm(hidden_channels)
        self.out = nn.Conv2d(hidden_channels, target_channels, 1)

    def forward(self, bottleneck_latent, flow, velocity):
        b, _, h, w = bottleneck_latent.shape
        if flow is not None:
            flow_ds = _resize_flow(flow, (h, w))
        else:
            flow_ds = torch.zeros(b, 2, h, w, device=bottleneck_latent.device, dtype=bottleneck_latent.dtype)
        vel_ds = _resize_flow(velocity, (h, w)) if velocity is not None else torch.zeros_like(flow_ds)
        x = F.relu(self.norm1(self.conv1(torch.cat([bottleneck_latent, flow_ds, vel_ds], dim=1))))
        x = F.relu(self.norm2(self.conv2(x)))
        return self.out(x)


def world_model_consistency_loss(pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity latent-consistency loss -- deliberately NOT raw
    MSE: MSE against a detached target can be cheaply "solved" by
    shrinking overall activation magnitude toward zero rather than
    learning real predictive structure, whereas a normalized (cosine)
    comparison removes that shortcut. One of several collapse-resistance
    measures for this loss, alongside the stop-gradient target (see
    TrainingEngine's peek/encode_frame usage, always under no_grad) and
    WorldModelHead's architectural asymmetry from its own input space."""
    pred_n = F.normalize(pred_latent, dim=1)
    target_n = F.normalize(target_latent, dim=1)
    return (1.0 - (pred_n * target_n).sum(dim=1)).mean()


class UncertaintyWeighter(nn.Module):
    """Learned homoscedastic uncertainty weighting (Kendall, Gal & Cipolla
    2018) for the auxiliary loss terms composed in compute_training_loss.
    Each term's static CLI weight becomes only its INITIAL value; from then
    on `log_var_i` is a trained parameter, so `weight_i = exp(-log_var_i)`
    adapts to that term's own loss scale instead of staying fixed -- at
    equilibrium `weight_i ~= 1/loss_i`, so every term's weighted
    contribution converges toward a comparable magnitude automatically.
    Terms whose initial weight is 0 get no parameter at all, so setting a
    CLI flag to 0 still fully disables that term, unchanged from before.
    `log_var` is clamped since this project's single-frame (batch-size-1)
    online updates make each term's gradient a noisy one-sample estimate,
    and there's no checkpointing to recover from a runaway value."""

    def __init__(self, initial_weights: dict, clamp: float = 6.0):
        super().__init__()
        self.clamp = clamp
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(-math.log(w)))
            for name, w in initial_weights.items() if w > 0
        })

    def _clamped(self, name):
        return torch.clamp(self.log_vars[name], -self.clamp, self.clamp)

    def weight(self, name):
        return torch.exp(-self._clamped(name))

    def weighted_term(self, name, loss):
        log_var = self._clamped(name)
        return torch.exp(-log_var) * loss + log_var


class Discriminator(nn.Module):
    """PatchGAN-style discriminator for the adversarial loss: a spectral-
    normalized (not GroupNorm/BatchNorm -- spectral norm needs no batch
    statistics at all, sidestepping the batch-size-1 online-training
    problem entirely, unlike GroupNorm which would be an awkward, purely
    approximate fit here) stride-2 conv stack producing a SPATIAL map of
    real/fake logits rather than one scalar per frame, so gradient
    concentrates on which regions look fake (typically the blur-prone
    regions this whole feature targets) instead of an undifferentiated
    whole-frame verdict. Lives outside NextFramePredictor (constructed and
    optimized separately by TrainingEngine) and is trained from scratch,
    like everything else in this project -- no pretrained weights."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_layers: int = 3):
        super().__init__()
        layers, c_in = [], in_channels
        for i in range(num_layers):
            c_out = base_channels * (2 ** i)
            layers += [
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.out_conv = nn.utils.parametrizations.spectral_norm(nn.Conv2d(c_in, 1, 3, padding=1))

    def forward(self, frame):
        return self.out_conv(self.features(frame))  # (B, 1, h', w') patch logits


def lsgan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """LSGAN (least-squares GAN) discriminator loss -- quadratic, not the
    vanilla BCE-GAN log-loss, specifically to avoid the classic cold-start
    problem: BCE's log(1 - D(fake)) saturates hard once D confidently
    rejects an early, still-noisy predictor, giving near-zero generator
    gradient exactly when it's needed most. LSGAN keeps penalizing
    "confidently wrong" logits proportionally instead."""
    return 0.5 * (F.mse_loss(real_logits, torch.ones_like(real_logits))
                  + F.mse_loss(fake_logits, torch.zeros_like(fake_logits)))


def lsgan_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss: push the discriminator's verdict on the
    generator's own output toward "real" (see lsgan_d_loss)."""
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


class WeightedMSELoss(nn.Module):
    """MSE with an optional per-pixel weight map; identical to plain MSE
    when weight_map is None."""

    def forward(self, pred, target, weight_map=None):
        sq = (pred - target) ** 2
        if weight_map is not None:
            sq = sq * weight_map
        return sq.mean()


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

    def forward(self, pred, target, weight_map=None):
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
        loss_map = 1.0 - ssim_map
        if weight_map is not None:
            loss_map = loss_map * weight_map
        return loss_map.mean()


class BlendedLoss(nn.Module):
    """final = (1 - ssim_weight) * MSE + ssim_weight * (1 - SSIM)."""

    def __init__(self, ssim_weight: float = 0.5, channels: int = 3):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse = WeightedMSELoss()
        self.ssim = SSIMLoss(channels=channels)

    def forward(self, pred, target, weight_map=None):
        return (1 - self.ssim_weight) * self.mse(
            pred, target, weight_map=weight_map
        ) + self.ssim_weight * self.ssim(pred, target, weight_map=weight_map)


def get_loss_fn(name: str, ssim_weight: float = 0.5, channels: int = 3):
    """Factory so the loss is swappable without touching the training loop."""
    name = name.lower()
    if name == "mse":
        return WeightedMSELoss()
    if name == "ssim":
        return SSIMLoss(channels=channels)
    if name == "mse_ssim":
        return BlendedLoss(ssim_weight=ssim_weight, channels=channels)
    raise ValueError(f"Unknown loss function: {name!r}")
