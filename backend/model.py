"""Conditional UNet with depthwise TCN blocks for flow matching.

Architecture:
  - Depthwise separable convolutions (TCN-style, dilated)
  - Adaptive Group Normalization conditioned on 688-dim tensor + timestep
  - ~10M parameters
  - Output: same spatial size as input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DepthwiseTCNBlock(nn.Module):
    """Depthwise separable conv block with dilated TCN-style convolutions.

    Three stacked layers with dilation 1, 2, 4 for growing receptive field.
    Structure: GN → SiLU → DWConv2d(d) → PWConv2d → residual.
    """

    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in [1, 2, 4]:
            self.layers.append(nn.Sequential(
                nn.GroupNorm(32, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                          groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class AdaGN(nn.Module):
    """Adaptive GroupNorm: per-channel scale & shift from conditioning."""

    def __init__(self, groups, channels, cond_dim):
        super().__init__()
        self.gn = nn.GroupNorm(groups, channels, affine=False)
        self.mlp = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        B, C = x.shape[:2]
        scale, shift = self.mlp(cond)[:, :, None, None].chunk(2, dim=1)
        return self.gn(x) * (1 + scale) + shift


class DownBlock(nn.Module):
    """Encoder: DepthwiseTCNBlock → downsample → AdaGN."""

    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.0):
        super().__init__()
        self.tcn = DepthwiseTCNBlock(in_ch, dropout)
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.ada_gn = AdaGN(32, out_ch, cond_dim)

    def forward(self, x, cond):
        x = self.tcn(x)
        x = self.proj(x)
        x = self.down(x)
        x = self.ada_gn(x, cond)
        return x


class UpBlock(nn.Module):
    """Decoder: upsample → concat skip → DepthwiseTCNBlock → AdaGN → proj."""

    def __init__(self, in_ch, skip_ch, out_ch, cond_dim, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        concat_ch = out_ch + skip_ch
        self.tcn = DepthwiseTCNBlock(concat_ch, dropout)
        self.ada_gn = AdaGN(32, concat_ch, cond_dim)
        self.proj = nn.Conv2d(concat_ch, out_ch, 1)

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.ada_gn(x, cond)
        x = self.tcn(x)
        x = self.proj(x)
        return x


class MiddleBlock(nn.Module):
    """Bottleneck: self-attention + DepthwiseTCNBlock."""

    def __init__(self, channels, cond_dim):
        super().__init__()
        self.tcn = DepthwiseTCNBlock(channels)
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ada_gn = AdaGN(32, channels, cond_dim)

    def forward(self, x, cond):
        x = self.tcn(x)
        B, C, H, W = x.shape
        xf = x.flatten(2).transpose(1, 2)
        a, _ = self.attn(self.norm(xf), self.norm(xf), self.norm(xf))
        x = x + a.transpose(1, 2).reshape(B, C, H, W)
        x = self.ada_gn(x, cond)
        return x


class ConditionalUNet(nn.Module):
    """Conditional UNet: (noisy_img, 688-dim, t) → predicted velocity.

    Encoder produces 3 feature levels (stride 2 each):
      64→32→16→8. Decoder goes 8→16→32→64.
    """

    def __init__(self, img_channels=3, img_size=64, base_channels=64,
                 cond_dim=688, embed_dim=256, dropout=0.0):
        super().__init__()
        self.img_size = img_size

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )

        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 4),
        )

        total_cond_dim = embed_dim * 8
        bc = base_channels

        # Input projection
        self.input_proj = nn.Conv2d(img_channels, bc, 3, padding=1)

        # Encoder: 64→32→16→8
        self.enc1 = DownBlock(bc, bc, total_cond_dim, dropout)       # 64→32
        self.enc2 = DownBlock(bc, bc * 2, total_cond_dim, dropout)   # 32→16
        self.enc3 = DownBlock(bc * 2, bc * 4, total_cond_dim, dropout)  # 16→8

        # Middle
        self.middle = MiddleBlock(bc * 4, total_cond_dim)

        # Decoder: 8→16→32→64
        self.dec3 = UpBlock(bc * 4, bc * 2, bc * 2, total_cond_dim, dropout)  # 8→16
        self.dec2 = UpBlock(bc * 2, bc, bc, total_cond_dim, dropout)           # 16→32
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(bc, bc, 4, stride=2, padding=1),               # 32→64
            DepthwiseTCNBlock(bc, dropout),
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.GroupNorm(32, bc),
            nn.SiLU(),
            nn.Conv2d(bc, img_channels, 3, padding=1),
        )

        n = sum(p.numel() for p in self.parameters())
        print(f"[ConditionalUNet] params: {n:,}")

    def forward(self, x, cond, t):
        t_emb = sinusoidal_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        c_emb = self.cond_embed(cond)
        combined = torch.cat([t_emb, c_emb], dim=-1)

        h = self.input_proj(x)

        # Encoder with skips
        s1 = self.enc1(h, combined)   # 32x32
        s2 = self.enc2(s1, combined)  # 16x16
        h = self.enc3(s2, combined)   # 8x8

        # Middle
        h = self.middle(h, combined)

        # Decoder (skip s2, s1 — not the bottleneck)
        h = self.dec3(h, s2, combined)  # 16x16
        h = self.dec2(h, s1, combined)  # 32x32
        h = self.dec1(h)                 # 64x64

        return self.out_proj(h)
