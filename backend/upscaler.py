"""TCN Upscaler: 64×64 + 688-dim conditioning → 256×256.

A lightweight depthwise TCN that super-resolves the 64×64 output
from Stage 1 into 256×256, conditioned on the original 688-dim tensor.

Architecture:
  - 8× nearest-neighbor upsample (64→256 is 4× per axis = 2 stages)
  - Each stage: TCN block with AdaGN conditioning
  - Output: 256×256×3
  - Total: ~2M params (tiny, fast to train)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TCNUpscaleBlock(nn.Module):
    """Upscale 2× using depthwise TCN with AdaGN conditioning."""

    def __init__(self, channels, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
        # TCN with dilations 1, 2, 4 (same as Stage 1)
        self.layers = nn.ModuleList()
        for d in [1, 2, 4]:
            self.layers.append(nn.Sequential(
                nn.GroupNorm(16, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                          groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=True),
            ))
        self.ada_gn = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        x = self.up(x)
        scale, shift = self.ada_gn(cond)[:, :, None, None].chunk(2, dim=1)
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        x = x * (1 + scale) + shift
        return x


class TCNUpscaler(nn.Module):
    """64×64 + 688-dim → 256×256 in 2 stages (64→128→256)."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=64,
                 cond_dim=688, embed_dim=128):
        super().__init__()

        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )

        self.cond_proj = nn.Identity()

        # Input feature extraction
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Two upscale stages: 64→128, 128→256
        self.upscale1 = TCNUpscaleBlock(base_channels, embed_dim * 2)
        self.upscale2 = TCNUpscaleBlock(base_channels, embed_dim * 2)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

        n = sum(p.numel() for p in self.parameters())
        print(f"[TCNUpscaler] params: {n:,}")

    def forward(self, x, cond):
        """
        Args:
            x: (B, 3, 64, 64) low-res image from Stage 1
            cond: (B, 688) conditioning tensor
        Returns:
            (B, 3, 256, 256) high-res image
        """
        c = self.cond_embed(cond)

        h = self.input_proj(x)
        h = self.upscale1(h, c)  # 64 → 128
        h = self.upscale2(h, c)  # 128 → 256
        out = self.out_proj(h)

        return out


def train_upscaler_epoch(model, dl, optimizer, device):
    """Train TCNUpscaler on paired low-res / high-res images.

    dl yields (low_res_64, high_res_256, cond)
    """
    model.train()
    total_loss = 0
    n = 0
    for lr, hr, cond in dl:
        lr, hr, cond = lr.to(device), hr.to(device), cond.to(device)
        pred = model(lr, cond)
        loss = F.mse_loss(pred, hr)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * lr.shape[0]
        n += lr.shape[0]
    return total_loss / n if n > 0 else 0.0


if __name__ == "__main__":
    model = TCNUpscaler()
    x = torch.randn(2, 3, 64, 64)
    cond = torch.randn(2, 688)
    out = model(x, cond)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 3, 256, 256), f"Wrong shape: {out.shape}"
    print("Shape OK")
