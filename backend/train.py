"""Flow matching training loop.

Trains a ConditionalUNet to predict the velocity field v_t(x)
for a probability path between noise (t=0) and data (t=1).
"""

import os
import sys

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

from backend.model import ConditionalUNet
from backend.dataset import SyntheticDataset


def flow_matching_loss(model, x1, cond):
    """Flow matching loss: L2 on predicted velocity.

    Args:
        model: ConditionalUNet
        x1: (B, C, H, W) clean data samples
        cond: (B, 688) conditioning tensors

    Returns:
        loss scalar
    """
    B = x1.shape[0]

    # Sample random time steps uniformly
    t = torch.rand(B, device=x1.device)  # uniform [0, 1]

    # Sample noise
    x0 = torch.randn_like(x1)

    # Interpolate: x_t = (1 - t) * x0 + t * x1
    t_reshape = t.view(B, 1, 1, 1)
    xt = (1 - t_reshape) * x0 + t_reshape * x1

    # Target velocity: v = x1 - x0 (constant velocity path)
    target = x1 - x0

    # Predict
    pred = model(xt, cond, t)

    return F.mse_loss(pred, target)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for cond, x1 in dataloader:
        cond = cond.to(device)
        x1 = x1.to(device)
        loss = flow_matching_loss(model, x1, cond)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * cond.shape[0]
        n += cond.shape[0]
    return total_loss / n


def train(model, dataloader, n_epochs=100, lr=1e-4, device="cpu"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        loss = train_epoch(model, dataloader, optimizer, device)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{n_epochs}: loss = {loss:.6f}")

    return model


def sample(model, cond, n_steps=3, device="cpu"):
    """Generate image from conditioning with flow matching ODE.

    Uses midpoint method for 1-3 steps.

    Args:
        model: ConditionalUNet
        cond: (1, 688) conditioning tensor
        n_steps: number of ODE steps (1, 2, or 3)
        device: torch device

    Returns:
        (1, C, H, W) generated image in [0, 1]
    """
    model.eval()
    img_size = model.img_size

    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size, device=device)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.full((1,), step * dt, device=device, dtype=torch.float32)

            # Midpoint method
            v1 = model(x, cond, t)
            x_mid = x + 0.5 * dt * v1

            # Cap x_mid if needed
            t_mid = t + 0.5 * dt
            v2 = model(x_mid, cond, t_mid)
            x = x + dt * v2

        return x.clamp(0, 1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Small test
    model = ConditionalUNet(img_size=64).to(device)

    ds = SyntheticDataset(img_size=64)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    cond, x1 = next(iter(dl))
    print(f"Cond: {cond.shape}, Image: {x1.shape}")

    # Forward pass test
    t = torch.rand(4, device=device)
    pred = model(x1.to(device), cond.to(device), t)
    print(f"Prediction: {pred.shape}, range: [{pred.min():.4f}, {pred.max():.4f}]")

    # Loss test
    loss = flow_matching_loss(model, x1.to(device), cond.to(device))
    print(f"Flow matching loss: {loss.item():.6f}")

    print("\nForward pass OK. Ready to train.")
