"""Optimized 20K Stanford Dogs training at 64×64.

Loads everything into memory as one tensor, batches fast.
"""

import os, sys, pickle, time, glob
import numpy as np

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
from torch.utils.data import TensorDataset, DataLoader
from backend.model import ConditionalUNet

CACHE_PATH = os.path.join(_pkg_root, "data", "stanford_dogs", "full_64_cache.pkl")
CKPT_DIR = os.path.join(_pkg_root, "checkpoints")


def load_data():
    print("Loading cache...")
    t0 = time.time()
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} items in {time.time()-t0:.0f}s")

    # Stack into single tensors
    conds = torch.stack([d[0] for d in data])
    imgs = torch.stack([d[1] for d in data])
    return conds, imgs


def train():
    device = "cpu"
    batch_size = 32
    n_epochs = 50
    print_every = 100

    conds, imgs = load_data()
    ds = TensorDataset(conds, imgs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = ConditionalUNet(img_size=64)

    # Try loading synthetic pretrain
    sp = os.path.join(_pkg_root, "synthetic_pretrain.pt")
    if os.path.exists(sp):
        model.load_state_dict(torch.load(sp))
        print("Loaded synthetic pretrain")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        total_loss, n, t0 = 0, 0, time.time()
        for bi, (cond, img) in enumerate(dl):
            x0 = torch.randn_like(img)
            t = torch.rand(img.shape[0])
            xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * img
            loss = torch.nn.functional.mse_loss(
                model(xt, cond, t), img - x0
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * cond.shape[0]
            n += cond.shape[0]

            if (bi + 1) % print_every == 0:
                elapsed = time.time() - t0
                steps_left = len(dl) - (bi + 1)
                eta = steps_left * (elapsed / (bi + 1)) / 60
                print(f"  [{bi+1}/{len(dl)}] current_loss={total_loss/n:.4f} ETA={eta:.0f}min")
                sys.stdout.flush()

        scheduler.step()
        avg_loss = total_loss / n
        elapsed = time.time() - t0

        fp = os.path.join(CKPT_DIR, f"dogs64_epoch_{epoch+1:02d}.pt")
        torch.save(model.state_dict(), fp)

        marker = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "dogs64_best.pt"))
            marker = " *"
        epochs_left = n_epochs - epoch - 1
        eta_h = elapsed * epochs_left / 3600
        print(f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f} ({elapsed:.0f}s ETA {eta_h:.1f}h){marker}")
        sys.stdout.flush()

    print(f"\nDone! Best loss: {best_loss:.6f}")
    return best_loss


if __name__ == "__main__":
    train()
