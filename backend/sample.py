"""Sampling pipeline: prompt → image in 1-3 steps.

Entry point: prompt_to_image(prompt, n_steps=3) → PIL Image
"""

import os
import sys

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import numpy as np
from backend.model import ConditionalUNet
from backend.dataset import SyntheticDataset
from backend.train import sample


def load_model(checkpoint_path=None, img_size=64, device="cpu"):
    """Load trained model or return a fresh one."""
    model = ConditionalUNet(img_size=img_size)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found — using untrained model (will produce noise)")
    model.to(device)
    return model


def prompt_to_image(prompt, model=None, n_steps=3, device="cpu", img_size=64):
    """Full pipeline: prompt string → PIL Image."""
    from frontend.encoder import prompt_to_tensor

    if model is None:
        model = load_model(device=device, img_size=img_size)

    tensor, _ = prompt_to_tensor(prompt)
    cond = tensor.unsqueeze(0).to(device)

    generated = sample(model, cond, n_steps=n_steps, device=device)
    img = generated.squeeze(0).cpu().permute(1, 2, 0).numpy()

    return np.clip(img, 0, 1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)

    for prompt in [
        "a red dog standing",
        "a blue bird flying",
        "a white cat on grass",
    ]:
        img = prompt_to_image(prompt, model, n_steps=3, device=device)
        print(f"\"{prompt}\": image shape {img.shape}, range [{img.min():.4f}, {img.max():.4f}]")

    try:
        from PIL import Image
        img = prompt_to_image("a red dog", model, n_steps=3, device=device)
        Image.fromarray((img * 255).astype(np.uint8)).save("/tmp/prompt_output.png")
        print("Sample saved to /tmp/prompt_output.png")
    except ImportError:
        print("PIL not available — skipping save")
