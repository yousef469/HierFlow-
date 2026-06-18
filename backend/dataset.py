"""Synthetic dataset: Renders 2D shapes from 688-dim tensors for training.

Uses the front-end to generate conditioning tensors from random prompts,
then renders simple colored shapes at the specified positions.
"""

import os
import sys

# Ensure frontend package is importable
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
import numpy as np
from frontend.encoder import prompt_to_tensor
from frontend import knowledge_base as kb
from frontend import parser as frontend_parser


_SAMPLE_PROMPTS = [
    "a red dog",
    "a blue cat",
    "a green bird",
    "a white rabbit",
    "a brown bear",
    "a yellow fish",
    "a black dog sitting",
    "a white cat sleeping",
    "a brown dog standing",
    "a red bird flying",
    "a blue fish swimming",
    "a green frog",
    "a white duck",
    "a black cat",
    "a brown horse",
    "a gray elephant",
    "a red fox",
    "a blue bird",
    "a green turtle",
    "a white sheep",
    "a brown cow",
    "a yellow lion",
    "a black bear",
    "a white swan",
    "a red parrot",
    "a brown eagle",
    "a gray wolf",
    "a white rabbit on green grass",
    "a brown dog under a tree",
    "a blue fish in water",
    "a red bird above a mountain",
    "a white cat on grass",
    "a brown bear near water",
    "a yellow bird flying in the sky",
    "a black dog on snow",
    "a green frog in water",
    "a white sheep on grass",
    "a red fox in a forest",
    "a blue bird flying above a lake",
    "a brown horse standing in a field",
    "a gray elephant near water",
    "a white duck swimming",
    "a black cat sitting on grass",
    "a brown cow in a field",
    "a yellow lion on grass",
    "a red parrot on a tree",
    "a white swan on water",
]


def _render_layout(layout_result, img_size=64):
    """Render layout as a simple 2D image with colored shapes.

    Args:
        layout_result: dict from layout.layout()
        img_size: output image size (H=W)

    Returns:
        numpy array (H, W, 3) in [0, 1]
    """
    H = W = img_size
    img = np.ones((H, W, 3), dtype=np.float32)

    # Background color
    scene = layout_result["scene"]
    bg = np.array([scene["bg_r"], scene["bg_g"], scene["bg_b"]], dtype=np.float32)
    img[:] = bg

    # Sort objects by depth (ascending = back to front)
    objs = sorted(layout_result["objects"], key=lambda o: o.get("depth", 0.5))

    for obj in objs:
        cx = obj["x"]
        cy = obj["y"]
        w = obj.get("w", 0.2)
        h = obj.get("h", 0.2)
        color = obj.get("color")
        if color is None or not isinstance(color, (tuple, list)):
            color = (0.6, 0.4, 0.2)

        # Map to pixel coords
        px = int(cx * W)
        py = int(cy * H)
        pw = max(4, int(w * W))
        ph = max(4, int(h * H))

        x0 = max(0, px - pw // 2)
        x1 = min(W, px + pw // 2)
        y0 = max(0, py - ph // 2)
        y1 = min(H, py + ph // 2)

        # Draw ellipse (approximate with filled circle/ellipse)
        yy, xx = np.ogrid[:y1 - y0, :x1 - x0]
        rx = (x1 - x0) / 2
        ry = (y1 - y0) / 2
        mask = ((xx - rx + 0.5) / max(rx, 1)) ** 2 + ((yy - ry + 0.5) / max(ry, 1)) ** 2 <= 1

        # Draw parts on top
        parts = obj.get("parts", [])
        if parts:
            # Use first part color for body
            c = np.array([color[0], color[1], color[2]], dtype=np.float32)
            img_slice = img[y0:y1, x0:x1]
            img_slice[mask] = c

            # Draw small dots for parts
            for part in parts[:5]:
                rx_p = part.get("rel_x", 0)
                ry_p = part.get("rel_y", 0)
                rw_p = max(2, int(part.get("rel_w", 0.05) * W))
                pc = np.array([part.get("color_r", 0.6),
                               part.get("color_g", 0.4),
                               part.get("color_b", 0.2)], dtype=np.float32)
                ppx = int((cx + rx_p) * W)
                ppy = int((cy + ry_p) * H)
                px0 = max(0, ppx - rw_p // 2)
                px1 = min(W, ppx + rw_p // 2)
                py0 = max(0, ppy - rw_p // 2)
                py1 = min(H, ppy + rw_p // 2)
                img[py0:py1, px0:px1] = pc
        else:
            c = np.array([color[0], color[1], color[2]], dtype=np.float32)
            img_slice = img[y0:y1, x0:x1]
            img_slice[mask] = c

    return img


class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset of (688-dim tensor, 64x64 image) pairs from random prompts."""

    def __init__(self, prompts=None, img_size=64, deterministic=False):
        self.img_size = img_size
        self.prompts = prompts or _SAMPLE_PROMPTS
        self.deterministic = deterministic

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        tensor, layout = prompt_to_tensor(prompt)
        img = _render_layout(layout, self.img_size)  # (H, W, 3)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        return tensor.float(), img_t


def create_dataloader(batch_size=32, img_size=64, shuffle=True):
    """Create training data loader."""
    ds = SyntheticDataset(img_size=img_size)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = SyntheticDataset()
    cond, img = ds[0]
    print(f"Conditioning: {cond.shape}")
    print(f"Image: {img.shape}")
    print(f"Range: [{img.min():.4f}, {img.max():.4f}]")

    # Preview first 4
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(min(8, len(ds))):
        cond, img = ds[i]
        r, c = i // 4, i % 4
        axes[r, c].imshow(img.permute(1, 2, 0).numpy())
        axes[r, c].set_title(ds.prompts[i][:30])
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig("/tmp/synthetic_preview.png", dpi=100)
    print("Preview saved to /tmp/synthetic_preview.png")
