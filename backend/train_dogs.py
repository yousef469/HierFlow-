"""Train ConditionalUNet on Stanford Dogs (CPU-optimized).

Precomputes 688-dim tensors once, then trains on a 2K subset at 32×32.
"""

import os, sys, json, time, pickle
import numpy as np
from pathlib import Path

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from backend.model import ConditionalUNet
from backend.stanford_dogs import StanfordDogsDataset, _STANFORD_BREED_OVERRIDES, IMAGES_DIR, ANNOT_DIR, _load_annotation
from frontend.encoder import prompt_to_tensor, encode_scene_to_tensor
from frontend import schema, knowledge_base as kb


CACHE_PATH = os.path.join(_pkg_root, "data", "stanford_dogs", "tensor_cache.pkl")


def _build_cache(max_samples, img_size):
    """Build list of (688-dim tensor, image tensor) pairs.

    Uses a generic prompt like "a brown dog standing" with the
    correct wnid from Stanford breed → KB mapping.
    """
    breed_dirs = sorted(os.listdir(IMAGES_DIR))
    cache = []
    total_added = 0

    for breed_dir in breed_dirs:
        if total_added >= max_samples:
            break
        breed_path = os.path.join(IMAGES_DIR, breed_dir)
        if not os.path.isdir(breed_path):
            continue

        breed_id = breed_dir.split("-")[0] if "-" in breed_dir else breed_dir
        wnid = _STANFORD_BREED_OVERRIDES.get(breed_id, 2084071)
        breed_name = breed_dir.split("-", 1)[1] if "-" in breed_dir else breed_dir

        # Annotation dir
        annot_breed_dir = os.path.join(ANNOT_DIR, breed_dir)
        if not os.path.exists(annot_breed_dir):
            annot_breed_dir = None

        images = sorted([f for f in os.listdir(breed_path)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        for img_name in images:
            if total_added >= max_samples:
                break
            img_path = os.path.join(breed_path, img_name)
            syn = kb.SYNSETS.get(wnid, {})
            base_name = syn.get("name", "dog").replace("_", " ")

            # Random attributes for variation
            color = np.random.choice(["brown", "black", "white", "golden", "gray", "tan"])
            action = np.random.choice(["standing", "sitting", "walking"])
            prompt = f"a {color} {base_name} {action}"

            # Get tensor from prompt
            tensor, layout = prompt_to_tensor(prompt)

            # Override wnid in tensor to match Stanford breed
            sv, objs, parts = schema.decode_scene(tensor)
            for oi in range(schema.MAX_OBJECTS):
                obj = objs[oi]
                if obj[0].item() > 0.01:
                    # This is an object slot — override wnid
                    obj[0] = wnid / 100_000_000.0
                    break
            tensor = schema.encode_scene(sv, objs, parts)

            # Override positions with bbox if available
            bbox = None
            if annot_breed_dir is not None:
                annot_name = os.path.splitext(img_name)[0]
                annot_path = os.path.join(annot_breed_dir, annot_name)
                if os.path.exists(annot_path):
                    bbox = _load_annotation(annot_path)

            if bbox is not None and len(layout["objects"]) > 0:
                cx, cy, bw, bh = bbox
                for oi in range(schema.MAX_OBJECTS):
                    obj = objs[oi]
                    if obj[0].item() > 0.01:
                        name = kb.SYNSETS.get(int(obj[0].item() * 100_000_000), {}).get("name", "")
                        if name not in ("sky", "ground", "grass", "tree", "water", "cloud", "mountain", "sun"):
                            obj[1] = max(0.0, min(1.0, cx))  # x
                            obj[2] = max(0.0, min(1.0, cy))  # y
                            obj[3] = max(0.05, min(1.0, bw))  # w
                            obj[4] = max(0.05, min(1.0, bh))  # h
                            break
                tensor = schema.encode_scene(sv, objs, parts)

            # Load and resize image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0

            cache.append((tensor.float(), img_t))
            total_added += 1

            if total_added % 200 == 0:
                print(f"  Cached {total_added}/{max_samples}")

    return cache


class CachedStanfordDogs(Dataset):
    def __init__(self, img_size=32, max_samples=2000, cache_path=CACHE_PATH,
                 rebuild=False):
        self.img_size = img_size
        self.max_samples = max_samples

        if not rebuild and os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
            print(f"Loading cached tensors from {cache_path}...")
            with open(cache_path, "rb") as f:
                self.cache = pickle.load(f)
            print(f"Loaded {len(self.cache)} items")
        else:
            print(f"Precomputing {max_samples} tensors (one-time)...")
            self.cache = _build_cache(max_samples, img_size)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"Cached {len(self.cache)} items to {cache_path}")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


def train_cached():
    device = "cpu"
    img_size = 32
    batch_size = 16
    n_epochs = 50
    subset = 2000

    print(f"Loading {subset} images at {img_size}×{img_size}...")
    ds = CachedStanfordDogs(img_size=img_size, max_samples=subset, rebuild=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ConditionalUNet(img_size=img_size)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_loss = float("inf")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n = 0
        t0 = time.time()
        for cond, img in dl:
            cond, img = cond.to(device), img.to(device)
            x0 = torch.randn_like(img)
            t = torch.rand(img.shape[0], device=device)
            t_reshape = t.view(-1, 1, 1, 1)
            xt = (1 - t_reshape) * x0 + t_reshape * img
            target = img - x0

            pred = model(xt, cond, t)
            loss = torch.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * cond.shape[0]
            n += cond.shape[0]

        scheduler.step()
        avg_loss = total_loss / n
        elapsed = time.time() - t0

        ckpt_dir = os.path.join(_pkg_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"dogs_epoch_{epoch+1:02d}.pt"))

        marker = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "dogs_best.pt"))
            marker = " *"
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f} ({elapsed:.1f}s){marker}")

    print(f"\nDone! Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {os.path.join(_pkg_root, 'checkpoints')}")
    return best_loss


if __name__ == "__main__":
    train_cached()
