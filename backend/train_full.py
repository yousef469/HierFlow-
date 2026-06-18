"""Full 20K Stanford Dogs training at 64×64.

Usage:
  python3 backend/train_full.py          # cache + train
  python3 backend/train_full.py --resume  # resume from latest checkpoint
"""

import os, sys, time, pickle, glob
import numpy as np

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from backend.model import ConditionalUNet
from backend.stanford_dogs import StanfordDogsDataset as RawStanfordDogs
from frontend.encoder import prompt_to_tensor
from frontend import schema, knowledge_base as kb

CACHE_PATH = os.path.join(_pkg_root, "data", "stanford_dogs", "full_64_cache.pkl")


def cache_all(max_samples=20580, img_size=64):
    """Cache all Stanford Dogs at given resolution."""
    import numpy as np
    from backend.stanford_dogs import _STANFORD_BREED_OVERRIDES, IMAGES_DIR, ANNOT_DIR, _load_annotation

    breed_dirs = sorted(os.listdir(IMAGES_DIR))
    cache = []
    total = 0

    for breed_dir in breed_dirs:
        breed_path = os.path.join(IMAGES_DIR, breed_dir)
        if not os.path.isdir(breed_path):
            continue
        breed_id = breed_dir.split("-")[0] if "-" in breed_dir else breed_dir
        wnid = _STANFORD_BREED_OVERRIDES.get(breed_id, 2084071)
        breed_name = breed_dir.split("-", 1)[1] if "-" in breed_dir else breed_dir
        annot_breed_dir = os.path.join(ANNOT_DIR, breed_dir)
        if not os.path.exists(annot_breed_dir):
            annot_breed_dir = None

        images = sorted([f for f in os.listdir(breed_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for img_name in images:
            if total >= max_samples:
                break
            img_path = os.path.join(breed_path, img_name)
            syn = kb.SYNSETS.get(wnid, {})
            base = syn.get("name", "dog").replace("_", " ")
            color = np.random.choice(["brown", "black", "white", "golden", "gray", "tan"])
            action = np.random.choice(["standing", "sitting", "walking"])
            prompt = f"a {color} {base} {action}"

            tensor, layout = prompt_to_tensor(prompt)
            sv, objs, parts = schema.decode_scene(tensor)
            for oi in range(schema.MAX_OBJECTS):
                if objs[oi][0].item() > 0.01:
                    objs[oi][0] = wnid / 100_000_000.0
                    break

            bbox = None
            if annot_breed_dir is not None:
                annot_name = os.path.splitext(img_name)[0]
                ap = os.path.join(annot_breed_dir, annot_name)
                if os.path.exists(ap):
                    bbox = _load_annotation(ap)
            if bbox is not None:
                cx, cy, bw, bh = bbox
                for oi in range(schema.MAX_OBJECTS):
                    if objs[oi][0].item() > 0.01:
                        name = kb.SYNSETS.get(int(objs[oi][0].item() * 100_000_000), {}).get("name", "")
                        if name not in ("sky", "ground", "grass", "tree", "water", "cloud", "mountain", "sun"):
                            objs[oi][1] = max(0.0, min(1.0, cx))
                            objs[oi][2] = max(0.0, min(1.0, cy))
                            objs[oi][3] = max(0.05, min(1.0, bw))
                            objs[oi][4] = max(0.05, min(1.0, bh))
                            break
            tensor = schema.encode_scene(sv, objs, parts)

            img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
            cache.append((tensor.float(), img_t))
            total += 1
            if total % 500 == 0:
                print(f"  Cached {total}/{max_samples}")
    return cache


class FullDogs(Dataset):
    def __init__(self, img_size=64, max_samples=20580, rebuild=False):
        self.img_size = img_size
        exists = os.path.exists(CACHE_PATH) and os.path.getsize(CACHE_PATH) > 100
        if not rebuild and exists:
            print(f"Loading cache from {CACHE_PATH}...")
            with open(CACHE_PATH, "rb") as f:
                self.cache = pickle.load(f)
        else:
            print(f"Building cache of {max_samples} images at {img_size}×{img_size}...")
            self.cache = cache_all(max_samples, img_size)
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(self.cache, f)
        print(f"Dataset: {len(self.cache)} items")

    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        return self.cache[idx]


def train_full(resume=False):
    device = "cpu"
    img_size = 64
    batch_size = 8
    n_epochs = 50

    print(f"Loading 20K dataset at {img_size}×{img_size}...")
    ds = FullDogs(img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ConditionalUNet(img_size=img_size)
    ckpt_dir = os.path.join(_pkg_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    if resume:
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "dogs64_epoch_*.pt")))
        if ckpts:
            last = ckpts[-1]
            start_epoch = int(last.split("_epoch_")[1].split(".pt")[0])
            model.load_state_dict(torch.load(last))
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("No checkpoints found, starting from scratch")
    else:
        sp = os.path.join(_pkg_root, "synthetic_pretrain.pt")
        if os.path.exists(sp):
            model.load_state_dict(torch.load(sp))
            print(f"Starting from synthetic pretrain ({sp})")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_loss = float("inf")
    for epoch in range(start_epoch, n_epochs):
        model.train()
        total_loss, n = 0, 0
        t0 = time.time()
        for cond, img in dl:
            cond, img = cond.to(device), img.to(device)
            x0 = torch.randn_like(img)
            t = torch.rand(img.shape[0], device=device)
            xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * img
            pred = model(xt, cond, t)
            loss = torch.nn.functional.mse_loss(pred, img - x0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * cond.shape[0]
            n += cond.shape[0]
        scheduler.step()
        avg_loss = total_loss / n
        elapsed = time.time() - t0

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"dogs64_epoch_{epoch+1:02d}.pt"))
        is_best = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "dogs64_best.pt"))
            is_best = " *"
        eta = elapsed * (n_epochs - epoch - 1) / 3600
        print(f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f} ({elapsed:.0f}s, ETA {eta:.1f}h){is_best}")

    print(f"\nDone! Best loss: {best_loss:.6f}")
    return best_loss


if __name__ == "__main__":
    import sys
    resume = "--resume" in sys.argv
    train_full(resume=resume)
