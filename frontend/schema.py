"""Output Tensor Schema: The numerical contract between front-end and back-end.

The front-end (code) outputs a fixed-size float tensor that the back-end
(diffusion / flow model) consumes as conditioning, replacing raw text tokens.

Layout:
  scene_vec:  [SCENE_DIM]             — background, lighting
  objects:    [MAX_OBJECTS, OBJ_DIM]  — what objects exist, where
  parts:      [MAX_OBJECTS, MAX_PARTS, PART_DIM]  — parts + attributes

Total dims: 8 + (5 * 16) + (5 * 10 * 12) = 8 + 80 + 600 = 688
"""

import torch
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
MAX_OBJECTS = 5
MAX_PARTS = 10

# Scene-level features
SCENE_DIM = 8
SCENE_KEYS = [
    "bg_r", "bg_g", "bg_b",
    "scene_type",       # 0=outdoor, 1=indoor, 2=abstract
    "lighting",         # 0=sunny, 1=cloudy, 2=dark, 3=artificial
    "time_of_day",      # 0=day, 1=dawn/dusk, 2=night
    "weather",          # 0=clear, 1=rain, 2=snow, 3=fog
]

# Object-level features
OBJ_DIM = 16
OBJ_KEYS = [
    "wnid",             # WordNet synset offset (int, scaled to 0-1)
    "x", "y",           # center position (normalized 0-1)
    "w", "h",           # size (normalized 0-1)
    "depth",            # z-order (0=front, 1=back)
    "scale",            # relative size (0.1-1.0)
    "rotation",         # 0-1 mapping to 0-360 degrees
    "action_id",        # 0=standing, 1=sitting, 2=running, 3=flying, 4=swimming
    "confidence",       # how sure we are this object exists (0-1)
    "age",              # 0=baby, 0.5=adult, 1=elderly (for animals)
    "size_category",    # 0=small, 1=medium, 2=large
    "left",             # bounding box left
    "top",              # bounding box top
    "right",            # bounding box right
    "bottom",           # bounding box bottom
]

# Part-level features
PART_DIM = 12
PART_KEYS = [
    "part_id",          # 0=head, 1=torso, 2=tail, 3=leg_lf, 4=leg_rf,
                        # 5=leg_lb, 6=leg_rb, 7=wing, 8=ear, 9=eye,
                        # 10=nose, 11=mouth, 12=horn, 13=fin
    "color_r", "color_g", "color_b",  # RGB normalized 0-1
    "rel_x", "rel_y",   # position relative to object center (-1 to 1)
    "rel_w", "rel_h",   # size relative to object (0-1)
    "texture_id",       # 0=fur, 1=feather, 2=scale, 3=skin, 4=smooth, 5=rough
    "pattern_id",       # 0=solid, 1=striped, 2=spotted, 3=patchy
    "opacity",          # 0-1
]


# ── Schema properties ──────────────────────────────────────────────────────
def scene_dim():
    return SCENE_DIM

def obj_dim():
    return OBJ_DIM

def part_dim():
    return PART_DIM

def total_dim():
    return SCENE_DIM + (MAX_OBJECTS * OBJ_DIM) + (MAX_OBJECTS * MAX_PARTS * PART_DIM)


def make_empty_scene_tensor():
    """Return a zero-filled scene tensor of shape (total_dim,)."""
    return torch.zeros(total_dim())


def default_scene_vec(scene_type=0):
    """Create a default scene vector with reasonable defaults."""
    sv = torch.zeros(SCENE_DIM)
    sv[0:3] = torch.tensor([0.53, 0.81, 0.92])  # sky blue background
    sv[3] = scene_type  # outdoor
    sv[4] = 0  # sunny
    sv[5] = 0  # day
    sv[6] = 0  # clear
    return sv


def encode_scene(scene_vec, objects, parts):
    """Encode a complete scene into a flat tensor.
    
    Args:
        scene_vec: Tensor of shape (SCENE_DIM,)
        objects: Tensor of shape (MAX_OBJECTS, OBJ_DIM)
        parts: Tensor of shape (MAX_OBJECTS, MAX_PARTS, PART_DIM)
    
    Returns:
        Tensor of shape (total_dim,)
    """
    t = torch.cat([
        scene_vec,
        objects.flatten(),
        parts.flatten(),
    ])
    return t


def decode_scene(tensor):
    """Reverse encode_scene.
    
    Returns:
        scene_vec: (SCENE_DIM,)
        objects: (MAX_OBJECTS, OBJ_DIM)
        parts: (MAX_OBJECTS, MAX_PARTS, PART_DIM)
    """
    offset_s = SCENE_DIM
    offset_o = offset_s + (MAX_OBJECTS * OBJ_DIM)
    
    scene_vec = tensor[:offset_s]
    objects = tensor[offset_s:offset_o].reshape(MAX_OBJECTS, OBJ_DIM)
    parts = tensor[offset_o:].reshape(MAX_OBJECTS, MAX_PARTS, PART_DIM)
    
    return scene_vec, objects, parts


def print_scene(tensor):
    """Pretty-print a scene tensor for debugging."""
    sv, objs, parts = decode_scene(tensor)
    
    print("── Scene ──")
    for i, key in enumerate(SCENE_KEYS):
        print(f"  {key}: {sv[i].item():.4f}")
    
    for oi in range(MAX_OBJECTS):
        obj = objs[oi]
        if obj[0].item() < 0.5:  # no object in this slot
            continue
        wnid_val = int(obj[0].item() * 1000000)
        print(f"\n── Object {oi}: wnid={wnid_val} ──")
        for j, key in enumerate(OBJ_KEYS):
            print(f"  {key}: {obj[j].item():.4f}")
        
        for pi in range(MAX_PARTS):
            part = parts[oi, pi]
            if part[0].item() < 0.5:  # no part in this slot
                continue
            pid = int(part[0].item() * 20)
            print(f"  Part {pi}: id={pid}")
            for j, key in enumerate(PART_KEYS):
                print(f"    {key}: {part[j].item():.4f}")


if __name__ == "__main__":
    print(f"Total tensor dims: {total_dim()}")
    t = make_empty_scene_tensor()
    print(f"Tensor shape: {t.shape}")
    print_scene(t)
