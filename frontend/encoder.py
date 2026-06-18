"""Encoder: Converts a layout scene into the output tensor format.

The output tensor is a fixed-size float tensor that a back-end diffusion
model consumes as conditioning. This is the "numerical contract" between
the front-end (pure code) and the back-end (neural model).

Tensor layout (total: 688 floats):
  [0:8]    — scene vector (background, lighting)
  [8:88]   — objects (5 slots × 16 dims)
  [88:688] — parts (5 objects × 10 parts × 12 dims)
"""

import torch
import math
import numpy as np
from . import schema
from . import knowledge_base as kb


# ── Normalization helpers ──────────────────────────────────────────────────
# Map wnid range to 0-1 for the tensor
# WordNet offsets go up to ~100,000,000, so divide by 1e8
_WNID_SCALE = 100_000_000.0


def _norm_wnid(wnid):
    return wnid / _WNID_SCALE


def _denorm_wnid(val):
    return int(round(val * _WNID_SCALE))


# Max action_id expected
_ACTION_MAX = 20
_PART_ID_MAX = 20


def encode_scene(scene_dict):
    """Encode scene dictionary into scene vector tensor."""
    sv = torch.zeros(schema.SCENE_DIM)
    sv[0] = scene_dict.get("bg_r", 0.53)
    sv[1] = scene_dict.get("bg_g", 0.81)
    sv[2] = scene_dict.get("bg_b", 0.92)
    sv[3] = scene_dict.get("scene_type", 0) / 10.0
    sv[4] = scene_dict.get("lighting", 0) / 10.0
    sv[5] = scene_dict.get("time_of_day", 0) / 10.0
    sv[6] = scene_dict.get("weather", 0) / 10.0
    return sv


def encode_object(obj_dict, idx):
    """Encode a single object dict into an OBJ_DIM tensor."""
    vec = torch.zeros(schema.OBJ_DIM)
    
    bbox_left = obj_dict.get("x", 0.5) - obj_dict.get("w", 0.25) / 2
    bbox_top = obj_dict.get("y", 0.5) - obj_dict.get("h", 0.25) / 2
    bbox_right = obj_dict.get("x", 0.5) + obj_dict.get("w", 0.25) / 2
    bbox_bottom = obj_dict.get("y", 0.5) + obj_dict.get("h", 0.25) / 2
    
    vec[0] = _norm_wnid(obj_dict.get("wnid", 0))
    vec[1] = obj_dict.get("x", 0.5)
    vec[2] = obj_dict.get("y", 0.5)
    vec[3] = obj_dict.get("w", 0.25)
    vec[4] = obj_dict.get("h", 0.25)
    vec[5] = obj_dict.get("depth", idx * 0.1)
    vec[6] = obj_dict.get("scale", 1) / 10.0
    vec[7] = obj_dict.get("rotation", 0.0)
    vec[8] = obj_dict.get("action_id", 0) / _ACTION_MAX
    vec[9] = obj_dict.get("confidence", 1.0)
    vec[10] = obj_dict.get("age", 0.5)
    vec[11] = obj_dict.get("size_category", 1) / 10.0
    vec[12] = bbox_left
    vec[13] = bbox_top
    vec[14] = bbox_right
    vec[15] = bbox_bottom
    
    return vec


def encode_part(part_dict):
    """Encode a single part dict into a PART_DIM tensor."""
    vec = torch.zeros(schema.PART_DIM)
    vec[0] = part_dict.get("part_id", 0) / _PART_ID_MAX
    vec[1] = part_dict.get("color_r", 0.6)
    vec[2] = part_dict.get("color_g", 0.4)
    vec[3] = part_dict.get("color_b", 0.2)
    vec[4] = part_dict.get("rel_x", 0.0)
    vec[5] = part_dict.get("rel_y", 0.0)
    vec[6] = part_dict.get("rel_w", 0.2)
    vec[7] = part_dict.get("rel_h", 0.2)
    vec[8] = part_dict.get("texture", 0) / 5.0
    vec[9] = part_dict.get("pattern", 0) / 3.0
    vec[10] = part_dict.get("opacity", 1.0)
    return vec


def encode_scene_to_tensor(layout_result):
    """Complete pipeline: layout dict → 688-dim tensor.
    
    Args:
        layout_result: dict from layout.layout()
    
    Returns:
        torch.Tensor of shape (total_dim,)
    """
    # Scene vector
    scene_vec = encode_scene(layout_result["scene"])
    
    # Object tensor: (MAX_OBJECTS, OBJ_DIM)
    objects_tensor = torch.zeros(schema.MAX_OBJECTS, schema.OBJ_DIM)
    for idx, obj in enumerate(layout_result["objects"]):
        if idx >= schema.MAX_OBJECTS:
            break
        objects_tensor[idx] = encode_object(obj, idx)
    
    # Parts tensor: (MAX_OBJECTS, MAX_PARTS, PART_DIM)
    parts_tensor = torch.zeros(schema.MAX_OBJECTS, schema.MAX_PARTS, schema.PART_DIM)
    for obj_idx, obj in enumerate(layout_result["objects"]):
        if obj_idx >= schema.MAX_OBJECTS:
            break
        for part_idx, part in enumerate(obj.get("parts", [])):
            if part_idx >= schema.MAX_PARTS:
                break
            parts_tensor[obj_idx, part_idx] = encode_part(part)
    
    return schema.encode_scene(scene_vec, objects_tensor, parts_tensor)


def prompt_to_tensor(prompt):
    """End-to-end: prompt string → 688-dim tensor.
    
    This is the main entry point. Feed the output to a back-end
    diffusion/flow model as conditioning.
    """
    from . import parser, layout as layout_mod
    
    parsed = parser.parse(prompt)
    layout_result = layout_mod.layout(parsed)
    tensor = encode_scene_to_tensor(layout_result)
    return tensor, layout_result


if __name__ == "__main__":
    test_prompts = [
        "a brown dog sitting under a tree",
        "a white cat sleeping",
        "a golden retriever puppy running in green grass",
    ]
    
    print("=== Encoder Test ===")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        tensor, layout_result = prompt_to_tensor(prompt)
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  Tensor dtype: {tensor.dtype}")
        print(f"  Tensor range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        print(f"  Objects: {len(layout_result['objects'])}")
        
        # Show first non-empty object's decoded values
        sv, objs, parts = schema.decode_scene(tensor)
        for oi in range(schema.MAX_OBJECTS):
            obj = objs[oi]
            if obj[0].item() < 0.01:
                continue
            wnid = _denorm_wnid(obj[0].item())
            name = kb.SYNSETS.get(wnid, {}).get("name", "unknown")
            print(f"  Object {oi}: {name} (wnid={wnid})")
            print(f"    position: ({obj[1]:.3f}, {obj[2]:.3f})")
            print(f"    action_id: {int(obj[8].item() * _ACTION_MAX)}")
        
        print(f"  Total dims: {tensor.shape[0]}")
    
    print("\nDone! The output tensor is ready to feed into a back-end diffusion model.")
