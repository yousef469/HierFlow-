"""Layout Engine: Assigns spatial positions and bboxes to scene objects."""

import random
from . import knowledge_base as kb


_BG_ELEMENTS = {
    "sky": {"y": 0.15, "h": 0.45, "full_width": True},
    "cloud": {"y": 0.15, "h": 0.15, "w": 0.25, "full_width": False, "spread": "random_top"},
    "ground": {"y": 0.85, "h": 0.3, "full_width": True},
    "grass": {"y": 0.85, "h": 0.3, "full_width": True},
    "tree": {"y": 0.55, "h": 0.5, "w": 0.15, "full_width": False, "spread": "sides"},
    "mountain": {"y": 0.4, "h": 0.5, "w": 0.4, "full_width": False, "spread": "background"},
    "water": {"y": 0.8, "h": 0.35, "full_width": True},
    "sun": {"y": 0.1, "h": 0.1, "w": 0.08, "full_width": False, "spread": "top_corner"},
}

_PART_POS = {
    0: (0.0, -0.35, 0.4, 0.35), 1: (0.0, 0.05, 0.6, 0.5),
    2: (0.3, 0.0, 0.3, 0.15), 3: (-0.15, 0.3, 0.12, 0.25),
    4: (0.15, 0.3, 0.12, 0.25), 5: (-0.15, 0.25, 0.12, 0.25),
    6: (0.15, 0.25, 0.12, 0.25), 7: (0.0, -0.1, 0.8, 0.3),
    8: (0.0, -0.35, 0.15, 0.1), 9: (0.0, -0.35, 0.08, 0.08),
    10: (0.0, -0.3, 0.1, 0.08), 11: (0.0, -0.28, 0.12, 0.08),
    12: (0.0, -0.4, 0.1, 0.15), 13: (0.0, 0.05, 0.3, 0.15),
    14: (-0.15, 0.35, 0.05, 0.05), 15: (0.0, -0.35, 0.12, 0.08),
    16: (0.0, -0.3, 0.15, 0.05), 17: (0.0, -0.1, 0.3, 0.15),
    18: (-0.25, -0.15, 0.25, 0.2), 19: (0.0, 0.0, 0.6, 0.5),
}

_PART_COLORS = {
    9: (1.0, 1.0, 1.0), 10: (0.3, 0.15, 0.1), 11: (0.5, 0.25, 0.2),
    14: (0.6, 0.6, 0.6), 15: (0.9, 0.7, 0.2), 16: (0.8, 0.8, 0.8),
    17: (0.6, 0.3, 0.1), 19: (0.6, 0.5, 0.0),
}

_DEFAULT_COLORS = {
    "polar_bear": (1.0, 1.0, 1.0), "brown_bear": (0.55, 0.27, 0.07),
    "golden_retriever": (1.0, 0.84, 0.5), "husky": (0.7, 0.7, 0.7),
    "goldfish": (1.0, 0.6, 0.0), "clownfish": (1.0, 0.4, 0.0),
    "elephant": (0.5, 0.5, 0.5), "lion": (0.85, 0.65, 0.13),
    "tiger": (1.0, 0.6, 0.0), "penguin": (0.1, 0.1, 0.2),
    "parrot": (0.0, 1.0, 0.0), "eagle": (0.4, 0.3, 0.2),
    "giraffe": (1.0, 0.8, 0.4), "zebra": (1.0, 1.0, 1.0),
    "duck": (1.0, 0.8, 0.2), "snake": (0.2, 0.5, 0.2),
    "canine": (0.6, 0.4, 0.2), "feline": (0.6, 0.4, 0.2),
    "tree": (0.13, 0.55, 0.13), "grass": (0.2, 0.7, 0.2),
    "water": (0.0, 0.3, 0.6), "sky": (0.53, 0.81, 0.92),
    "mountain": (0.4, 0.3, 0.2), "flower": (1.0, 0.0, 0.5),
    "cloud": (1.0, 1.0, 1.0),
}


def _name(wnid):
    syn = kb.SYNSETS.get(wnid, {})
    return syn.get("name", "unknown")


def _get_layout(wnid):
    name = _name(wnid)
    if name in _BG_ELEMENTS:
        return _BG_ELEMENTS[name]
    habitat = kb.get_habitat(wnid)
    if habitat == 1:
        return {"y": 0.3, "h": 0.2, "w": 0.2, "full_width": False, "spread": "random_top"}
    elif habitat == 2:
        return {"y": 0.65, "h": 0.15, "w": 0.2, "full_width": False, "spread": "center"}
    return {"y": 0.65, "h": 0.3, "w": 0.25, "full_width": False, "spread": "bottom_center"}


def _assign_pos(layout, idx, total):
    w = layout.get("w", 0.25)
    h = layout.get("h", 0.25)
    if layout.get("full_width"):
        return (0.5, layout["y"], 1.0, layout["h"])
    spread = layout.get("spread", "center")
    if spread in ("bottom_center", "center"):
        x = 0.2 + idx * (0.6 / max(total, 1)) if total > 1 else 0.5
        y = layout["y"] + random.uniform(-0.03, 0.03)
    elif spread == "random_top":
        x, y = random.uniform(0.1, 0.9), layout["y"] + random.uniform(-0.03, 0.03)
    elif spread == "sides":
        x, y = 0.15 + idx * 0.7, layout["y"]
    else:
        x, y = 0.5, layout["y"]
    return (x, y, w, h)


def assign_parts(wnid, obj_color):
    pids = kb.get_inherited_parts(wnid)
    results = []
    for pid in pids:
        rx, ry, rw, rh = _PART_POS.get(pid, (0.0, 0.0, 0.2, 0.2))
        color = _PART_COLORS.get(pid, obj_color or (0.6, 0.4, 0.2))
        tex = 1 if pid == 7 else (2 if pid in (13, 19) else 0)
        results.append({
            "part_id": pid, "color_r": color[0], "color_g": color[1], "color_b": color[2],
            "rel_x": rx, "rel_y": ry, "rel_w": rw, "rel_h": rh,
            "texture": tex, "opacity": 1.0,
        })
    return results


def layout(parsed):
    entities = parsed["entities"]
    scene = parsed["scene"]
    relationships = parsed["relationships"]

    object_layouts = []
    for idx, ent in enumerate(entities):
        default = _get_layout(ent["wnid"])
        x, y, w, h = _assign_pos(default, idx, len(entities))
        color = ent.get("color")
        if color is None:
            color = _DEFAULT_COLORS.get(_name(ent["wnid"]), (0.6, 0.4, 0.2))
        action = ent.get("action", 0) or 0
        size = ent.get("size", 1) or 1
        object_layouts.append({
            "wnid": ent["wnid"], "x": x, "y": y, "w": w, "h": h,
            "depth": idx * 0.1, "scale": size, "rotation": 0.0,
            "action_id": action, "confidence": 1.0, "age": 0.5,
            "size_category": size, "color": color,
            "parts": assign_parts(ent["wnid"], color),
        })

    for rel in relationships:
        subj = next((o for o in object_layouts if o["wnid"] == rel["subject"]), None)
        obj = next((o for o in object_layouts if o["wnid"] == rel["object"]), None)
        if subj and obj and rel["type"] == "above":
            subj["y"] = obj["y"] - obj["h"] / 2 - subj["h"] / 2 - 0.05

    bg = scene.get("bg_color")
    bg = (0.53, 0.81, 0.92) if (bg is None or not isinstance(bg, (tuple, list))) else bg
    bg_r, bg_g, bg_b = bg

    has_ground = any(_name(o["wnid"]) in ("ground", "grass") for o in object_layouts)
    has_sky = any(_name(o["wnid"]) == "sky" for o in object_layouts)

    bg_objects = []
    if not has_sky:
        bg_objects.append({
            "wnid": kb.resolve_name("sky"), "x": 0.5, "y": 0.15, "w": 1.0, "h": 0.45,
            "depth": 1.0, "scale": 2, "rotation": 0.0, "action_id": 0,
            "confidence": 0.8, "age": 0.5, "size_category": 2,
            "color": (bg_r, bg_g, bg_b), "parts": [],
        })
    if not has_ground:
        gc = (0.3, 0.6, 0.15) if scene.get("scene_type", 0) else (0.7, 0.7, 0.7)
        bg_objects.append({
            "wnid": kb.resolve_name("ground"), "x": 0.5, "y": 0.85, "w": 1.0, "h": 0.3,
            "depth": 1.0, "scale": 2, "rotation": 0.0, "action_id": 0,
            "confidence": 0.8, "age": 0.5, "size_category": 2,
            "color": gc, "parts": [],
        })

    return {
        "objects": bg_objects + object_layouts,
        "scene": {
            "bg_r": bg_r, "bg_g": bg_g, "bg_b": bg_b,
            "scene_type": scene.get("scene_type", 0) or 0,
            "lighting": scene.get("lighting", 0) or 0,
            "time_of_day": scene.get("time_of_day", 0) or 0,
            "weather": scene.get("weather", 0) or 0,
        }
    }


if __name__ == "__main__":
    from . import parser
    for p in ["a brown dog sitting under a tree", "a white cat sleeping"]:
        parsed = parser.parse(p)
        r = layout(parsed)
        print(f"Prompt: {p}")
        for o in r["objects"]:
            c = o.get("color", (0,0,0))
            print(f"  {_name(o['wnid'])}: ({o['x']:.2f},{o['y']:.2f}) parts={len(o.get('parts',[]))}")
