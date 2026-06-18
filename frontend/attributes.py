"""Attribute Database: colors, textures, patterns, actions, and scenes.

Maps natural language descriptions to numerical encodings
that the output tensor schema can consume.
"""

import math

# ── Colors: name → normalized RGB ────────────────────────────────────────
COLORS = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "dark gray": (0.3, 0.3, 0.3),
    "light gray": (0.7, 0.7, 0.7),
    "red": (1.0, 0.0, 0.0),
    "dark red": (0.5, 0.0, 0.0),
    "orange": (1.0, 0.65, 0.0),
    "dark orange": (0.8, 0.4, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "gold": (1.0, 0.84, 0.0),
    "green": (0.0, 1.0, 0.0),
    "dark green": (0.0, 0.4, 0.0),
    "light green": (0.56, 0.93, 0.56),
    "lime": (0.75, 1.0, 0.0),
    "olive": (0.5, 0.5, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "dark blue": (0.0, 0.0, 0.5),
    "light blue": (0.68, 0.85, 0.9),
    "sky blue": (0.53, 0.81, 0.92),
    "navy": (0.0, 0.0, 0.5),
    "cyan": (0.0, 1.0, 1.0),
    "teal": (0.0, 0.5, 0.5),
    "purple": (0.5, 0.0, 0.5),
    "violet": (0.93, 0.51, 0.93),
    "pink": (1.0, 0.71, 0.76),
    "hot pink": (1.0, 0.41, 0.71),
    "magenta": (1.0, 0.0, 1.0),
    "brown": (0.55, 0.27, 0.07),
    "dark brown": (0.4, 0.2, 0.0),
    "light brown": (0.71, 0.56, 0.38),
    "tan": (0.82, 0.71, 0.55),
    "beige": (0.96, 0.96, 0.86),
    "cream": (1.0, 0.99, 0.82),
    "chocolate": (0.48, 0.25, 0.0),
    "rust": (0.7, 0.23, 0.12),
    "coral": (1.0, 0.5, 0.31),
    "salmon": (0.98, 0.5, 0.45),
    "copper": (0.72, 0.45, 0.2),
    "bronze": (0.8, 0.5, 0.2),
    "silver": (0.75, 0.75, 0.75),
    "rainbow": (1.0, 0.0, 0.0),  # placeholder
    "transparent": (0.0, 0.0, 0.0),
    "clear": (0.0, 0.0, 0.0),
}


# ── Color synonyms ────────────────────────────────────────────────────────
_COLOR_SYNONYMS = {
    "blonde": "yellow",
    "ginger": "orange",
    "chestnut": "brown",
    "auburn": "rust",
    "snowy": "white",
    "snow": "white",
    "midnight": "dark blue",
    "emerald": "dark green",
    "ruby": "red",
    "sapphire": "blue",
    "ivory": "cream",
    "ash": "gray",
    "charcoal": "dark gray",
    "pearl": "white",
    "jet": "black",
    "crimson": "red",
    "scarlet": "red",
    "maroon": "dark red",
    "lavender": "purple",
    "lilac": "purple",
    "mint": "light green",
    "forest": "dark green",
    "sky": "sky blue",
    "ocean": "blue",
    "sand": "tan",
    "dusty": "gray",
    "pale": "light gray",
}


def resolve_color(name):
    """Return normalized RGB tuple for a color name, or None."""
    key = name.lower().strip()
    if key in COLORS:
        return COLORS[key]
    if key in _COLOR_SYNONYMS:
        return COLORS[_COLOR_SYNONYMS[key]]
    # Try "light X" / "dark X" patterns
    for prefix in ["light ", "pale ", "bright "]:
        if key.startswith(prefix):
            base = key[len(prefix):]
            if base in COLORS:
                r, g, b = COLORS[base]
                return (min(1.0, r + 0.3), min(1.0, g + 0.3), min(1.0, b + 0.3))
    for prefix in ["dark ", "deep ", "rich "]:
        if key.startswith(prefix):
            base = key[len(prefix):]
            if base in COLORS:
                r, g, b = COLORS[base]
                return (r * 0.5, g * 0.5, b * 0.5)
    return None


# ── Textures ──────────────────────────────────────────────────────────────
TEXTURES = {
    "fur": 0,
    "furry": 0,
    "fluffy": 0,
    "feather": 1,
    "feathery": 1,
    "feathered": 1,
    "scale": 2,
    "scaly": 2,
    "scales": 2,
    "skin": 3,
    "smooth": 4,
    "slick": 4,
    "shiny": 4,
    "rough": 5,
    "bumpy": 5,
    "wool": 0,
    "wooly": 0,
    "fuzzy": 0,
    "spiky": 5,
    "slimy": 3,
    "leathery": 3,
    "feathers": 1,
}


def resolve_texture(name):
    """Return texture_id or 0 (fur default)."""
    key = name.lower().strip()
    return TEXTURES.get(key, 0)


# ── Patterns ──────────────────────────────────────────────────────────────
PATTERNS = {
    "solid": 0,
    "striped": 1,
    "stripes": 1,
    "spotted": 2,
    "spots": 2,
    "patchy": 3,
    "patches": 3,
    "calico": 3,
    "tabby": 1,
    "brindle": 1,
    "dappled": 2,
    "mottled": 3,
    "splotchy": 3,
    "camouflage": 3,
    "camouflaged": 3,
    "two_tone": 3,
    "tuxedo": 3,
}


def resolve_pattern(name):
    """Return pattern_id or 0 (solid default)."""
    key = name.lower().strip()
    return PATTERNS.get(key, 0)


# ── Actions/Poses ─────────────────────────────────────────────────────────
ACTIONS = {
    "standing": 0,
    "stand": 0,
    "stands": 0,
    "sitting": 1,
    "sit": 1,
    "sits": 1,
    "running": 2,
    "run": 2,
    "runs": 2,
    "flying": 3,
    "fly": 3,
    "flies": 3,
    "soaring": 3,
    "soar": 3,
    "swimming": 4,
    "swim": 4,
    "swims": 4,
    "floating": 5,
    "float": 5,
    "floats": 5,
    "sleeping": 6,
    "sleep": 6,
    "sleeps": 6,
    "napping": 6,
    "laying": 6,
    "lying": 6,
    "laying down": 6,
    "running": 2,
    "jumping": 7,
    "jump": 7,
    "jumps": 7,
    "hopping": 7,
    "hop": 7,
    "hops": 7,
    "eating": 8,
    "eat": 8,
    "eats": 8,
    "drinking": 8,
    "drink": 8,
    "drinks": 8,
    "barking": 0,
    "bark": 0,
    "howling": 0,
    "howl": 0,
    "chasing": 2,
    "walking": 0,
    "walk": 0,
    "walks": 0,
    "crawling": 7,
    "crawl": 7,
    "climbing": 7,
    "climb": 7,
    "wagging": 0,
    "wag": 0,
}


def resolve_action(name):
    """Return action_id or 0 (standing default)."""
    key = name.lower().strip()
    return ACTIONS.get(key, 0)


# ── Scene types ───────────────────────────────────────────────────────────
SCENE_TYPES = {
    "outdoor": 0, "outside": 0, "outdoors": 0,
    "nature": 0, "wild": 0, "field": 0, "forest": 0,
    "indoor": 1, "inside": 1, "indoors": 1, "house": 1, "room": 1,
    "abstract": 2, "studio": 2, "plain": 2,
}


def resolve_scene_type(name):
    """Return scene_type_id or 0 (outdoor default)."""
    key = name.lower().strip()
    for k, v in SCENE_TYPES.items():
        if k in key or key in k:
            return v
    return 0


# ── Lighting ──────────────────────────────────────────────────────────────
LIGHTING = {
    "sunny": 0, "sun": 0, "bright": 0, "daylight": 0, "day": 0,
    "cloudy": 1, "overcast": 1, "gray": 1, "grey": 1,
    "dark": 2, "night": 2, "moonlight": 2,
    "artificial": 3, "indoor lighting": 3, "indoor": 3, "lamp": 3,
}


def resolve_lighting(name):
    """Return lighting_id or 0 (sunny default)."""
    key = name.lower().strip()
    for k, v in LIGHTING.items():
        if k in key or key in k:
            return v
    return 0


# ── Time of day ───────────────────────────────────────────────────────────
TIMES = {
    "day": 0, "daytime": 0, "noon": 0, "morning": 0, "afternoon": 0,
    "dawn": 1, "dusk": 1, "sunset": 1, "sunrise": 1, "twilight": 1,
    "night": 2, "midnight": 2, "evening": 2,
}


def resolve_time(name):
    """Return time_id or 0 (day default)."""
    key = name.lower().strip()
    for k, v in TIMES.items():
        if k in key or key in k:
            return v
    return 0


# ── Weather ───────────────────────────────────────────────────────────────
WEATHERS = {
    "clear": 0, "sunny": 0, "fair": 0,
    "rain": 1, "rainy": 1, "raining": 1, "storm": 1, "stormy": 1,
    "snow": 2, "snowy": 2, "snowing": 2,
    "fog": 3, "foggy": 3, "mist": 3, "misty": 3, "hazy": 3,
}


def resolve_weather(name):
    """Return weather_id or 0 (clear default)."""
    key = name.lower().strip()
    for k, v in WEATHERS.items():
        if k in key or key in k:
            return v
    return 0


# ── Size descriptions ────────────────────────────────────────────────────
SIZE_WORDS = {
    "tiny": 0, "small": 0, "little": 0, "baby": 0, "mini": 0,
    "medium": 1, "mid": 1, "medium-sized": 1,
    "big": 2, "large": 2, "huge": 2, "giant": 2, "massive": 2, "enormous": 2,
}


def resolve_size(name):
    """Return size_category or None."""
    key = name.lower().strip()
    return SIZE_WORDS.get(key)


if __name__ == "__main__":
    # Test attribute resolution
    test_colors = ["red", "dark blue", "sky blue", "blonde", "emerald"]
    for c in test_colors:
        rgb = resolve_color(c)
        print(f"  {c} → {rgb}")
    
    test_actions = ["running", "flying", "swimming", "sleeping", "jumping"]
    for a in test_actions:
        aid = resolve_action(a)
        print(f"  {a} → action_id={aid}")
    
    print(f"\n  Total colors: {len(COLORS)}")
    print(f"  Total actions: {len(ACTIONS)}")
