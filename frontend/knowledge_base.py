"""Knowledge Base: WordNet-style hierarchical ontology for animals and objects.

Each concept has:
  - wnid: WordNet synset offset (unique 8-digit ID)
  - name: common English name
  - hypernym: parent concept's wnid
  - parts: list of part IDs this concept typically has
  - size_category: 0=small, 1=medium, 2=large
  - habitat: 0=land, 1=air, 2=water, 3=both

Part IDs:
  0=head, 1=torso, 2=tail, 3=leg_lf, 4=leg_rf,
  5=leg_lb, 6=leg_rb, 7=wing, 8=ear, 9=eye,
  10=nose, 11=mouth, 12=horn, 13=fin, 14=claw,
  15=beak, 16=whisker, 17=mane, 18=trunk, 19=shell
"""

# ── WordNet-style synset IDs (real WordNet offsets) ────────────────────────
# These are real WordNet 3.0 / 3.1 offsets

SYNSETS = {
    # Root
    0: {
        "name": "entity",
        "hypernym": None,
        "parts": [],
        "size": 1,
        "habitat": 0,
    },
    1000000: {
        "name": "living_thing",
        "hypernym": 0,
        "parts": [],
        "size": 1,
        "habitat": 0,
    },
    2000000: {
        "name": "animal",
        "hypernym": 1000000,
        "parts": [0, 1, 9, 11],  # head, torso, eye, mouth
        "size": 1,
        "habitat": 0,
    },

    # Mammals (WordNet: mammal 02484310)
    2484310: {
        "name": "mammal",
        "hypernym": 2000000,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 11],
        "size": 1,
        "habitat": 0,
    },

    # Dog and breeds (WordNet: dog 02084071, canine 02083028)
    2084071: {
        "name": "dog",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 1,
        "habitat": 0,
    },
    2084072: {
        "name": "husky",
        "hypernym": 2084071,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 1,
        "habitat": 0,
    },
    2084073: {
        "name": "poodle",
        "hypernym": 2084071,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 1,
        "habitat": 0,
    },
    2084074: {
        "name": "golden_retriever",
        "hypernym": 2084071,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 2,
        "habitat": 0,
    },
    2084075: {
        "name": "bulldog",
        "hypernym": 2084071,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 1,
        "habitat": 0,
    },
    2084076: {
        "name": "pug",
        "hypernym": 2084071,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 0,
        "habitat": 0,
    },
    2084077: {
        "name": "puppy",
        "hypernym": 2084071,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 0,
        "habitat": 0,
    },

    # Cat and breeds (WordNet: cat 02121808)
    2121808: {
        "name": "cat",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 1,
        "habitat": 0,
    },
    2121809: {
        "name": "kitten",
        "hypernym": 2121808,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 0,
        "habitat": 0,
    },
    2121810: {
        "name": "siamese_cat",
        "hypernym": 2121808,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 1,
        "habitat": 0,
    },
    2121811: {
        "name": "persian_cat",
        "hypernym": 2121808,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 1,
        "habitat": 0,
    },

    # Horse (WordNet: horse 02374451)
    2374451: {
        "name": "horse",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 17],
        "size": 2,
        "habitat": 0,
    },
    2374452: {
        "name": "foal",
        "hypernym": 2374451,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 17],
        "size": 1,
        "habitat": 0,
    },

    # Cow / Cattle (WordNet: bovine 02436123)
    2436123: {
        "name": "cow",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        "size": 2,
        "habitat": 0,
    },
    2436124: {
        "name": "bull",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        "size": 2,
        "habitat": 0,
    },
    2436125: {
        "name": "calf",
        "hypernym": 2436123,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        "size": 1,
        "habitat": 0,
    },

    # Sheep (WordNet: sheep 02411705)
    2411705: {
        "name": "sheep",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 1,
        "habitat": 0,
    },
    2411706: {
        "name": "lamb",
        "hypernym": 2411705,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 0,
        "habitat": 0,
    },

    # Pig (WordNet: pig 02390527)
    2390527: {
        "name": "pig",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 1,
        "habitat": 0,
    },

    # Goat (WordNet: goat 02419390)
    2419390: {
        "name": "goat",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        "size": 1,
        "habitat": 0,
    },

    # Deer (WordNet: deer 02431208)
    2431208: {
        "name": "deer",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        "size": 1,
        "habitat": 0,
    },
    2431209: {
        "name": "fawn",
        "hypernym": 2431208,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        "size": 0,
        "habitat": 0,
    },

    # Rabbit (WordNet: rabbit 02325336)
    2325336: {
        "name": "rabbit",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 0,
        "habitat": 0,
    },
    2325337: {
        "name": "bunny",
        "hypernym": 2325336,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 0,
        "habitat": 0,
    },

    # Bear (WordNet: bear 02132236)
    2132236: {
        "name": "bear",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 2,
        "habitat": 0,
    },
    2132237: {
        "name": "polar_bear",
        "hypernym": 2132236,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 2,
        "habitat": 0,
    },
    2132238: {
        "name": "brown_bear",
        "hypernym": 2132236,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 2,
        "habitat": 0,
    },

    # Elephant (WordNet: elephant 02507101)
    2507101: {
        "name": "elephant",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 18],
        "size": 2,
        "habitat": 0,
    },

    # Lion / Big cats (WordNet: lion 02128938, tiger 02129523)
    2128938: {
        "name": "lion",
        "hypernym": 2121808,  # actually felid, but simplified to cat
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 17],
        "size": 2,
        "habitat": 0,
    },
    2129523: {
        "name": "tiger",
        "hypernym": 2121808,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 2,
        "habitat": 0,
    },

    # Fox (WordNet: fox 02134098)
    2134098: {
        "name": "fox",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 1,
        "habitat": 0,
    },

    # Squirrel (WordNet: squirrel 02356922)
    2356922: {
        "name": "squirrel",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        "size": 0,
        "habitat": 0,
    },

    # Mouse / Rodent (WordNet: mouse 02329229)
    2329229: {
        "name": "mouse",
        "hypernym": 2484310,
        "parts": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16],
        "size": 0,
        "habitat": 0,
    },

    # Birds (WordNet: bird 01503061)
    1503061: {
        "name": "bird",
        "hypernym": 2000000,
        "parts": [0, 1, 2, 7, 9, 11],
        "size": 0,
        "habitat": 1,
    },
    1503062: {
        "name": "eagle",
        "hypernym": 1503061,
        "parts": [0, 1, 2, 7, 9, 11, 15],
        "size": 1,
        "habitat": 1,
    },
    1503063: {
        "name": "hawk",
        "hypernym": 1503061,
        "parts": [0, 1, 2, 7, 9, 11, 15],
        "size": 1,
        "habitat": 1,
    },
    1503064: {
        "name": "parrot",
        "hypernym": 1503061,
        "parts": [0, 1, 2, 7, 9, 11, 15],
        "size": 0,
        "habitat": 1,
    },
    1503065: {
        "name": "pigeon",
        "hypernym": 1503061,
        "parts": [0, 1, 2, 7, 9, 11, 15],
        "size": 0,
        "habitat": 1,
    },
    1503066: {
        "name": "chicken",
        "hypernym": 1503061,
        "parts": [0, 1, 2, 7, 9, 11, 15],
        "size": 0,
        "habitat": 0,
    },
    1503067: {
        "name": "duck",
        "hypernym": 1503061,
        "parts": [0, 1, 2, 7, 9, 11, 15],
        "size": 0,
        "habitat": 3,
    },
    1503068: {
        "name": "owl",
        "hypernym": 1503061,
        "parts": [0, 1, 2, 7, 9, 11, 15],
        "size": 0,
        "habitat": 1,
    },
    1503069: {
        "name": "penguin",
        "hypernym": 1503061,
        "parts": [0, 1, 2, 7, 9, 11],
        "size": 1,
        "habitat": 3,
    },

    # Fish (WordNet: fish 01471728)
    1471728: {
        "name": "fish",
        "hypernym": 2000000,
        "parts": [0, 1, 2, 13, 9, 11],
        "size": 0,
        "habitat": 2,
    },
    1471729: {
        "name": "goldfish",
        "hypernym": 1471728,
        "parts": [0, 1, 2, 13, 9, 11],
        "size": 0,
        "habitat": 2,
    },
    1471730: {
        "name": "shark",
        "hypernym": 1471728,
        "parts": [0, 1, 2, 13, 9, 11],
        "size": 2,
        "habitat": 2,
    },
    1471731: {
        "name": "clownfish",
        "hypernym": 1471728,
        "parts": [0, 1, 2, 13, 9, 11],
        "size": 0,
        "habitat": 2,
    },

    # Reptiles (WordNet: reptile 01662765)
    1662765: {
        "name": "reptile",
        "hypernym": 2000000,
        "parts": [0, 1, 2, 3, 4, 5, 6, 9, 11],
        "size": 0,
        "habitat": 0,
    },
    1662766: {
        "name": "snake",
        "hypernym": 1662765,
        "parts": [0, 1, 2, 9, 11],
        "size": 1,
        "habitat": 0,
    },
    1662767: {
        "name": "turtle",
        "hypernym": 1662765,
        "parts": [0, 1, 3, 4, 5, 6, 9, 11, 19],
        "size": 0,
        "habitat": 3,
    },
    1662768: {
        "name": "lizard",
        "hypernym": 1662765,
        "parts": [0, 1, 2, 3, 4, 5, 6, 9, 11],
        "size": 0,
        "habitat": 0,
    },
    1662769: {
        "name": "crocodile",
        "hypernym": 1662765,
        "parts": [0, 1, 2, 3, 4, 5, 6, 9, 11],
        "size": 2,
        "habitat": 3,
    },

    # Insects (WordNet: insect 02159955)
    2159955: {
        "name": "insect",
        "hypernym": 2000000,
        "parts": [0, 1, 2, 3, 4, 5, 6, 7],
        "size": 0,
        "habitat": 0,
    },
    2159956: {
        "name": "butterfly",
        "hypernym": 2159955,
        "parts": [0, 1, 2, 3, 4, 5, 6, 7],
        "size": 0,
        "habitat": 1,
    },
    2159957: {
        "name": "bee",
        "hypernym": 2159955,
        "parts": [0, 1, 2, 3, 4, 5, 6, 7],
        "size": 0,
        "habitat": 1,
    },

    # ── Non-animal scene elements ──────────────────────────────────────
    9223372: {
        "name": "sky",
        "hypernym": 0,
        "parts": [],
        "size": 2,
        "habitat": 1,
    },
    9223373: {
        "name": "grass",
        "hypernym": 1000000,
        "parts": [],
        "size": 2,
        "habitat": 0,
    },
    9223374: {
        "name": "tree",
        "hypernym": 1000000,
        "parts": [],
        "size": 2,
        "habitat": 0,
    },
    9223375: {
        "name": "water",
        "hypernym": 0,
        "parts": [],
        "size": 2,
        "habitat": 2,
    },
    9223376: {
        "name": "ground",
        "hypernym": 0,
        "parts": [],
        "size": 2,
        "habitat": 0,
    },
    9223377: {
        "name": "sun",
        "hypernym": 0,
        "parts": [],
        "size": 0,
        "habitat": 1,
    },
    9223378: {
        "name": "cloud",
        "hypernym": 0,
        "parts": [],
        "size": 1,
        "habitat": 1,
    },
    9223379: {
        "name": "moon",
        "hypernym": 0,
        "parts": [],
        "size": 0,
        "habitat": 1,
    },
    9223380: {
        "name": "flower",
        "hypernym": 1000000,
        "parts": [],
        "size": 0,
        "habitat": 0,
    },
    9223381: {
        "name": "mountain",
        "hypernym": 0,
        "parts": [],
        "size": 2,
        "habitat": 0,
    },
}


# ── Part descriptions ─────────────────────────────────────────────────────
PART_NAMES = {
    0: "head",
    1: "torso",
    2: "tail",
    3: "leg_left_front",
    4: "leg_right_front",
    5: "leg_left_back",
    6: "leg_right_back",
    7: "wing",
    8: "ear",
    9: "eye",
    10: "nose",
    11: "mouth",
    12: "horn",
    13: "fin",
    14: "claw",
    15: "beak",
    16: "whisker",
    17: "mane",
    18: "trunk",
    19: "shell",
}


# ── Part synonyms for matching ────────────────────────────────────────────
PART_SYNONYMS = {
    "head": 0, "face": 0,
    "torso": 1, "body": 1, "chest": 1,
    "tail": 2,
    "leg": 3, "front_leg": 3, "paw": 3,
    "back_leg": 5, "hind_leg": 5, "rear_leg": 5,
    "wing": 7,
    "ear": 8,
    "eye": 9, "eyes": 9,
    "nose": 10, "snout": 10, "muzzle": 10,
    "mouth": 11,
    "horn": 12, "antler": 12,
    "fin": 13,
    "claw": 14, "nail": 14,
    "beak": 15,
    "whisker": 16,
    "mane": 17,
    "trunk": 18,
    "shell": 19,
}


# ── Name → wnid lookup (including plural forms) ───────────────────────────
_NAME_TO_WNID = {}
for wid, syn in SYNSETS.items():
    name = syn["name"]
    # Remove underscores and store
    _NAME_TO_WNID[name] = wid
    _NAME_TO_WNID[name.replace("_", " ")] = wid
    # Plural
    if not name.endswith("s"):
        _NAME_TO_WNID[name + "s"] = wid
    if name.endswith("y"):
        _NAME_TO_WNID[name[:-1] + "ies"] = wid


def resolve_name(name):
    """Look up a name and return its wnid, or None."""
    key = name.lower().strip().replace("-", "_")
    # Direct lookups
    if key in _NAME_TO_WNID:
        return _NAME_TO_WNID[key]
    # Try removing article-like prefixes
    for prefix in ["a ", "an ", "the "]:
        if key.startswith(prefix):
            return _NAME_TO_WNID.get(key[len(prefix):])
    # Fuzzy: check if it's a substring of any key
    for k, v in _NAME_TO_WNID.items():
        if key in k or k in key:
            return v
    return None


def resolve_part(name):
    """Look up a part name and return its part_id, or None."""
    key = name.lower().strip()
    if key in PART_SYNONYMS:
        return PART_SYNONYMS[key]
    # Try plural
    if key.endswith("s") and key[:-1] in PART_SYNONYMS:
        return PART_SYNONYMS[key[:-1]]
    return None


def get_hypernym_chain(wnid):
    """Return the chain from this wnid up to root."""
    chain = []
    while wnid is not None and wnid in SYNSETS:
        chain.append(wnid)
        wnid = SYNSETS[wnid]["hypernym"]
    return chain


def is_animal(wnid):
    """Check if wnid is an animal or descendant of animal."""
    chain = get_hypernym_chain(wnid)
    return 2000000 in chain  # animal


def get_parts(wnid):
    """Get default parts for a concept, inheriting from hypernyms."""
    if wnid in SYNSETS:
        return SYNSETS[wnid]["parts"]
    return []


def get_inherited_parts(wnid):
    """Get parts including those inherited from parent concepts."""
    parts = set()
    for wid in get_hypernym_chain(wnid):
        parts.update(SYNSETS.get(wid, {}).get("parts", []))
    return sorted(parts)


def get_size(wnid):
    """Get size category, inheriting from hypernyms."""
    for wid in get_hypernym_chain(wnid):
        if wid in SYNSETS:
            s = SYNSETS[wid].get("size")
            if s is not None:
                return s
    return 1


def get_habitat(wnid):
    """Get habitat, inheriting from hypernyms."""
    for wid in get_hypernym_chain(wnid):
        if wid in SYNSETS:
            h = SYNSETS[wid].get("habitat")
            if h is not None:
                return h
    return 0


if __name__ == "__main__":
    # Test the knowledge base
    print("=== Knowledge Base Test ===")
    for name in ["dog", "cat", "husky", "eagle", "shark", "butterfly"]:
        wid = resolve_name(name)
        if wid:
            syn = SYNSETS.get(wid, {})
            parts = get_inherited_parts(wid)
            part_names = [PART_NAMES.get(p, f"unknown_{p}") for p in parts]
            size = get_size(wid)
            habitat = get_habitat(wid)
            print(f"\n{name} (wnid={wid}):")
            print(f"  Parts: {part_names}")
            print(f"  Size: {['small', 'medium', 'large'][size]}")
            print(f"  Habitat: {['land', 'air', 'water', 'both'][habitat]}")
            print(f"  Chain: {get_hypernym_chain(wid)}")
        else:
            print(f"\n{name}: NOT FOUND")
