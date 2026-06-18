"""Text Parser: Extracts entities, attributes, and scene context from prompts.

Strategy:
  1. Split prompt into clauses (by prepositions/conjunctions)
  2. For each clause, extract the main noun → entity
  3. Extract modifiers: color, size, action, pattern
  4. Extract scene-level attributes: background, lighting, weather, time
  5. Extract spatial relationships: "above", "below", "next to", etc.
"""

import re
from . import knowledge_base as kb
from . import attributes as attr


# ── Parts-of-speech helpers ───────────────────────────────────────────────
# Words that might indicate a color is coming
_COLOR_TRIGGERS = {"color", "colored", "coloured", "-colored"}
_ACTION_INDICATORS = {"is", "was", "are", "were", "has", "have", "with"}


def _tokenize(text):
    """Split text into lowercased tokens, preserving known multi-word terms."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    # Merge known multi-word colors
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            pair = f"{tokens[i]} {tokens[i+1]}"
            if attr.resolve_color(pair) is not None:
                merged.append(pair)
                i += 2
                continue
        merged.append(tokens[i])
        i += 1
    return merged


_STOP_WORDS = {
    "a", "an", "the", "on", "in", "under", "over", "with", "near",
    "next", "to", "at", "by", "and", "or", "is", "was", "are", "were",
    "of", "for", "from", "behind", "beside", "between", "above", "below",
    "beneath", "inside", "upon", "has", "have", "having",
}


def _find_entities(tokens):
    """Find entity mentions (animals, objects) in token stream."""
    entities = []
    for token in tokens:
        if len(token) < 2 or token in _STOP_WORDS:
            continue
        if attr.resolve_color(token) is not None:
            continue
        if attr.resolve_action(token) != 0:
            continue
        wid = kb.resolve_name(token)
        if wid is not None:
            syn = kb.SYNSETS.get(wid, {})
            entities.append({
                "name": syn.get("name", token),
                "wnid": wid,
                "token": token,
            })
    return entities


def _extract_attributes(tokens, entity_idx):
    """Extract color, size, action, pattern, texture for an entity.
    
    Looks at tokens before the entity (adjectives, max 3 back) and
    after (verb phrases, max 3 forward).
    """
    color = None
    size = None
    action = None
    pattern = None
    texture = None
    age = None
    
    # Check up to 3 tokens BEFORE entity for adjectives
    lookback_start = max(0, entity_idx - 3)
    for token in tokens[lookback_start:entity_idx]:
        rgb = attr.resolve_color(token)
        if rgb is not None:
            color = rgb
            continue
        sz = attr.resolve_size(token)
        if sz is not None:
            size = sz
            continue
        pat = attr.resolve_pattern(token)
        if pat != 0:
            pattern = pat
            continue
        tex = attr.resolve_texture(token)
        if tex != 0:
            texture = tex
            continue
        if token in {"baby", "infant", "young", "tiny", "newborn"}:
            age = 0.0
        elif token in {"adult"}:
            age = 0.5
        elif token in {"elderly", "old", "ancient"}:
            age = 1.0
    
    # Check up to 3 tokens AFTER entity for actions only
    lookahead_end = min(len(tokens), entity_idx + 4)
    for token in tokens[entity_idx + 1:lookahead_end]:
        act = attr.resolve_action(token)
        if act != 0:
            action = act
    
    return {
        "color": color,
        "size": size,
        "action": action,
        "pattern": pattern,
        "texture": texture,
        "age": age,
    }


def _extract_scene_attrs(tokens):
    """Extract scene-level attributes (background, lighting, weather, time)."""
    scene_type = None
    lighting = None
    time_of_day = None
    weather = None
    
    for token in tokens:
        if scene_type is None:
            st = attr.resolve_scene_type(token)
            if st != 0:
                scene_type = st
        if lighting is None:
            lt = attr.resolve_lighting(token)
            if lt is not None:
                lighting = lt
        if time_of_day is None:
            td = attr.resolve_time(token)
            if td is not None:
                time_of_day = td
        if weather is None:
            wt = attr.resolve_weather(token)
            if wt is not None:
                weather = wt
    
    return {
        "scene_type": scene_type,
        "lighting": lighting,
        "time_of_day": time_of_day,
        "weather": weather,
        "bg_color": None,
    }


def _extract_relationships(tokens, entities):
    """Extract spatial relationships between entities."""
    rels = []
    keywords = {
        "above": "above", "below": "below", "over": "above",
        "under": "below", "beneath": "below", "underneath": "below",
        "next to": "next_to", "beside": "next_to", "near": "next_to",
        "on": "on", "upon": "on", "in": "in", "inside": "in",
        "behind": "behind", "in front of": "in_front", "before": "in_front",
        "left of": "left_of", "right of": "right_of",
        "between": "between",
    }
    
    # Simple proximity-based: check if a relationship keyword appears
    # between two entity mentions
    text_lower = " ".join(tokens)
    for keyword, rel_type in keywords.items():
        if keyword in text_lower:
            # Find which entities are on each side
            idx = text_lower.find(keyword)
            before = text_lower[:idx].strip()
            after = text_lower[idx + len(keyword):].strip()
            e_before = None
            e_after = None
            for e in entities:
                if e["token"] in before:
                    e_before = e
                if e["token"] in after:
                    e_after = e
            if e_before and e_after:
                rels.append({
                    "type": rel_type,
                    "subject": e_before["wnid"],
                    "object": e_after["wnid"],
                })
    
    return rels


def parse(prompt):
    """Parse a prompt into structured scene description.
    
    Returns:
        dict with:
          - entities: list of {name, wnid, attributes, bbox_guess}
          - scene: {scene_type, lighting, time_of_day, weather, bg_color}
          - relationships: list of {type, subject, object}
          - raw_tokens: list of tokens
    """
    tokens = _tokenize(prompt)
    
    # Find entities
    entities_raw = _find_entities(tokens)
    
    # Deduplicate entities by name
    seen = set()
    entities = []
    for e in entities_raw:
        if e["name"] not in seen:
            seen.add(e["name"])
            entities.append(e)
    
    # Extract attributes per entity
    for ent in entities:
        ent_idx = None
        for i, t in enumerate(tokens):
            if t == ent["token"]:
                ent_idx = i
                break
        attrs = _extract_attributes(tokens, ent_idx) if ent_idx is not None else {}
        ent.update(attrs)
    
    # Extract scene attributes
    scene = _extract_scene_attrs(tokens)
    
    # Extract relationships
    relationships = _extract_relationships(tokens, entities)
    
    # Infer missing attributes from knowledge base
    for ent in entities:
        if ent["size"] is None:
            ent["size"] = kb.get_size(ent["wnid"])
    
    return {
        "entities": entities,
        "scene": scene,
        "relationships": relationships,
        "raw_tokens": tokens,
    }


if __name__ == "__main__":
    test_prompts = [
        "a brown dog sitting under a tree",
        "a white cat sleeping on a blue sky with clouds",
        "a golden retriever puppy running in green grass",
        "a black bird flying above a mountain at sunset",
        "a horse standing in a field near a tree",
        "a husky sitting on snow with a cloudy sky",
        "a tiger running on grass",
        "a white rabbit hopping on grass",
        "a brown bear standing near water",
        "a goldfish swimming in blue water",
        "a red and white clownfish swimming in blue water",
    ]
    
    print("=== Parser Test ===\n")
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        result = parse(prompt)
        print(f"  Entities ({len(result['entities'])}):")
        for e in result["entities"]:
            rgb_str = f"rgb{e['color']}" if e["color"] else "default"
            action = ["standing", "sitting", "running", "flying", "swimming"][e["action"]] if e["action"] is not None else "default"
            size = ["small", "medium", "large"][e["size"]] if e["size"] is not None else "default"
            print(f"    {e['name']} (wnid={e['wnid']}) — color={rgb_str}, action={action}, size={size}")
        print(f"  Scene: {result['scene']}")
        if result["relationships"]:
            print(f"  Relations: {result['relationships']}")
        print()
