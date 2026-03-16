import re
from typing import List, Tuple, Dict


TAG_WITH_TEXT = re.compile(r"\[([^\]:]+):\s*([^\]]+)\]")
METADATA_TAGS = {"background color style", "position"}


def extract_tag_value(text: str, tag_name: str) -> str:
    pattern = re.compile(rf"\[{re.escape(tag_name)}:\s*([^\]]+)\]", flags=re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def strip_tags_except_metadata(text: str) -> str:
    text = re.sub(r"\]\s*\[", "] [", text)

    def replacer(match: re.Match) -> str:
        label = match.group(1).strip().lower()
        content = match.group(2).strip()
        if label in METADATA_TAGS:
            return ""
        return content

    cleaned = TAG_WITH_TEXT.sub(replacer, text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def process_object_segment(sentences: List[str], object_index: int) -> Dict[str, str]:
    combined = " ".join(sentences)
    object_name = extract_tag_value(combined, "ARG0") or f"Object {object_index + 1}"
    background = extract_tag_value(combined, "Background color style")
    position = extract_tag_value(combined, "Position")

    cleaned_sentences = []
    for sentence in sentences:
        cleaned = strip_tags_except_metadata(sentence)
        if cleaned:
            cleaned_sentences.append(cleaned.rstrip("."))

    description = ". ".join(cleaned_sentences).strip()
    if description:
        description = description[0].upper() + description[1:]
        if not description.endswith("."):
            description += "."

    return {
        "name": object_name,
        "background": background,
        "position": position,
        "description": description or object_name,
    }


def normalize_position_label(label: str) -> str:
    if not label:
        return ""
    label = label.lower()
    if "left" in label:
        return "left"
    if "right" in label:
        return "right"
    if "top" in label or "upper" in label:
        return "top"
    if "bottom" in label or "lower" in label:
        return "bottom"
    return ""


def enforce_direction(positions: List[str]) -> Tuple[List[str], set]:
    canonical = [normalize_position_label(pos) for pos in positions]
    orientation = None
    if any(pos in {"left", "right"} for pos in canonical):
        orientation = "horizontal"
    if any(pos in {"top", "bottom"} for pos in canonical):
        orientation = "vertical"
    if orientation is None:
        orientation = "horizontal"

    if orientation == "vertical":
        desired = ["top", "bottom"]
    else:
        desired = ["left", "right"]

    adjusted = []
    for idx, pos in enumerate(canonical):
        if pos in desired:
            adjusted.append(pos)
        else:
            adjusted.append(desired[idx % len(desired)])

    direction_set = {"top", "bottom"} if orientation == "vertical" else {"left", "right"}
    return adjusted, direction_set


def decoded_sentences_to_prompt_and_direction(decoded_sentences: List[str]) -> Tuple[List[str], set]:
    num_objects = 2
    if not decoded_sentences or len(decoded_sentences) % num_objects != 0:
        raise ValueError(
            "decoded_sentences length should be divisible by 2 (two sentences per object for two objects)."
        )

    sentences_per_object = len(decoded_sentences) // num_objects
    if sentences_per_object == 0:
        raise ValueError("decoded_sentences does not contain enough entries to form objects.")

    object_infos = []
    for obj_idx in range(num_objects):
        start = obj_idx * sentences_per_object
        end = start + sentences_per_object
        object_sentences = decoded_sentences[start:end]
        object_infos.append(process_object_segment(object_sentences, obj_idx))

    background_candidates = [info["background"] for info in object_infos if info["background"]]
    background_text = background_candidates[0] if background_candidates else "neutral"

    positions = [info["position"] or "" for info in object_infos]
    adjusted_positions, direction = enforce_direction(positions)
    for info, pos in zip(object_infos, adjusted_positions):
        info["position"] = pos.capitalize()

    obj_names = [info["name"] for info in object_infos]
    prompt0 = f"{obj_names[0]}, {obj_names[1]}, {background_text} background"

    prompt = [
        prompt0,
        object_infos[0]["description"],
        object_infos[1]["description"],
    ]

    return prompt, direction
