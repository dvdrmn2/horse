from __future__ import annotations

from config.config import CANONICAL_LANDMARKS, HORSE10_LANDMARKS
from models.landmark import LandmarkSet

# Best-effort mapping from Horse-10 (camera-relative) to our canonical schema.
# Unmapped canonical points stay empty until Stage 4/5 custom training.
HORSE10_TO_CANONICAL = {
    "nose": "nose",
    "eye": "left_eye",
    "wither": "withers",
    "mid_shoulder": "neck",
    "shoulder": "left_shoulder",
    "elbow": "left_elbow",
    "girth": "spine_mid",
    "hip": "left_hip",
    "stifle": "left_stifle",
    "ischium": "croup",
    "near_knee": "left_carpus",
    "near_front_fetlock": "left_fetlock",
    "near_front_foot": "left_front_hoof",
    "off_knee": "right_carpus",
    "off_front_fetlock": "right_fetlock",
    "off_front_foot": "right_front_hoof",
    "near_hind_hock": "left_hock",
    "near_hind_fetlock": "left_hind_fetlock",
    "near_hind_foot": "left_hind_hoof",
    "off_hind_hock": "right_hock",
    "off_hind_fetlock": "right_hind_fetlock",
    "off_hind_foot": "right_hind_hoof",
}


def map_horse10_to_canonical(source: LandmarkSet) -> LandmarkSet:
    target = LandmarkSet.template(CANONICAL_LANDMARKS)

    for source_name, target_name in HORSE10_TO_CANONICAL.items():
        landmark = source.get(source_name)
        if landmark is None or not landmark.visible:
            continue

        target = target.with_point(
            target_name,
            landmark.x,
            landmark.y,
            landmark.confidence,
        )

    return target


def map_to_canonical(source: LandmarkSet, source_schema: str) -> LandmarkSet:
    if source_schema == "horse10":
        if tuple(source.schema) != tuple(HORSE10_LANDMARKS):
            raise ValueError("Expected a Horse-10 landmark set")
        return map_horse10_to_canonical(source)

    if source_schema == "canonical":
        return source

    raise ValueError(f"Unsupported landmark schema: {source_schema}")
