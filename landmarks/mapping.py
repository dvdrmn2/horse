from __future__ import annotations

from config.config import CANONICAL_LANDMARKS, HORSE10_LANDMARKS, KEYPOINT_SCORE_THRESHOLD
from landmarks.derived import apply_derived_landmarks
from models.landmark import LandmarkSet
from models.landmark_types import LandmarkSource, LandmarkStatus
from models.skeleton import Skeleton

ANIMALPOSE_LANDMARKS = (
    "L_Eye",
    "R_Eye",
    "L_EarBase",
    "R_EarBase",
    "Nose",
    "Throat",
    "TailBase",
    "Withers",
    "L_F_Elbow",
    "R_F_Elbow",
    "L_B_Elbow",
    "R_B_Elbow",
    "L_F_Knee",
    "R_F_Knee",
    "L_B_Knee",
    "R_B_Knee",
    "L_F_Paw",
    "R_F_Paw",
    "L_B_Paw",
    "R_B_Paw",
)

ANIMALPOSE_TO_CANONICAL = {
    "Nose": "nose",
    "L_Eye": "left_eye",
    "R_Eye": "right_eye",
    "Throat": "neck",
    "Withers": "withers",
    "TailBase": "tail_head",
    "L_F_Elbow": "left_elbow",
    "R_F_Elbow": "right_elbow",
    "L_F_Knee": "left_carpus",
    "R_F_Knee": "right_carpus",
    "L_F_Paw": "left_front_hoof",
    "R_F_Paw": "right_front_hoof",
    "L_B_Knee": "left_stifle",
    "R_B_Knee": "right_stifle",
    "L_B_Paw": "left_hind_hoof",
    "R_B_Paw": "right_hind_hoof",
}

ANIMALPOSE_CANONICAL_COVERAGE = frozenset(ANIMALPOSE_TO_CANONICAL.values())

ANIMALPOSE_UNMAPPED_CANONICAL = [
    name for name in CANONICAL_LANDMARKS if name not in ANIMALPOSE_CANONICAL_COVERAGE
]

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

HORSE10_CANONICAL_COVERAGE = frozenset(HORSE10_TO_CANONICAL.values())

HORSE10_UNMAPPED_CANONICAL = [
    name for name in CANONICAL_LANDMARKS if name not in HORSE10_CANONICAL_COVERAGE
]


def _status_for_source_point(confidence: float | None, visible: bool) -> LandmarkStatus:
    if not visible or confidence is None:
        return LandmarkStatus.NOT_VISIBLE
    if confidence >= KEYPOINT_SCORE_THRESHOLD:
        return LandmarkStatus.DETECTED
    if confidence > 0:
        return LandmarkStatus.LOW_CONFIDENCE
    return LandmarkStatus.NOT_VISIBLE


def _initialize_unmapped(target: LandmarkSet, unmapped_names: list[str]) -> LandmarkSet:
    for name in unmapped_names:
        target = target.with_status(name, LandmarkStatus.UNMAPPED, LandmarkSource.NONE)
    return target


def map_horse10_to_canonical(source: LandmarkSet) -> LandmarkSet:
    target = _initialize_unmapped(LandmarkSet.template(CANONICAL_LANDMARKS), HORSE10_UNMAPPED_CANONICAL)

    for source_name, target_name in HORSE10_TO_CANONICAL.items():
        landmark = source.get(source_name)
        if landmark is None:
            target = target.with_status(target_name, LandmarkStatus.NOT_VISIBLE, LandmarkSource.HORSE10)
            continue

        status = _status_for_source_point(landmark.confidence, landmark.visible)
        if status is LandmarkStatus.NOT_VISIBLE:
            target = target.with_status(target_name, status, LandmarkSource.HORSE10)
            continue

        target = target.with_point(
            target_name,
            landmark.x,
            landmark.y,
            landmark.confidence,
            visible=True,
            status=status,
            source=LandmarkSource.HORSE10,
        )

    return apply_derived_landmarks(target)


def map_animalpose_to_canonical(source: LandmarkSet) -> LandmarkSet:
    target = _initialize_unmapped(LandmarkSet.template(CANONICAL_LANDMARKS), ANIMALPOSE_UNMAPPED_CANONICAL)

    for source_name, target_name in ANIMALPOSE_TO_CANONICAL.items():
        landmark = source.get(source_name)
        if landmark is None:
            target = target.with_status(target_name, LandmarkStatus.NOT_VISIBLE, LandmarkSource.ANIMALPOSE)
            continue

        status = _status_for_source_point(landmark.confidence, landmark.visible)
        if status is LandmarkStatus.NOT_VISIBLE:
            target = target.with_status(target_name, status, LandmarkSource.ANIMALPOSE)
            continue

        target = target.with_point(
            target_name,
            landmark.x,
            landmark.y,
            landmark.confidence,
            visible=True,
            status=status,
            source=LandmarkSource.ANIMALPOSE,
        )

    return apply_derived_landmarks(target)


def map_to_canonical(source: LandmarkSet, source_schema: str) -> LandmarkSet:
    if source_schema == "horse10":
        if tuple(source.schema) != tuple(HORSE10_LANDMARKS):
            raise ValueError("Expected a Horse-10 landmark set")
        return map_horse10_to_canonical(source)

    if source_schema == "animalpose":
        return map_animalpose_to_canonical(source)

    if source_schema == "canonical":
        return source

    raise ValueError(f"Unsupported landmark schema: {source_schema}")


def to_skeleton(landmarks: LandmarkSet, backend: str) -> Skeleton:
    return Skeleton.from_landmark_set(landmarks, backend=backend)
