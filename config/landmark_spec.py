"""Canonical landmark specification for horse pose analysis.

Definitions are grounded in equine anatomy and what a human annotator can
reliably click in video — not copied from AnimalPose or Horse-10 schemas.
Backend model landmarks are mapped *into* this system; they do not define it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ViewName = Literal["front", "side", "rear", "oblique", "unknown"]
LandmarkCategory = Literal["head", "neck", "torso", "forelimb", "hindlimb"]


@dataclass(frozen=True)
class ViewSuitability:
    """How reliably a landmark can be localized from each camera view (0–1)."""

    front: float
    side: float
    rear: float
    oblique: float

    def score(self, view: ViewName) -> float:
        if view == "unknown":
            return 0.5
        return getattr(self, view)


@dataclass(frozen=True)
class LandmarkDefinition:
    name: str
    anatomical_definition: str
    annotation_definition: str
    category: LandmarkCategory
    tier: int
    view_suitability: ViewSuitability
    required: bool = True
    symmetric: bool = False


# fmt: off
LANDMARK_DEFINITIONS: dict[str, LandmarkDefinition] = {
    "nose": LandmarkDefinition(
        name="nose",
        anatomical_definition="Tip of the muzzle at the anterior-most point of the nasal planum.",
        annotation_definition="Click the visually identifiable tip of the muzzle.",
        category="head",
        tier=1,
        view_suitability=ViewSuitability(front=1.0, side=0.95, rear=0.2, oblique=0.85),
        required=True,
        symmetric=False,
    ),
    "left_eye": LandmarkDefinition(
        name="left_eye",
        anatomical_definition="Center of the left orbital opening.",
        annotation_definition="Center of the visible left eye or orbital region.",
        category="head",
        tier=2,
        view_suitability=ViewSuitability(front=0.9, side=0.85, rear=0.15, oblique=0.75),
        required=False,
        symmetric=True,
    ),
    "right_eye": LandmarkDefinition(
        name="right_eye",
        anatomical_definition="Center of the right orbital opening.",
        annotation_definition="Center of the visible right eye or orbital region.",
        category="head",
        tier=2,
        view_suitability=ViewSuitability(front=0.9, side=0.85, rear=0.15, oblique=0.75),
        required=False,
        symmetric=True,
    ),
    "poll": LandmarkDefinition(
        name="poll",
        anatomical_definition="Dorsal junction between the atlas and the occiput, behind the ears.",
        annotation_definition="Highest palpable point behind the ears where the neck meets the skull.",
        category="head",
        tier=2,
        view_suitability=ViewSuitability(front=0.55, side=0.9, rear=0.7, oblique=0.8),
        required=False,
        symmetric=False,
    ),
    "neck": LandmarkDefinition(
        name="neck",
        anatomical_definition="Midpoint along the ventral cervical midline between throat and chest.",
        annotation_definition="Center of the visible throat/underside of the neck in profile.",
        category="neck",
        tier=2,
        view_suitability=ViewSuitability(front=0.45, side=0.9, rear=0.35, oblique=0.7),
        required=False,
        symmetric=False,
    ),
    "withers": LandmarkDefinition(
        name="withers",
        anatomical_definition="Highest point of the thoracic spinous processes at the base of the neck.",
        annotation_definition="Highest visible point of the back at the shoulder/neck junction.",
        category="torso",
        tier=1,
        view_suitability=ViewSuitability(front=0.6, side=1.0, rear=0.85, oblique=0.9),
        required=True,
        symmetric=False,
    ),
    "spine_mid": LandmarkDefinition(
        name="spine_mid",
        anatomical_definition="Midpoint of the thoracic/lumbar dorsal midline between withers and croup.",
        annotation_definition="Center of the visible back line halfway between withers and croup.",
        category="torso",
        tier=2,
        view_suitability=ViewSuitability(front=0.25, side=0.95, rear=0.8, oblique=0.75),
        required=False,
        symmetric=False,
    ),
    "croup": LandmarkDefinition(
        name="croup",
        anatomical_definition="Highest point of the sacral dorsal midline above the pelvis.",
        annotation_definition="Highest visible point of the rump/top of the pelvis.",
        category="torso",
        tier=2,
        view_suitability=ViewSuitability(front=0.3, side=0.95, rear=0.95, oblique=0.85),
        required=False,
        symmetric=False,
    ),
    "tail_head": LandmarkDefinition(
        name="tail_head",
        anatomical_definition="Base of the tail at its attachment to the sacrum.",
        annotation_definition="Center of the visible tail root where the tail leaves the body.",
        category="torso",
        tier=2,
        view_suitability=ViewSuitability(front=0.2, side=0.85, rear=0.95, oblique=0.7),
        required=False,
        symmetric=False,
    ),
    "left_shoulder": LandmarkDefinition(
        name="left_shoulder",
        anatomical_definition="Point of the shoulder at the scapulohumeral joint region.",
        annotation_definition="Center of the visible left shoulder point of the horse.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.75, side=0.95, rear=0.35, oblique=0.85),
        required=True,
        symmetric=True,
    ),
    "right_shoulder": LandmarkDefinition(
        name="right_shoulder",
        anatomical_definition="Point of the shoulder at the scapulohumeral joint region.",
        annotation_definition="Center of the visible right shoulder point of the horse.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.75, side=0.95, rear=0.35, oblique=0.85),
        required=True,
        symmetric=True,
    ),
    "left_elbow": LandmarkDefinition(
        name="left_elbow",
        anatomical_definition="Center of the left carpal (elbow) joint.",
        annotation_definition="Center of the visually identifiable left elbow joint.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.5, side=0.95, rear=0.3, oblique=0.8),
        required=True,
        symmetric=True,
    ),
    "right_elbow": LandmarkDefinition(
        name="right_elbow",
        anatomical_definition="Center of the right carpal (elbow) joint.",
        annotation_definition="Center of the visually identifiable right elbow joint.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.5, side=0.95, rear=0.3, oblique=0.8),
        required=True,
        symmetric=True,
    ),
    "left_carpus": LandmarkDefinition(
        name="left_carpus",
        anatomical_definition="Center of the left carpus (knee) joint.",
        annotation_definition="Center of the visually identifiable left front knee.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.45, side=0.95, rear=0.25, oblique=0.75),
        required=True,
        symmetric=True,
    ),
    "right_carpus": LandmarkDefinition(
        name="right_carpus",
        anatomical_definition="Center of the right carpus (knee) joint.",
        annotation_definition="Center of the visually identifiable right front knee.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.45, side=0.95, rear=0.25, oblique=0.75),
        required=True,
        symmetric=True,
    ),
    "left_fetlock": LandmarkDefinition(
        name="left_fetlock",
        anatomical_definition="Center of the left metacarpophalangeal (fetlock) joint.",
        annotation_definition="Center of the visually identifiable left front fetlock.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.4, side=0.9, rear=0.2, oblique=0.7),
        required=True,
        symmetric=True,
    ),
    "right_fetlock": LandmarkDefinition(
        name="right_fetlock",
        anatomical_definition="Center of the right metacarpophalangeal (fetlock) joint.",
        annotation_definition="Center of the visually identifiable right front fetlock.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.4, side=0.9, rear=0.2, oblique=0.7),
        required=True,
        symmetric=True,
    ),
    "left_front_hoof": LandmarkDefinition(
        name="left_front_hoof",
        anatomical_definition="Ground contact point of the left front hoof.",
        annotation_definition="Lowest visible point of the left front hoof or pastern tip.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.35, side=0.85, rear=0.15, oblique=0.65),
        required=True,
        symmetric=True,
    ),
    "right_front_hoof": LandmarkDefinition(
        name="right_front_hoof",
        anatomical_definition="Ground contact point of the right front hoof.",
        annotation_definition="Lowest visible point of the right front hoof or pastern tip.",
        category="forelimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.35, side=0.85, rear=0.15, oblique=0.65),
        required=True,
        symmetric=True,
    ),
    "left_hip": LandmarkDefinition(
        name="left_hip",
        anatomical_definition="Lateral aspect of the left hip (coxofemoral region).",
        annotation_definition="Center of the visible left hip/pelvis outline.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.55, side=0.9, rear=0.85, oblique=0.8),
        required=True,
        symmetric=True,
    ),
    "right_hip": LandmarkDefinition(
        name="right_hip",
        anatomical_definition="Lateral aspect of the right hip (coxofemoral region).",
        annotation_definition="Center of the visible right hip/pelvis outline.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.55, side=0.9, rear=0.85, oblique=0.8),
        required=True,
        symmetric=True,
    ),
    "left_stifle": LandmarkDefinition(
        name="left_stifle",
        anatomical_definition="Center of the left stifle (femorotibial) joint.",
        annotation_definition="Center of the visually identifiable left stifle.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.35, side=0.95, rear=0.55, oblique=0.75),
        required=True,
        symmetric=True,
    ),
    "right_stifle": LandmarkDefinition(
        name="right_stifle",
        anatomical_definition="Center of the right stifle (femorotibial) joint.",
        annotation_definition="Center of the visually identifiable right stifle.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.35, side=0.95, rear=0.55, oblique=0.75),
        required=True,
        symmetric=True,
    ),
    "left_hock": LandmarkDefinition(
        name="left_hock",
        anatomical_definition="Center of the left tarsal (hock) joint.",
        annotation_definition="Center of the visually identifiable left hock joint.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.25, side=1.0, rear=0.85, oblique=0.75),
        required=True,
        symmetric=True,
    ),
    "right_hock": LandmarkDefinition(
        name="right_hock",
        anatomical_definition="Center of the right tarsal (hock) joint.",
        annotation_definition="Center of the visually identifiable right hock joint.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.25, side=1.0, rear=0.85, oblique=0.75),
        required=True,
        symmetric=True,
    ),
    "left_hind_fetlock": LandmarkDefinition(
        name="left_hind_fetlock",
        anatomical_definition="Center of the left hind metatarsophalangeal (fetlock) joint.",
        annotation_definition="Center of the visually identifiable left hind fetlock.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.2, side=0.9, rear=0.7, oblique=0.65),
        required=True,
        symmetric=True,
    ),
    "right_hind_fetlock": LandmarkDefinition(
        name="right_hind_fetlock",
        anatomical_definition="Center of the right hind metatarsophalangeal (fetlock) joint.",
        annotation_definition="Center of the visually identifiable right hind fetlock.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.2, side=0.9, rear=0.7, oblique=0.65),
        required=True,
        symmetric=True,
    ),
    "left_hind_hoof": LandmarkDefinition(
        name="left_hind_hoof",
        anatomical_definition="Ground contact point of the left hind hoof.",
        annotation_definition="Lowest visible point of the left hind hoof or pastern tip.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.15, side=0.85, rear=0.75, oblique=0.6),
        required=True,
        symmetric=True,
    ),
    "right_hind_hoof": LandmarkDefinition(
        name="right_hind_hoof",
        anatomical_definition="Ground contact point of the right hind hoof.",
        annotation_definition="Lowest visible point of the right hind hoof or pastern tip.",
        category="hindlimb",
        tier=1,
        view_suitability=ViewSuitability(front=0.15, side=0.85, rear=0.75, oblique=0.6),
        required=True,
        symmetric=True,
    ),
}
# fmt: on

CANONICAL_LANDMARKS: list[str] = list(LANDMARK_DEFINITIONS.keys())

CANONICAL_SKELETON: list[tuple[str, str]] = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "poll"),
    ("right_eye", "poll"),
    ("poll", "neck"),
    ("neck", "withers"),
    ("withers", "spine_mid"),
    ("spine_mid", "croup"),
    ("croup", "tail_head"),
    ("withers", "left_shoulder"),
    ("withers", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_carpus"),
    ("left_carpus", "left_fetlock"),
    ("left_fetlock", "left_front_hoof"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_carpus"),
    ("right_carpus", "right_fetlock"),
    ("right_fetlock", "right_front_hoof"),
    ("croup", "left_hip"),
    ("croup", "right_hip"),
    ("left_hip", "left_stifle"),
    ("left_stifle", "left_hock"),
    ("left_hock", "left_hind_fetlock"),
    ("left_hind_fetlock", "left_hind_hoof"),
    ("right_hip", "right_stifle"),
    ("right_stifle", "right_hock"),
    ("right_hock", "right_hind_fetlock"),
    ("right_hind_fetlock", "right_hind_hoof"),
]


def get_landmark_definition(name: str) -> LandmarkDefinition:
    return LANDMARK_DEFINITIONS[name]


def landmark_names_for_tier(tier: int) -> list[str]:
    return [name for name, definition in LANDMARK_DEFINITIONS.items() if definition.tier == tier]


def observable_landmarks_for_view(view: ViewName, min_score: float = 0.5) -> list[str]:
    return [
        name
        for name, definition in LANDMARK_DEFINITIONS.items()
        if definition.view_suitability.score(view) >= min_score
    ]
