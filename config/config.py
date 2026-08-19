# Target landmark schema — see config/landmark_spec.py for full definitions.
from config.landmark_spec import CANONICAL_LANDMARKS, CANONICAL_SKELETON, LANDMARK_DEFINITIONS

# Horse-10 schema used by MMPose / SuperAnimal quadruped models.
# Near/off are camera-relative (near = closer to camera).
HORSE10_LANDMARKS = [
    "nose",
    "eye",
    "near_knee",
    "near_front_fetlock",
    "near_front_foot",
    "off_knee",
    "off_front_fetlock",
    "off_front_foot",
    "shoulder",
    "mid_shoulder",
    "elbow",
    "girth",
    "wither",
    "near_hind_hock",
    "near_hind_fetlock",
    "near_hind_foot",
    "hip",
    "stifle",
    "off_hind_hock",
    "off_hind_fetlock",
    "off_hind_foot",
    "ischium",
]

HORSE10_SKELETON = [
    ("nose", "eye"),
    ("eye", "wither"),
    ("wither", "mid_shoulder"),
    ("mid_shoulder", "shoulder"),
    ("shoulder", "elbow"),
    ("elbow", "girth"),
    ("girth", "hip"),
    ("hip", "stifle"),
    ("stifle", "ischium"),
    ("shoulder", "near_knee"),
    ("near_knee", "near_front_fetlock"),
    ("near_front_fetlock", "near_front_foot"),
    ("shoulder", "off_knee"),
    ("off_knee", "off_front_fetlock"),
    ("off_front_fetlock", "off_front_foot"),
    ("hip", "near_hind_hock"),
    ("near_hind_hock", "near_hind_fetlock"),
    ("near_hind_fetlock", "near_hind_foot"),
    ("hip", "off_hind_hock"),
    ("off_hind_hock", "off_hind_fetlock"),
    ("off_hind_fetlock", "off_hind_foot"),
]

import os
from pathlib import Path

# Stage 2 pose backend: template | horse10_mmpose | superanimal
POSE_BACKEND = os.getenv("POSE_BACKEND", "template")

KEYPOINT_SCORE_THRESHOLD = 0.3

# MMPose quadruped pose (AnimalPose HRNet-W48).
# Horse-10 configs/checkpoints were removed from mmpose 1.3; AnimalPose still ships
# in the model zoo and works on horses for Stage 2 proof-of-concept.
MMPOSE_MODEL = os.getenv("MMPOSE_MODEL", "td-hm_hrnet-w48_8xb64-210e_animalpose-256x256")
MMPOSE_MODEL_DIR = os.getenv("MMPOSE_MODEL_DIR", "/opt/mmpose-model")
MMPOSE_CONFIG = os.getenv(
    "MMPOSE_CONFIG",
    str(Path(MMPOSE_MODEL_DIR) / f"{MMPOSE_MODEL}.py"),
)
MMPOSE_CHECKPOINT = os.getenv("MMPOSE_CHECKPOINT", "")

# SuperAnimal quadruped — optional, see requirements-optional.txt
SUPERANIMAL_MODEL = "superanimal_quadruped"
