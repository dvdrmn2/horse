from __future__ import annotations

from config.config import POSE_BACKEND
from detection.pose_backends.base import PoseBackend
from detection.pose_backends.horse10_mmpose import Horse10MMPoseBackend
from detection.pose_backends.superanimal import SuperAnimalPoseBackend
from detection.pose_backends.template import TemplatePoseBackend

BACKENDS = {
    "template": TemplatePoseBackend,
    "horse10_mmpose": Horse10MMPoseBackend,
    "superanimal": SuperAnimalPoseBackend,
}


def create_pose_backend(name: str | None = None) -> PoseBackend:
    backend_name = (name or POSE_BACKEND).lower()

    if backend_name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unknown pose backend '{backend_name}'. Available: {available}"
        )

    return BACKENDS[backend_name]()
