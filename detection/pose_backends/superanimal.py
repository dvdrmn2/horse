from __future__ import annotations

from config.config import HORSE10_LANDMARKS, SUPERANIMAL_MODEL
from detection.pose_backends.base import PoseBackend
from models.detection import Detection


class SuperAnimalPoseBackend(PoseBackend):
    """Horse-10-compatible pose inference via DeepLabCut SuperAnimal."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or SUPERANIMAL_MODEL
        self._model = None

    @property
    def schema_name(self) -> str:
        return "horse10"

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            import deeplabcut
        except ImportError as exc:
            raise ImportError(
                "DeepLabCut is not installed. Install optional deps with:\n"
                "  pip install -r requirements-optional-superanimal.txt"
            ) from exc

        # Stage 2 placeholder: wire SuperAnimal inference here.
        raise NotImplementedError(
            "SuperAnimal backend is scaffolded but not wired yet. "
            f"Model alias: {self.model_name}. "
            "Use POSE_BACKEND='horse10_mmpose' or 'template' for now."
        )

    def estimate(self, frame, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        self._load_model()

        raise NotImplementedError(
            "SuperAnimal inference is not implemented yet. "
            f"Expected output schema: {HORSE10_LANDMARKS}"
        )
