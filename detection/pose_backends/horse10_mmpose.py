from __future__ import annotations

from pathlib import Path

import numpy as np

from config.config import (
    HORSE10_LANDMARKS,
    KEYPOINT_SCORE_THRESHOLD,
    MMPOSE_CHECKPOINT,
    MMPOSE_CONFIG,
)
from detection.pose_backends.base import PoseBackend
from models.detection import Detection
from models.landmark import LandmarkSet


class Horse10MMPoseBackend(PoseBackend):
    """Horse-10 pose inference via MMPose (optional dependency)."""

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        score_threshold: float = KEYPOINT_SCORE_THRESHOLD,
    ):
        self.config_path = config_path or MMPOSE_CONFIG
        self.checkpoint_path = checkpoint_path or MMPOSE_CHECKPOINT
        self.score_threshold = score_threshold
        self.device = device or self._default_device()
        self._model = None

    @property
    def schema_name(self) -> str:
        return "horse10"

    def _default_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass

        return "cpu"

    def _resolve_config_path(self) -> str:
        config_path = Path(self.config_path)

        if config_path.is_file():
            return str(config_path)

        try:
            import mmpose

            bundled_config = (
                Path(mmpose.__file__).resolve().parent
                / ".mim"
                / "configs"
                / self.config_path
            )
            if bundled_config.is_file():
                return str(bundled_config)
        except ImportError as exc:
            raise ImportError(
                "MMPose is not installed. Install optional deps with:\n"
                "  pip install -r requirements-optional-mmpose.txt"
            ) from exc

        raise FileNotFoundError(
            "Could not find the Horse-10 MMPose config. "
            "Reinstall mmpose or pass an explicit config_path."
        )

    def _load_model(self):
        if self._model is not None:
            return self._model

        from mmpose.apis import init_model

        self._model = init_model(
            self._resolve_config_path(),
            self.checkpoint_path,
            device=self.device,
        )
        return self._model

    def estimate(self, frame, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        from mmpose.apis import inference_topdown

        model = self._load_model()
        bboxes = np.array([detection.bbox for detection in detections], dtype=np.float32)
        pose_results = inference_topdown(model, frame, bboxes=bboxes)

        enriched = []

        for detection, pose_result in zip(detections, pose_results):
            keypoints = pose_result.pred_instances.keypoints[0]
            scores = pose_result.pred_instances.keypoint_scores[0]

            enriched.append(
                Detection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    landmarks=LandmarkSet.from_arrays(
                        keypoints,
                        scores,
                        HORSE10_LANDMARKS,
                        score_threshold=self.score_threshold,
                    ),
                )
            )

        return enriched
