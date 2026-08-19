from __future__ import annotations

from pathlib import Path

import numpy as np

from config.config import (
    KEYPOINT_SCORE_THRESHOLD,
    MMPOSE_CHECKPOINT,
    MMPOSE_CONFIG,
    MMPOSE_MODEL,
    MMPOSE_MODEL_DIR,
)
from detection.pose_backends.base import PoseBackend
from models.detection import Detection
from models.landmark import LandmarkSet


class Horse10MMPoseBackend(PoseBackend):
    """Quadruped pose inference via MMPose (AnimalPose by default)."""

    MAC_INSTALL_HELP = (
        "MMPose requires mmcv, which has no native wheel for Mac ARM + Python 3.9.\n"
        "Options:\n"
        "  1) Install Python 3.11, recreate venv, then:\n"
        "     pip install torch==2.1.0 mmengine\n"
        "     pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html\n"
        "     pip install 'mmpose>=1.3.0' --no-deps\n"
        "     pip install json-tricks munkres matplotlib scipy pillow\n"
        "  2) Use POSE_BACKEND=template until SuperAnimal backend is wired\n"
        "  3) Run MMPose in Docker/Linux"
    )

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        score_threshold: float = KEYPOINT_SCORE_THRESHOLD,
    ):
        self.config_path = config_path or MMPOSE_CONFIG
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else MMPOSE_CHECKPOINT
        self.score_threshold = score_threshold
        self.device = device or self._default_device()
        self._model = None
        self._schema: list[str] | None = None
        self._check_dependencies()

    @classmethod
    def _check_dependencies(cls) -> None:
        try:
            import mmcv  # noqa: F401
            import mmpose  # noqa: F401
            from mmpose.apis.inference import init_model, inference_topdown  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"horse10_mmpose backend is missing dependencies: {exc}\n\n"
                f"{cls.MAC_INSTALL_HELP}"
            ) from exc

    @property
    def schema_name(self) -> str:
        return "animalpose"

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

        candidate_paths = [
            Path(MMPOSE_MODEL_DIR) / f"{MMPOSE_MODEL}.py",
            Path(MMPOSE_MODEL_DIR) / "td-hm_hrnet-w48_8xb64-210e_animalpose-256x256.py",
        ]

        try:
            import mmpose

            mim_configs = Path(mmpose.__file__).resolve().parent / ".mim" / "configs"
            candidate_paths.append(
                mim_configs
                / "animal_2d_keypoint/topdown_heatmap/animalpose"
                / f"{MMPOSE_MODEL}.py"
            )
        except ImportError as exc:
            raise ImportError(
                "MMPose is not installed. Install optional deps with:\n"
                "  pip install -r requirements-optional-mmpose.txt"
            ) from exc

        for candidate in candidate_paths:
            if candidate.is_file():
                return str(candidate)

        raise FileNotFoundError(
            "Could not find the MMPose config. "
            f"Expected {MMPOSE_MODEL}.py under {MMPOSE_MODEL_DIR}. "
            "Rebuild Docker or run:\n"
            "  mim download mmpose --config "
            f"{MMPOSE_MODEL} --dest {MMPOSE_MODEL_DIR}"
        )

    def _resolve_checkpoint_path(self, config_file: str) -> str:
        checkpoint_path = Path(self.checkpoint_path) if self.checkpoint_path else None

        if checkpoint_path and checkpoint_path.is_file():
            return str(checkpoint_path)

        if self.checkpoint_path and self.checkpoint_path.startswith(("http://", "https://")):
            return self.checkpoint_path

        config_dir = Path(config_file).resolve().parent
        checkpoints = sorted(config_dir.glob("*.pth"), key=lambda path: path.stat().st_size, reverse=True)
        if checkpoints:
            return str(checkpoints[0])

        raise FileNotFoundError(
            f"No checkpoint found for {config_file}. "
            f"Download weights with:\n"
            f"  mim download mmpose --config {MMPOSE_MODEL} --dest {config_dir}"
        )

    def _keypoint_schema(self) -> list[str]:
        if self._schema is not None:
            return self._schema

        model = self._load_model()
        id_to_name = model.dataset_meta.get("keypoint_id2name", {})
        if not id_to_name:
            raise ValueError("Loaded MMPose model is missing keypoint metadata")

        self._schema = [id_to_name[index] for index in range(len(id_to_name))]
        return self._schema

    def _load_model(self):
        if self._model is not None:
            return self._model

        from mmpose.apis.inference import init_model

        config_file = self._resolve_config_path()
        checkpoint_file = self._resolve_checkpoint_path(config_file)

        self._model = init_model(
            config_file,
            checkpoint_file,
            device=self.device,
        )
        return self._model

    def estimate(self, frame, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        from mmpose.apis.inference import inference_topdown

        model = self._load_model()
        schema = self._keypoint_schema()
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
                        schema,
                        score_threshold=self.score_threshold,
                    ),
                )
            )

        return enriched
