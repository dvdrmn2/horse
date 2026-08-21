"""Horse-relative coordinate systems with view-aware reliability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from config.landmark_spec import ViewName
from models.landmark import Landmark, LandmarkSet


class BodyFrameQuality(str, Enum):
    RELIABLE = "reliable"
    LOW_CONFIDENCE = "low_confidence"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BodyFrame:
    """Side-style body frame: longitudinal axis from withers toward tail_head."""

    withers: tuple[float, float]
    tail_head: tuple[float, float]
    axis_length: float
    longitudinal_unit: tuple[float, float]
    lateral_unit: tuple[float, float]
    quality: BodyFrameQuality = BodyFrameQuality.RELIABLE
    quality_reason: str = "ok"

    @classmethod
    def from_points(
        cls,
        withers: tuple[float, float],
        tail_head: tuple[float, float],
        *,
        min_axis_length: float = 1.0,
        bbox: Sequence[float] | None = None,
        view: ViewName = "unknown",
    ) -> BodyFrame | None:
        axis_x = tail_head[0] - withers[0]
        axis_y = tail_head[1] - withers[1]
        axis_length = math.hypot(axis_x, axis_y)
        if axis_length < min_axis_length:
            return None

        unit_x = axis_x / axis_length
        unit_y = axis_y / axis_length
        quality, reason = assess_side_body_frame_quality(
            axis_length=axis_length,
            longitudinal_unit=(unit_x, unit_y),
            bbox=bbox,
            view=view,
        )
        return cls(
            withers=withers,
            tail_head=tail_head,
            axis_length=axis_length,
            longitudinal_unit=(unit_x, unit_y),
            lateral_unit=(-unit_y, unit_x),
            quality=quality,
            quality_reason=reason,
        )

    @classmethod
    def from_landmarks(
        cls,
        landmarks: LandmarkSet,
        *,
        min_confidence: float = 0.0,
        bbox: Sequence[float] | None = None,
        view: ViewName = "unknown",
    ) -> BodyFrame | None:
        withers = landmarks.get("withers")
        tail = landmarks.get("tail_head")
        if withers is None or tail is None:
            return None
        if withers.x is None or withers.y is None or tail.x is None or tail.y is None:
            return None
        if not withers.visible or not tail.visible:
            return None
        if min_confidence > 0:
            if (withers.confidence or 0) < min_confidence or (tail.confidence or 0) < min_confidence:
                return None
        return cls.from_points(
            (withers.x, withers.y),
            (tail.x, tail.y),
            bbox=bbox,
            view=view,
        )

    @property
    def is_usable(self) -> bool:
        return self.quality is not BodyFrameQuality.UNAVAILABLE

    def to_body(self, x: float, y: float) -> tuple[float, float] | None:
        if not self.is_usable:
            return None

        rel_x = x - self.withers[0]
        rel_y = y - self.withers[1]
        longitudinal = rel_x * self.longitudinal_unit[0] + rel_y * self.longitudinal_unit[1]
        lateral = rel_x * self.lateral_unit[0] + rel_y * self.lateral_unit[1]
        return (longitudinal / self.axis_length, lateral / self.axis_length)

    def landmark_to_body(self, landmark: Landmark) -> tuple[float, float] | None:
        if landmark.x is None or landmark.y is None:
            return None
        return self.to_body(landmark.x, landmark.y)

    def to_video(self, longitudinal: float, lateral: float) -> tuple[float, float]:
        scale = self.axis_length
        x = (
            self.withers[0]
            + (longitudinal * scale) * self.longitudinal_unit[0]
            + (lateral * scale) * self.lateral_unit[0]
        )
        y = (
            self.withers[1]
            + (longitudinal * scale) * self.longitudinal_unit[1]
            + (lateral * scale) * self.lateral_unit[1]
        )
        return (x, y)


def _bbox_diagonal(bbox: Sequence[float] | None) -> float:
    if bbox is None:
        return 1.0
    x1, y1, x2, y2 = bbox
    return max(1.0, math.hypot(x2 - x1, y2 - y1))


def assess_side_body_frame_quality(
    *,
    axis_length: float,
    longitudinal_unit: tuple[float, float],
    bbox: Sequence[float] | None,
    view: ViewName,
) -> tuple[BodyFrameQuality, str]:
    """Decide whether withers→tail is a trustworthy in-plane longitudinal axis."""

    bbox_diag = _bbox_diagonal(bbox)
    in_plane = abs(longitudinal_unit[0])
    axis_ratio = axis_length / bbox_diag

    if view in {"rear", "front"} and in_plane < 0.45:
        return BodyFrameQuality.UNAVAILABLE, f"side body frame not supported for {view} view"

    if axis_length < bbox_diag * 0.18:
        return BodyFrameQuality.UNAVAILABLE, "withers-tail axis foreshortened in image"

    if in_plane < 0.45:
        return BodyFrameQuality.UNAVAILABLE, "longitudinal axis mostly out of image plane"

    if axis_ratio < 0.28:
        return BodyFrameQuality.LOW_CONFIDENCE, "short projected body axis"

    if in_plane < 0.62:
        return BodyFrameQuality.LOW_CONFIDENCE, "weak in-plane longitudinal component"

    return BodyFrameQuality.RELIABLE, "ok"


def body_velocity(
    long_a: float,
    lat_a: float,
    long_b: float,
    lat_b: float,
) -> float:
    return math.hypot(long_b - long_a, lat_b - lat_a)


def _view_for_body_frame(horse: dict) -> ViewName:
    """Recompute view from landmarks instead of using stale stored frame view."""

    from landmarks.view import _estimate_view_from_horse
    from models.horse import HorseInstance
    from models.landmark_types import LandmarkSource, LandmarkStatus

    landmarks = []
    for index, item in enumerate(horse.get("landmarks", [])):
        landmarks.append(
            Landmark(
                name=item["name"],
                index=index,
                x=item.get("x"),
                y=item.get("y"),
                confidence=item.get("confidence"),
                status=LandmarkStatus(item.get("status", "unavailable")),
                source=LandmarkSource(item.get("source", "none")),
                visible=item.get("visible", False),
            )
        )

    instance = HorseInstance(
        track_id=horse.get("track_id", 0),
        bbox=horse.get("bbox", [0, 0, 1, 1]),
        detection_confidence=horse.get("detection_confidence", 1.0),
        class_id=0,
        landmarks=LandmarkSet(
            landmarks=landmarks,
            schema=tuple(item["name"] for item in horse.get("landmarks", [])),
        ),
    )
    return _estimate_view_from_horse(instance).view


def extract_body_points(
    frames: Sequence[dict],
    track_id: int,
    landmark_name: str,
) -> list[dict]:
    points: list[dict] = []

    for frame in frames:
        horse = next(
            (item for item in frame.get("horses", []) if item.get("track_id") == track_id),
            None,
        )
        if horse is None:
            continue

        landmarks = horse.get("landmarks", [])
        landmark = next((item for item in landmarks if item.get("name") == landmark_name), None)
        if landmark is None or landmark.get("x") is None or landmark.get("y") is None:
            continue
        if landmark.get("status") in {"unmapped", "unavailable", "not_visible"}:
            continue

        withers = next((item for item in landmarks if item.get("name") == "withers"), None)
        tail = next((item for item in landmarks if item.get("name") == "tail_head"), None)
        body_long = None
        body_lat = None
        body_frame_quality = BodyFrameQuality.UNAVAILABLE.value
        body_frame_reason = "missing withers or tail"
        if (
            withers
            and tail
            and withers.get("x") is not None
            and tail.get("x") is not None
        ):
            frame_axes = BodyFrame.from_points(
                (withers["x"], withers["y"]),
                (tail["x"], tail["y"]),
                bbox=horse.get("bbox"),
                view=_view_for_body_frame(horse),
            )
            if frame_axes is not None:
                body_frame_quality = frame_axes.quality.value
                body_frame_reason = frame_axes.quality_reason
                coords = frame_axes.to_body(landmark["x"], landmark["y"])
                if coords is not None:
                    body_long, body_lat = coords

        points.append(
            {
                "frame": frame.get("frame_index"),
                "timestamp_sec": frame.get("timestamp_sec"),
                "x": landmark["x"],
                "y": landmark["y"],
                "body_long": body_long,
                "body_lat": body_lat,
                "body_frame_quality": body_frame_quality,
                "body_frame_reason": body_frame_reason,
                "confidence": landmark.get("confidence"),
                "effective_confidence": landmark.get("effective_confidence"),
                "status": landmark.get("status"),
                "outlier": landmark.get("outlier", False),
                "bbox": horse.get("bbox"),
                "view": frame.get("view"),
            }
        )

    return points
