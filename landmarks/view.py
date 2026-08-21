from __future__ import annotations

import math
from dataclasses import dataclass

from config.landmark_spec import ViewName
from models.horse import HorseInstance


@dataclass(frozen=True)
class ViewEstimate:
    view: ViewName
    reason: str


def _visible_point(horse: HorseInstance, name: str) -> tuple[float, float] | None:
    if horse.landmarks is None:
        return None

    landmark = horse.landmarks.get(name)
    if landmark is None or landmark.x is None or landmark.y is None:
        return None
    if not landmark.visible:
        return None

    return landmark.x, landmark.y


def _bbox_aspect(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return width / height


def _bbox_width(bbox: list[float]) -> float:
    x1, _, x2, _ = bbox
    return max(1.0, x2 - x1)


def estimate_camera_view(horses: list[HorseInstance]) -> ViewName:
    return estimate_camera_view_detailed(horses).view


def estimate_camera_view_detailed(horses: list[HorseInstance]) -> ViewEstimate:
    if not horses:
        return ViewEstimate("unknown", "no horses")

    ranked = sorted(
        horses,
        key=lambda horse: len(horse.landmarks.visible_landmarks) if horse.landmarks else 0,
        reverse=True,
    )

    for horse in ranked:
        estimate = _estimate_view_from_horse(horse)
        if estimate.view != "unknown":
            return estimate

    return ViewEstimate("unknown", "insufficient landmarks")


def _estimate_view_from_horse(horse: HorseInstance) -> ViewEstimate:
    if horse.landmarks is None:
        return ViewEstimate("unknown", "no landmarks")

    left_eye = _visible_point(horse, "left_eye")
    right_eye = _visible_point(horse, "right_eye")
    nose = _visible_point(horse, "nose")
    tail = _visible_point(horse, "tail_head")
    withers = _visible_point(horse, "withers")
    neck = _visible_point(horse, "neck")
    left_stifle = _visible_point(horse, "left_stifle")
    right_stifle = _visible_point(horse, "right_stifle")

    visible_count = len(horse.landmarks.visible_landmarks)
    aspect = _bbox_aspect(horse.bbox)
    bbox_width = _bbox_width(horse.bbox)

    body_axis_points = [point for point in (nose, neck, withers, tail) if point]
    horizontal_span = 0.0
    vertical_span = 0.0
    if len(body_axis_points) >= 2:
        xs = [point[0] for point in body_axis_points]
        ys = [point[1] for point in body_axis_points]
        horizontal_span = max(xs) - min(xs)
        vertical_span = max(ys) - min(ys)

    if left_eye and right_eye and nose:
        eye_sep = abs(left_eye[0] - right_eye[0])
        eye_vert = max(abs(left_eye[1] - right_eye[1]), 1.0)
        nose_mid_x = (left_eye[0] + right_eye[0]) / 2.0
        nose_centered = abs(nose[0] - nose_mid_x) < eye_sep * 0.35
        if eye_sep > eye_vert * 1.4 and nose_centered:
            return ViewEstimate("front", "both eyes visible with centered nose")

    if len(body_axis_points) >= 2 and horizontal_span > max(vertical_span * 0.85, 30):
        return ViewEstimate(
            "side",
            f"elongated body axis ({horizontal_span:.0f}px horizontal vs {vertical_span:.0f}px vertical)",
        )

    if aspect > 1.15 and visible_count >= 4:
        return ViewEstimate("side", f"wide bbox aspect {aspect:.2f}")

    head_landmarks = sum(1 for point in (nose, left_eye, right_eye) if point)
    if tail and withers and head_landmarks == 0:
        return ViewEstimate("rear", "tail and withers visible without head landmarks")

    if left_stifle and right_stifle and tail:
        stifle_sep = abs(left_stifle[0] - right_stifle[0])
        body_not_elongated = horizontal_span < max(vertical_span * 1.1, 40)
        stifle_dominates = stifle_sep > bbox_width * 0.28
        if stifle_dominates and body_not_elongated:
            return ViewEstimate(
                "rear",
                f"wide stifle separation ({stifle_sep:.0f}px) with compact body axis",
            )

    if aspect > 1.0 and visible_count >= 3:
        return ViewEstimate("side", f"moderate bbox aspect {aspect:.2f}")

    if visible_count >= 5:
        return ViewEstimate("oblique", f"{visible_count} visible landmarks")

    return ViewEstimate("unknown", "insufficient view cues")


def estimate_identity_confidence(horse: HorseInstance, view: ViewName) -> float:
    if view == "unknown":
        return 1.0

    if view in {"front", "side"}:
        return 1.0

    if view == "rear":
        return 0.65

    if view == "oblique":
        return 0.85

    return 0.8
