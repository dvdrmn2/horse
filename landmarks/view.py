from __future__ import annotations

from config.landmark_spec import ViewName
from models.horse import HorseInstance


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


def estimate_camera_view(horses: list[HorseInstance]) -> ViewName:
    """Estimate scene camera view from visible landmarks on any horse in frame."""

    if not horses:
        return "unknown"

    ranked = sorted(
        horses,
        key=lambda horse: len(horse.landmarks.visible_landmarks) if horse.landmarks else 0,
        reverse=True,
    )

    for horse in ranked:
        view = _estimate_view_from_horse(horse)
        if view != "unknown":
            return view

    return "unknown"


def _estimate_view_from_horse(horse: HorseInstance) -> ViewName:
    if horse.landmarks is None:
        return "unknown"

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

    if left_eye and right_eye and nose:
        eye_sep = abs(left_eye[0] - right_eye[0])
        eye_vert = max(abs(left_eye[1] - right_eye[1]), 1.0)
        nose_mid_x = (left_eye[0] + right_eye[0]) / 2.0
        nose_centered = abs(nose[0] - nose_mid_x) < eye_sep * 0.35
        if eye_sep > eye_vert * 1.4 and nose_centered:
            return "front"

    if tail and withers and not nose and not left_eye and not right_eye:
        return "rear"

    if left_stifle and right_stifle and tail:
        stifle_sep = abs(left_stifle[0] - right_stifle[0])
        if stifle_sep > 15:
            return "rear"

    body_axis_points = [point for point in (nose, neck, withers, tail) if point]
    if len(body_axis_points) >= 2:
        xs = [point[0] for point in body_axis_points]
        ys = [point[1] for point in body_axis_points]
        horizontal_span = max(xs) - min(xs)
        vertical_span = max(ys) - min(ys)
        if horizontal_span > max(vertical_span * 0.85, 30):
            return "side"

    if aspect > 1.15 and visible_count >= 4:
        return "side"

    if visible_count >= 5:
        return "oblique"

    if aspect > 1.0:
        return "side"

    return "oblique" if visible_count >= 3 else "unknown"


def estimate_identity_confidence(horse: HorseInstance, view: ViewName) -> float:
    """Lower confidence in left/right labels when the view is ambiguous."""

    if view == "unknown":
        return 1.0

    if view in {"front", "side"}:
        return 1.0

    if view == "rear":
        return 0.65

    if view == "oblique":
        return 0.85

    return 0.8
