from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from config.landmark_spec import ViewName
from models.horse import HorseInstance

PRIMARY_VIEWS: tuple[ViewName, ...] = ("side", "front", "rear")
ALL_VIEWS: tuple[ViewName, ...] = ("side", "front", "rear", "oblique", "unknown")


@dataclass(frozen=True)
class ViewScores:
    """Continuous view estimate in [0, 1] before normalization."""

    side: float = 0.0
    front: float = 0.0
    rear: float = 0.0
    oblique: float = 0.0

    def normalized(self) -> dict[str, float]:
        values = {
            "side": max(0.0, self.side),
            "front": max(0.0, self.front),
            "rear": max(0.0, self.rear),
            "oblique": max(0.0, self.oblique),
        }
        total = sum(values.values())
        if total <= 1e-9:
            return {key: 0.0 for key in values} | {"unknown": 1.0}
        return {key: value / total for key, value in values.items()}

    def dominant(self) -> tuple[ViewName, float]:
        distribution = self.normalized()
        if distribution.get("unknown", 0.0) >= 1.0:
            return "unknown", 1.0
        best_view = max(("side", "front", "rear", "oblique"), key=lambda name: distribution.get(name, 0.0))
        return best_view, distribution[best_view]

    def __add__(self, other: ViewScores) -> ViewScores:
        return ViewScores(
            side=self.side + other.side,
            front=self.front + other.front,
            rear=self.rear + other.rear,
            oblique=self.oblique + other.oblique,
        )

    def __truediv__(self, divisor: float) -> ViewScores:
        if divisor <= 0:
            return ViewScores()
        return ViewScores(
            side=self.side / divisor,
            front=self.front / divisor,
            rear=self.rear / divisor,
            oblique=self.oblique / divisor,
        )


@dataclass(frozen=True)
class ViewFeatures:
    visible_count: int = 0
    bbox_aspect: float = 1.0
    bbox_width: float = 1.0
    body_horizontal_span: float = 0.0
    body_vertical_span: float = 0.0
    eye_horizontal_sep: float = 0.0
    eye_vertical_sep: float = 1.0
    nose_centered: bool = False
    head_landmarks: int = 0
    tail_withers_visible: bool = False
    stifle_sep: float = 0.0
  # diagnostic strings for debugging
    cues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ViewEstimate:
    view: ViewName
    confidence: float
    scores: ViewScores
    distribution: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.distribution:
            object.__setattr__(self, "distribution", self.scores.normalized())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(value: float, *, center: float = 0.0, scale: float = 1.0) -> float:
    if scale <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(value - center) / scale))


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


BILATERAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_stifle", "right_stifle"),
    ("left_elbow", "right_elbow"),
    ("left_carpus", "right_carpus"),
    ("left_front_hoof", "right_front_hoof"),
    ("left_hind_hoof", "right_hind_hoof"),
)


@dataclass(frozen=True)
class ViewDiagnosticSnapshot:
    """Human-readable feature dump for view-classification debugging."""

    frame_index: int | None
    features: dict[str, float]
    raw_scores: dict[str, float]
    distribution: dict[str, float]
    dominant_view: str
    dominant_confidence: float
    cues: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [f"Frame {self.frame_index if self.frame_index is not None else '?'}", "", "Features", "--------"]
        for key, value in self.features.items():
            lines.append(f"{key:24s} {value:.4f}" if isinstance(value, float) else f"{key:24s} {value}")
        lines.extend(["", "Scores", "------"])
        for key in ("side", "oblique", "rear", "front"):
            lines.append(f"{key:24s} {self.distribution.get(key, 0.0):.2f}")
        if self.cues:
            lines.extend(["", f"Cues: {'; '.join(self.cues)}"])
        return "\n".join(lines)


def _bilateral_symmetry(horse: HorseInstance, bbox_width: float, bbox_height: float) -> float | None:
    if horse.landmarks is None:
        return None

    x1, y1, x2, y2 = horse.bbox
    center_x = (x1 + x2) / 2.0
    pair_scores: list[float] = []

    for left_name, right_name in BILATERAL_PAIRS:
        left = _visible_point(horse, left_name)
        right = _visible_point(horse, right_name)
        if left is None or right is None:
            continue

        left_offset = abs(left[0] - center_x)
        right_offset = abs(right[0] - center_x)
        horizontal_balance = 1.0 - min(1.0, abs(left_offset - right_offset) / max(bbox_width * 0.5, 1.0))
        vertical_balance = 1.0 - min(1.0, abs(left[1] - right[1]) / max(bbox_height * 0.5, 1.0))
        pair_scores.append(0.6 * horizontal_balance + 0.4 * vertical_balance)

    if not pair_scores:
        return None
    return sum(pair_scores) / len(pair_scores)


def build_view_diagnostics(
    horse: HorseInstance,
    *,
    frame_index: int | None = None,
) -> ViewDiagnosticSnapshot:
    """Extract diagnostic features and scores without changing the classifier."""

    features = extract_view_features(horse)
    scores = compute_view_scores(features)
    estimate = classify_view_scores(scores)
    distribution = scores.normalized()

    x1, y1, x2, y2 = horse.bbox
    bbox_height = max(1.0, y2 - y1)
    symmetry = _bilateral_symmetry(horse, features.bbox_width, bbox_height)

    body_length = max(features.body_vertical_span, bbox_height * 0.6)
    body_width = max(features.body_horizontal_span, features.bbox_width * 0.4)
    width_length_ratio = body_width / max(body_length, 1.0)

    tail_visible = 1.0 if _visible_point(horse, "tail_head") else 0.0
    withers_visible = 1.0 if _visible_point(horse, "withers") else 0.0
    head_visibility = features.head_landmarks / 3.0
    stifle_ratio = features.stifle_sep / max(features.bbox_width, 1.0)

    diagnostic_features = {
        "stifle_separation_px": features.stifle_sep,
        "stifle_separation_ratio": stifle_ratio,
        "body_width_px": body_width,
        "body_length_px": body_length,
        "width_length_ratio": width_length_ratio,
        "bbox_width_px": features.bbox_width,
        "bbox_aspect": features.bbox_aspect,
        "head_visibility": head_visibility,
        "tail_visibility": tail_visible,
        "withers_visibility": withers_visible,
        "tail_head_absent": 1.0 if tail_visible and head_visibility == 0.0 else 0.0,
        "left_right_symmetry": symmetry if symmetry is not None else float("nan"),
        "visible_landmarks": float(features.visible_count),
        "body_horizontal_span_px": features.body_horizontal_span,
        "body_vertical_span_px": features.body_vertical_span,
    }

    return ViewDiagnosticSnapshot(
        frame_index=frame_index,
        features=diagnostic_features,
        raw_scores={
            "side": scores.side,
            "front": scores.front,
            "rear": scores.rear,
            "oblique": scores.oblique,
        },
        distribution=distribution,
        dominant_view=estimate.view,
        dominant_confidence=estimate.confidence,
        cues=features.cues,
    )


def extract_view_features(horse: HorseInstance) -> ViewFeatures:
    if horse.landmarks is None:
        return ViewFeatures()

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

    eye_horizontal_sep = abs(left_eye[0] - right_eye[0]) if left_eye and right_eye else 0.0
    eye_vertical_sep = (
        max(abs(left_eye[1] - right_eye[1]), 1.0) if left_eye and right_eye else 1.0
    )
    nose_centered = False
    if left_eye and right_eye and nose:
        eye_sep = abs(left_eye[0] - right_eye[0])
        nose_mid_x = (left_eye[0] + right_eye[0]) / 2.0
        nose_centered = abs(nose[0] - nose_mid_x) < eye_sep * 0.35

    head_landmarks = sum(1 for point in (nose, left_eye, right_eye) if point)
    tail_withers_visible = bool(tail and withers)
    stifle_sep = abs(left_stifle[0] - right_stifle[0]) if left_stifle and right_stifle else 0.0

    cues: list[str] = []
    if nose_centered and eye_horizontal_sep > eye_vertical_sep * 1.2:
        cues.append("front-facing head geometry")
    if horizontal_span > max(vertical_span * 0.85, 30):
        cues.append("elongated body axis")
    if aspect > 1.1:
        cues.append(f"wide bbox aspect {aspect:.2f}")
    if tail_withers_visible and head_landmarks == 0:
        cues.append("tail/withers without head")
    if stifle_sep > bbox_width * 0.28 and horizontal_span < max(vertical_span * 1.1, 40):
        cues.append("wide stifle separation with compact body")

    return ViewFeatures(
        visible_count=visible_count,
        bbox_aspect=aspect,
        bbox_width=bbox_width,
        body_horizontal_span=horizontal_span,
        body_vertical_span=vertical_span,
        eye_horizontal_sep=eye_horizontal_sep,
        eye_vertical_sep=eye_vertical_sep,
        nose_centered=nose_centered,
        head_landmarks=head_landmarks,
        tail_withers_visible=tail_withers_visible,
        stifle_sep=stifle_sep,
        cues=tuple(cues),
    )


def compute_view_scores(features: ViewFeatures) -> ViewScores:
    """Map geometric features to continuous view scores."""

    if features.visible_count == 0:
        return ViewScores()

    elongation_ratio = features.body_horizontal_span / max(features.body_vertical_span, 1.0)
    elongated_body = _sigmoid(elongation_ratio, center=0.9, scale=0.15)
    elongated_body *= _sigmoid(features.body_horizontal_span, center=30.0, scale=12.0)
    wide_bbox = _sigmoid(features.bbox_aspect, center=1.15, scale=0.08)
    side_score = max(elongated_body, wide_bbox * 0.85)

    eye_ratio = features.eye_horizontal_sep / max(features.eye_vertical_sep, 1.0)
    front_face = _sigmoid(eye_ratio, center=1.4, scale=0.2)
    if features.nose_centered:
        front_face = min(1.0, front_face + 0.25)
    front_score = front_face if features.head_landmarks >= 2 else 0.0

    stifle_ratio = features.stifle_sep / max(features.bbox_width, 1.0)
    compact_body = 1.0 - _sigmoid(
        features.body_horizontal_span / max(features.body_vertical_span, 1.0),
        center=0.95,
        scale=0.12,
    )
    rear_stifle = _sigmoid(stifle_ratio, center=0.28, scale=0.05) * compact_body
    rear_tail = 0.85 if features.tail_withers_visible and features.head_landmarks == 0 else 0.0
    rear_score = max(rear_stifle, rear_tail)
    if features.head_landmarks >= 2 and features.nose_centered:
        rear_score *= 0.35
    elif features.head_landmarks >= 1 and rear_tail > 0:
        rear_score = max(rear_score, rear_tail)

    if rear_stifle > 0.35 or rear_tail > 0:
        side_score *= 0.55

    primary_scores = sorted((side_score, front_score, rear_score))
    peak = primary_scores[-1]
    runner_up = primary_scores[-2]
    ambiguity = _clamp01(1.0 - (peak - runner_up) / max(peak, 0.15))
    landmark_coverage = _sigmoid(features.visible_count, center=4.0, scale=1.2)
    oblique_score = ambiguity * landmark_coverage
    if peak >= 0.6:
        oblique_score *= 0.35
    if peak < 0.2 and features.visible_count >= 3:
        oblique_score = max(oblique_score, 0.25)

    return ViewScores(
        side=side_score,
        front=front_score,
        rear=rear_score,
        oblique=oblique_score,
    )


def classify_view_scores(scores: ViewScores) -> ViewEstimate:
    view, confidence = scores.dominant()
    distribution = scores.normalized()

    if view == "unknown":
        return ViewEstimate(
            view="unknown",
            confidence=1.0,
            scores=scores,
            distribution=distribution,
            reason="insufficient view cues",
        )

    reason_parts = []
    ordered = sorted(
        ((name, distribution.get(name, 0.0)) for name in ("side", "front", "rear", "oblique")),
        key=lambda item: item[1],
        reverse=True,
    )
    for name, value in ordered[:2]:
        if value >= 0.08:
            reason_parts.append(f"{name}={value:.2f}")

    return ViewEstimate(
        view=view,
        confidence=confidence,
        scores=scores,
        distribution=distribution,
        reason=", ".join(reason_parts) if reason_parts else view,
    )


def average_view_scores(scores: list[ViewScores]) -> ViewScores:
    if not scores:
        return ViewScores()
    total = ViewScores()
    for item in scores:
        total += item
    return total / len(scores)


class ViewSmoother:
    """Temporal smoothing over continuous view scores."""

    def __init__(self, window: int = 15) -> None:
        self._history: deque[ViewScores] = deque(maxlen=window)

    def update(self, raw_scores: ViewScores) -> ViewEstimate:
        if sum(raw_scores.normalized().values()) > 0:
            self._history.append(raw_scores)
        if not self._history:
            return classify_view_scores(raw_scores)
        smoothed = average_view_scores(list(self._history))
        return classify_view_scores(smoothed)


def estimate_view_from_horse(horse: HorseInstance) -> ViewEstimate:
    features = extract_view_features(horse)
    scores = compute_view_scores(features)
    estimate = classify_view_scores(scores)
    if features.cues and estimate.view != "unknown":
        return ViewEstimate(
            view=estimate.view,
            confidence=estimate.confidence,
            scores=estimate.scores,
            distribution=estimate.distribution,
            reason="; ".join(features.cues[:2]),
        )
    return estimate


def estimate_camera_view(horses: list[HorseInstance]) -> ViewName:
    return estimate_camera_view_detailed(horses).view


def estimate_camera_view_detailed(horses: list[HorseInstance]) -> ViewEstimate:
    if not horses:
        return ViewEstimate("unknown", 1.0, ViewScores(), reason="no horses")

    ranked = sorted(
        horses,
        key=lambda horse: len(horse.landmarks.visible_landmarks) if horse.landmarks else 0,
        reverse=True,
    )

    for horse in ranked:
        estimate = estimate_view_from_horse(horse)
        if estimate.view != "unknown":
            return estimate

    return ViewEstimate("unknown", 1.0, ViewScores(), reason="insufficient landmarks")


# Backward-compatible alias used by debug tooling.
_estimate_view_from_horse = estimate_view_from_horse


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


def aggregate_view_distribution(estimates: list[ViewEstimate]) -> dict[str, float]:
    if not estimates:
        return {"unknown": 1.0}

    totals = {"side": 0.0, "front": 0.0, "rear": 0.0, "oblique": 0.0}
    for estimate in estimates:
        for key in totals:
            totals[key] += estimate.distribution.get(key, 0.0)

    total = sum(totals.values())
    if total <= 0:
        return {"unknown": 1.0}
    return {key: value / total for key, value in totals.items()}


def dominant_view_from_distribution(distribution: dict[str, float]) -> tuple[ViewName, float]:
    if not distribution:
        return "unknown", 0.0
    best_view = max(distribution, key=distribution.get)
    return best_view, distribution[best_view]
