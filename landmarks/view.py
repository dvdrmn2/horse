from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from config.landmark_spec import ViewName
from landmarks.view_evidence import (
    DirectionEvidence,
    assess_view_evidence,
    resolve_view_classification,
)
from landmarks.view_states import effective_view_for_measurement
from landmarks.rear_scoring import compute_rear_score_breakdown
from landmarks.view_features import FeatureValue
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
    bbox_height: float = 1.0
    body_horizontal_span: FeatureValue = field(default_factory=FeatureValue.unavailable)
    body_vertical_span: FeatureValue = field(default_factory=FeatureValue.unavailable)
    eye_horizontal_sep: FeatureValue = field(default_factory=FeatureValue.unavailable)
    eye_vertical_sep: FeatureValue = field(default_factory=FeatureValue.unavailable)
    nose_centered: FeatureValue = field(default_factory=FeatureValue.unavailable)
    head_landmarks_visible: int = 0
    head_landmarks_total: int = 3
    tail_visible: bool = False
    withers_visible: bool = False
    tail_withers_visible: bool = False
    stifle_sep: FeatureValue = field(default_factory=FeatureValue.unavailable)
    bilateral_symmetry: FeatureValue = field(default_factory=FeatureValue.unavailable)
    cues: tuple[str, ...] = ()

    def availability_summary(self) -> dict[str, bool]:
        return {
            "body_horizontal_span": self.body_horizontal_span.available,
            "body_vertical_span": self.body_vertical_span.available,
            "stifle_separation": self.stifle_sep.available,
            "eye_separation": self.eye_horizontal_sep.available,
            "nose_centered": self.nose_centered.available,
            "bilateral_symmetry": self.bilateral_symmetry.available,
            "tail_visible": self.tail_visible,
            "withers_visible": self.withers_visible,
        }


@dataclass(frozen=True)
class ViewEstimate:
    view: ViewName
    confidence: float
    scores: ViewScores
    distribution: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    classification: str = "confident"
    view_label: str = ""
    ambiguous_with: tuple[str, ...] = ()
    evidence: dict[str, dict] = field(default_factory=dict)
    score_breakdown: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.distribution:
            object.__setattr__(self, "distribution", self.scores.normalized())
        if not self.view_label:
            object.__setattr__(self, "view_label", self.view)


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
    features: dict[str, dict[str, Any]]
    availability: dict[str, bool]
    raw_scores: dict[str, float]
    distribution: dict[str, float]
    dominant_view: str
    dominant_confidence: float
    classification: str = "confident"
    view_label: str = ""
    evidence: dict[str, dict] = field(default_factory=dict)
    score_breakdown: dict[str, dict] = field(default_factory=dict)
    cues: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [f"Frame {self.frame_index if self.frame_index is not None else '?'}", "", "Features", "--------"]
        for key, payload in self.features.items():
            if isinstance(payload, dict):
                if payload.get("available"):
                    lines.append(f"{key:24s} {payload['value']:.4f}")
                else:
                    lines.append(f"{key:24s} unavailable")
            else:
                lines.append(f"{key:24s} {payload}")
        lines.extend(["", "Availability", "------------"])
        for key, available in self.availability.items():
            lines.append(f"{key:24s} {'yes' if available else 'no'}")
        lines.extend(["", "Scores", "------"])
        for key in ("side", "oblique", "rear", "front"):
            lines.append(f"{key:24s} {self.distribution.get(key, 0.0):.2f}")
        if self.cues:
            lines.extend(["", f"Cues: {'; '.join(self.cues)}"])
        if self.evidence:
            lines.extend(["", "Evidence", "--------"])
            for direction, payload in self.evidence.items():
                lines.append(f"{direction:8s} overall={payload.get('overall', '?'):10s}")
                for cue_name, cue in payload.get("cues", {}).items():
                    lines.append(f"           {cue_name}: {cue.get('strength')} ({cue.get('detail', '')})")
        rear_breakdown = self.score_breakdown.get("rear")
        if rear_breakdown:
            lines.extend(["", "Rear score breakdown", "--------------------"])
            lines.append(f"rear_score={rear_breakdown.get('rear_score', 0.0):.2f}  "
                         f"strength={rear_breakdown.get('evidence_strength', '?')}")
            for cue_name, cue in rear_breakdown.get("evidence", {}).items():
                if cue.get("available"):
                    lines.append(f"  {cue_name:24s} +{cue.get('contribution', 0.0):.2f}")
                else:
                    lines.append(f"  {cue_name:24s} unavailable")
        lines.extend(["", f"Classification: {self.classification} ({self.view_label})"])
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
    scores, score_breakdown = compute_view_scores(features)
    estimate = classify_view_scores(scores, features=features, score_breakdown=score_breakdown)
    distribution = scores.normalized()

    x1, y1, x2, y2 = horse.bbox
    bbox_height = max(1.0, y2 - y1)

    body_length = (
        features.body_vertical_span.value
        if features.body_vertical_span.available
        else bbox_height * 0.6
    )
    body_width = (
        features.body_horizontal_span.value
        if features.body_horizontal_span.available
        else features.bbox_width * 0.4
    )
    width_length_ratio = body_width / max(body_length, 1.0)
    stifle_ratio = (
        features.stifle_sep.value / max(features.bbox_width, 1.0)
        if features.stifle_sep.available and features.stifle_sep.value is not None
        else None
    )

    diagnostic_features = {
        "stifle_separation_px": features.stifle_sep.as_dict(),
        "stifle_separation_ratio": {
            "value": stifle_ratio,
            "available": features.stifle_sep.available,
        },
        "body_horizontal_span_px": features.body_horizontal_span.as_dict(),
        "body_vertical_span_px": features.body_vertical_span.as_dict(),
        "body_width_px": {
            "value": body_width,
            "available": features.body_horizontal_span.available,
        },
        "body_length_px": {
            "value": body_length,
            "available": features.body_vertical_span.available,
        },
        "width_length_ratio": {
            "value": width_length_ratio if features.body_horizontal_span.available and features.body_vertical_span.available else None,
            "available": features.body_horizontal_span.available and features.body_vertical_span.available,
        },
        "bbox_width_px": {"value": features.bbox_width, "available": True},
        "bbox_aspect": {"value": features.bbox_aspect, "available": True},
        "head_visibility": {
            "value": features.head_landmarks_visible / max(features.head_landmarks_total, 1),
            "available": True,
        },
        "tail_visibility": {"value": float(features.tail_visible), "available": True},
        "withers_visibility": {"value": float(features.withers_visible), "available": True},
        "tail_head_absent": {
            "value": float(features.tail_visible and features.head_landmarks_visible == 0),
            "available": features.tail_visible,
        },
        "left_right_symmetry": features.bilateral_symmetry.as_dict(),
        "visible_landmarks": {"value": float(features.visible_count), "available": True},
    }

    return ViewDiagnosticSnapshot(
        frame_index=frame_index,
        features=diagnostic_features,
        availability=features.availability_summary(),
        raw_scores={
            "side": scores.side,
            "front": scores.front,
            "rear": scores.rear,
            "oblique": scores.oblique,
        },
        distribution=distribution,
        dominant_view=estimate.view,
        dominant_confidence=estimate.confidence,
        classification=estimate.classification,
        view_label=estimate.view_label,
        evidence=estimate.evidence,
        score_breakdown=estimate.score_breakdown,
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
    x1, y1, x2, y2 = horse.bbox
    aspect = _bbox_aspect(horse.bbox)
    bbox_width = _bbox_width(horse.bbox)
    bbox_height = max(1.0, y2 - y1)

    body_axis_points = [point for point in (nose, neck, withers, tail) if point]
    body_horizontal_span = FeatureValue.unavailable()
    body_vertical_span = FeatureValue.unavailable()
    if len(body_axis_points) >= 2:
        xs = [point[0] for point in body_axis_points]
        ys = [point[1] for point in body_axis_points]
        body_horizontal_span = FeatureValue.measured(max(xs) - min(xs))
        body_vertical_span = FeatureValue.measured(max(ys) - min(ys))

    eye_horizontal_sep = FeatureValue.unavailable()
    eye_vertical_sep = FeatureValue.unavailable()
    nose_centered = FeatureValue.unavailable()
    if left_eye and right_eye:
        eye_horizontal_sep = FeatureValue.measured(abs(left_eye[0] - right_eye[0]))
        eye_vertical_sep = FeatureValue.measured(max(abs(left_eye[1] - right_eye[1]), 1.0))
        if nose:
            eye_sep = abs(left_eye[0] - right_eye[0])
            nose_mid_x = (left_eye[0] + right_eye[0]) / 2.0
            nose_centered = FeatureValue.measured(
                1.0 if abs(nose[0] - nose_mid_x) < eye_sep * 0.35 else 0.0
            )

    head_landmarks_visible = sum(1 for point in (nose, left_eye, right_eye) if point)
    tail_visible = tail is not None
    withers_visible = withers is not None
    tail_withers_visible = tail_visible and withers_visible

    stifle_sep = FeatureValue.unavailable()
    if left_stifle and right_stifle:
        stifle_sep = FeatureValue.measured(abs(left_stifle[0] - right_stifle[0]))

    symmetry_value = _bilateral_symmetry(horse, bbox_width, bbox_height)
    bilateral_symmetry = (
        FeatureValue.measured(symmetry_value)
        if symmetry_value is not None
        else FeatureValue.unavailable()
    )

    cues: list[str] = []
    if (
        nose_centered.available
        and nose_centered.value == 1.0
        and eye_horizontal_sep.available
        and eye_vertical_sep.available
        and eye_horizontal_sep.value > eye_vertical_sep.value * 1.2
    ):
        cues.append("front-facing head geometry")
    if (
        body_horizontal_span.available
        and body_vertical_span.available
        and body_horizontal_span.value > max(body_vertical_span.value * 0.85, 30)
    ):
        cues.append("elongated body axis")
    if aspect > 1.1:
        cues.append(f"wide bbox aspect {aspect:.2f}")
    if tail_withers_visible and head_landmarks_visible == 0:
        cues.append("tail/withers without head")
    if (
        stifle_sep.available
        and body_horizontal_span.available
        and body_vertical_span.available
        and stifle_sep.value > bbox_width * 0.28
        and body_horizontal_span.value < max(body_vertical_span.value * 1.1, 40)
    ):
        cues.append("wide stifle separation with compact body")
    if not stifle_sep.available and (left_stifle or right_stifle):
        cues.append("single stifle only — separation unavailable")

    return ViewFeatures(
        visible_count=visible_count,
        bbox_aspect=aspect,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        body_horizontal_span=body_horizontal_span,
        body_vertical_span=body_vertical_span,
        eye_horizontal_sep=eye_horizontal_sep,
        eye_vertical_sep=eye_vertical_sep,
        nose_centered=nose_centered,
        head_landmarks_visible=head_landmarks_visible,
        tail_visible=tail_visible,
        withers_visible=withers_visible,
        tail_withers_visible=tail_withers_visible,
        stifle_sep=stifle_sep,
        bilateral_symmetry=bilateral_symmetry,
        cues=tuple(cues),
    )


def compute_view_scores(features: ViewFeatures) -> tuple[ViewScores, dict[str, dict]]:
    """Map geometric features to continuous view scores.

    Unavailable features contribute no evidence — they are not treated as zero.
    Returns scores plus an internal per-direction score breakdown for debugging.
    """

    if features.visible_count == 0:
        return ViewScores(), {}

    side_score = 0.0
    if features.body_horizontal_span.available and features.body_vertical_span.available:
        horizontal_span = features.body_horizontal_span.value or 0.0
        vertical_span = max(features.body_vertical_span.value or 1.0, 1.0)
        elongation_ratio = horizontal_span / vertical_span
        elongated_body = _sigmoid(elongation_ratio, center=0.9, scale=0.15)
        elongated_body *= _sigmoid(horizontal_span, center=30.0, scale=12.0)
        side_score = max(side_score, elongated_body)

    wide_bbox = _sigmoid(features.bbox_aspect, center=1.15, scale=0.08)
    side_score = max(side_score, wide_bbox * 0.85)

    front_score = 0.0
    if (
        features.eye_horizontal_sep.available
        and features.eye_vertical_sep.available
        and features.head_landmarks_visible >= 2
    ):
        eye_ratio = (features.eye_horizontal_sep.value or 0.0) / max(
            features.eye_vertical_sep.value or 1.0, 1.0
        )
        front_face = _sigmoid(eye_ratio, center=1.4, scale=0.2)
        if features.nose_centered.available and features.nose_centered.value == 1.0:
            front_face = min(1.0, front_face + 0.25)
        front_score = front_face

    rear_breakdown = compute_rear_score_breakdown(features)
    rear_score = rear_breakdown.rear_score

    if rear_score > 0.35:
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
    ), {"rear": rear_breakdown.as_dict()}


def classify_view_scores(
    scores: ViewScores,
    *,
    features: ViewFeatures | None = None,
    score_breakdown: dict[str, dict] | None = None,
) -> ViewEstimate:
    distribution = scores.normalized()
    view, confidence = scores.dominant()

    if view == "unknown":
        return ViewEstimate(
            view="unknown",
            confidence=1.0,
            scores=scores,
            distribution=distribution,
            classification="insufficient",
            view_label="unknown",
            reason="insufficient view cues",
        )

    evidence = assess_view_evidence(features, scores) if features is not None else {
        name: DirectionEvidence("unavailable", {}) for name in ("side", "front", "rear")
    }
    evidence_dict = {name: direction.as_dict() for name, direction in evidence.items()}
    resolved = resolve_view_classification(distribution, evidence, raw_scores=scores)

    reason = resolved.reason
    if features and features.cues:
        reason = f"{reason}; {'; '.join(features.cues[:2])}"

    return ViewEstimate(
        view=effective_view_for_measurement(resolved.dominant_view, resolved.classification),
        confidence=resolved.confidence,
        scores=scores,
        distribution=distribution,
        classification=resolved.classification,
        view_label=resolved.view_label,
        ambiguous_with=resolved.ambiguous_with,
        evidence=evidence_dict,
        score_breakdown=score_breakdown or {},
        reason=reason,
    )


def average_view_scores(scores: list[ViewScores]) -> ViewScores:
    if not scores:
        return ViewScores()
    total = ViewScores()
    for item in scores:
        total += item
    return total / len(scores)


class ViewSmoother:
    """Temporal smoothing over continuous view scores.

    TODO: make evidence-aware — average scores together with per-frame evidence
    strength so insufficient frames do not drift into oblique via score-only means.
  """

    def __init__(self, window: int = 15) -> None:
        self._history: deque[ViewScores] = deque(maxlen=window)

    def update(self, raw_scores: ViewScores, *, features: ViewFeatures | None = None, score_breakdown: dict[str, dict] | None = None) -> ViewEstimate:
        if sum(raw_scores.normalized().values()) > 0:
            self._history.append(raw_scores)
        if not self._history:
            return classify_view_scores(raw_scores, features=features, score_breakdown=score_breakdown)
        smoothed = average_view_scores(list(self._history))
        return classify_view_scores(smoothed, features=features, score_breakdown=score_breakdown)


def estimate_view_from_horse(horse: HorseInstance) -> ViewEstimate:
    features = extract_view_features(horse)
    scores, score_breakdown = compute_view_scores(features)
    return classify_view_scores(scores, features=features, score_breakdown=score_breakdown)


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
