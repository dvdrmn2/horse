from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from landmarks.view_features import EvidenceStrength

if TYPE_CHECKING:
    from landmarks.view import ViewFeatures, ViewScores

PRIMARY_VIEWS = ("side", "front", "rear")


@dataclass(frozen=True)
class CueEvidence:
    strength: EvidenceStrength
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"strength": self.strength, "detail": self.detail}


@dataclass(frozen=True)
class DirectionEvidence:
    overall: EvidenceStrength
    cues: dict[str, CueEvidence] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "cues": {name: cue.as_dict() for name, cue in self.cues.items()},
        }


def _strength_from_score(score: float, *, strong: float = 0.55, weak: float = 0.2) -> EvidenceStrength:
    if score >= strong:
        return "strong"
    if score >= weak:
        return "weak"
    return "unavailable"


def _overall_from_cues(
    cues: dict[str, CueEvidence],
    *,
    anchor_cues: tuple[str, ...] = (),
) -> EvidenceStrength:
    strengths = [cue.strength for cue in cues.values()]
    if not strengths or all(strength == "unavailable" for strength in strengths):
        return "unavailable"

    strong_count = sum(1 for strength in strengths if strength == "strong")
    moderate_count = sum(1 for strength in strengths if strength == "moderate")
    weak_count = sum(1 for strength in strengths if strength == "weak")

    has_anchor = True
    if anchor_cues:
        has_anchor = any(
            cues.get(name) is not None and cues[name].strength in {"strong", "moderate"}
            for name in anchor_cues
        )

    if anchor_cues and not has_anchor:
        if strong_count >= 2:
            return "moderate"
        if strong_count == 1 or moderate_count >= 2:
            return "moderate"
        if weak_count >= 1:
            return "weak"
        return "unavailable"

    if strong_count >= 2:
        return "strong"
    if strong_count == 1 or moderate_count >= 2 or (strong_count >= 1 and moderate_count >= 1):
        return "moderate"
    if weak_count >= 1:
        return "weak"
    return "unavailable"


def assess_view_evidence(features: "ViewFeatures", scores: "ViewScores") -> dict[str, DirectionEvidence]:
    """Summarize per-direction evidence strength without changing view scores."""

    side_cues: dict[str, CueEvidence] = {}
    if features.body_horizontal_span.available and features.body_vertical_span.available:
        horizontal = features.body_horizontal_span.value or 0.0
        vertical = max(features.body_vertical_span.value or 1.0, 1.0)
        elongation = horizontal / vertical
        if elongation > 0.9 and horizontal >= 30:
            side_cues["body_elongation"] = CueEvidence("strong", f"{horizontal:.0f}x{vertical:.0f}px axis")
        else:
            side_cues["body_elongation"] = CueEvidence("weak", f"elongation={elongation:.2f}")
    else:
        side_cues["body_elongation"] = CueEvidence("unavailable", "insufficient body-axis landmarks")

    if features.bbox_aspect > 1.1:
        side_cues["bbox_aspect"] = CueEvidence("weak", f"aspect={features.bbox_aspect:.2f}")
    else:
        side_cues["bbox_aspect"] = CueEvidence("unavailable", "compact bbox")

    front_cues: dict[str, CueEvidence] = {}
    if features.eye_horizontal_sep.available and features.eye_vertical_sep.available:
        eye_ratio = (features.eye_horizontal_sep.value or 0.0) / max(features.eye_vertical_sep.value or 1.0, 1.0)
        nose_centered = features.nose_centered.available and features.nose_centered.value == 1.0
        if eye_ratio > 1.3 and nose_centered:
            front_cues["head_geometry"] = CueEvidence("strong", f"eye_ratio={eye_ratio:.2f}")
        elif eye_ratio > 1.0:
            front_cues["head_geometry"] = CueEvidence("weak", f"eye_ratio={eye_ratio:.2f}")
        else:
            front_cues["head_geometry"] = CueEvidence("weak", "eyes visible but not front-facing")
    else:
        front_cues["head_geometry"] = CueEvidence("unavailable", "both eyes not visible")

    rear_cues: dict[str, CueEvidence] = {}
    from landmarks.rear_scoring import evaluate_rear_cues

    for cue in evaluate_rear_cues(features):
        rear_cues[cue.name] = CueEvidence(cue.strength, cue.detail)

    return {
        "side": DirectionEvidence(
            _overall_from_cues(side_cues, anchor_cues=("body_elongation",)),
            side_cues,
        ),
        "front": DirectionEvidence(
            _overall_from_cues(front_cues, anchor_cues=("head_geometry",)),
            front_cues,
        ),
        "rear": DirectionEvidence(
            _overall_from_cues(rear_cues, anchor_cues=("stifle_separation",)),
            rear_cues,
        ),
    }


ViewClassification = Literal["confident", "ambiguous", "insufficient"]

AMBIGUOUS_MARGIN = 0.12
CONFIDENT_PEAK = 0.55


@dataclass(frozen=True)
class ViewClassificationResult:
    classification: ViewClassification
    view_label: str
    dominant_view: str
    confidence: float
    ambiguous_with: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "view_label": self.view_label,
            "dominant_view": self.dominant_view,
            "confidence": self.confidence,
            "ambiguous_with": list(self.ambiguous_with),
            "reason": self.reason,
        }


def resolve_view_classification(
    distribution: dict[str, float],
    evidence: dict[str, DirectionEvidence],
    *,
    raw_scores: "ViewScores | None" = None,
) -> ViewClassificationResult:
    """Turn a distribution into confident, ambiguous, or insufficient view labels."""

    if distribution.get("unknown", 0.0) >= 1.0:
        return ViewClassificationResult(
            classification="insufficient",
            view_label="unknown",
            dominant_view="unknown",
            confidence=1.0,
            reason="insufficient view cues",
        )

    ordered = sorted(
        ((name, distribution.get(name, 0.0)) for name in ("side", "front", "rear", "oblique")),
        key=lambda item: item[1],
        reverse=True,
    )
    dominant, peak = ordered[0]
    runner_up, runner_value = ordered[1]

    primary_peak = max(distribution.get(name, 0.0) for name in PRIMARY_VIEWS)
    primary_evidence = [evidence[name].overall for name in PRIMARY_VIEWS]
    has_primary_evidence = any(strength in {"strong", "moderate"} for strength in primary_evidence)

    if dominant == "oblique" and not has_primary_evidence:
        return ViewClassificationResult(
            classification="insufficient",
            view_label="insufficient_evidence",
            dominant_view="unknown",
            confidence=peak,
            reason="no measurable side/front/rear evidence; distribution peak is not a reliable view",
        )

    if peak - runner_value < AMBIGUOUS_MARGIN and runner_value >= 0.15:
        pair = tuple(sorted((dominant, runner_up)))
        label = "/".join(pair)
        return ViewClassificationResult(
            classification="ambiguous",
            view_label=label,
            dominant_view=dominant,
            confidence=peak,
            ambiguous_with=(runner_up,),
            reason=f"close distribution: {dominant}={peak:.2f}, {runner_up}={runner_value:.2f}",
        )

    if peak < CONFIDENT_PEAK and dominant in PRIMARY_VIEWS:
        direction = evidence.get(dominant)
        if direction and direction.overall not in {"strong", "moderate"}:
            return ViewClassificationResult(
                classification="ambiguous",
                view_label=f"{dominant}?",
                dominant_view=dominant,
                confidence=peak,
                reason=f"low-confidence {dominant} ({direction.overall} evidence)",
            )

    return ViewClassificationResult(
        classification="confident",
        view_label=dominant,
        dominant_view=dominant,
        confidence=peak,
        reason=f"stable {dominant} view",
    )
