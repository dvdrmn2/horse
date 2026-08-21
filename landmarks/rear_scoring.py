"""Evidence-aware rear view scoring.

Each cue is evaluated only when its underlying features are observable.
Missing features contribute nothing — values are never imputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from landmarks.view_evidence import _overall_from_cues, CueEvidence
from landmarks.view_features import EvidenceStrength

if TYPE_CHECKING:
    from landmarks.view import ViewFeatures


def _sigmoid(value: float, *, center: float = 0.0, scale: float = 1.0) -> float:
    if scale <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(value - center) / scale))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class RearCueContribution:
    """Per-cue rear evidence used for scoring and debugging."""

    name: str
    available: bool
    contribution: float
    strength: EvidenceStrength
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "contribution": round(self.contribution, 4),
            "strength": self.strength,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RearScoreBreakdown:
    """Internal rear scoring artifact — explains why a rear score was produced."""

    rear_score: float
    evidence_strength: EvidenceStrength
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    cues: tuple[RearCueContribution, ...] = ()
    head_penalty_applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "rear_score": round(self.rear_score, 4),
            "evidence_strength": self.evidence_strength,
            "evidence": self.evidence,
            "head_penalty_applied": self.head_penalty_applied,
        }

    def render(self) -> str:
        lines = [f"Rear score = {self.rear_score:.2f}", f"Evidence strength = {self.evidence_strength}", ""]
        for cue in self.cues:
            label = cue.name.replace("_", " ").title()
            if cue.available:
                lines.append(f"{label:24s} +{cue.contribution:.2f} ({cue.strength})")
            else:
                lines.append(f"{label:24s} unavailable")
        if self.head_penalty_applied:
            lines.append("")
            lines.append("Head penalty applied (front-facing head contradicts rear)")
        return "\n".join(lines)


def _evaluate_stifle_separation(features: "ViewFeatures") -> RearCueContribution:
    if not features.stifle_sep.available:
        return RearCueContribution(
            name="stifle_separation",
            available=False,
            contribution=0.0,
            strength="unavailable",
            detail="both stifles not visible",
        )

    stifle_px = features.stifle_sep.value or 0.0
    ratio = stifle_px / max(features.bbox_width, 1.0)
    contribution = _sigmoid(ratio, center=0.28, scale=0.05)

    if ratio > 0.28:
        strength: EvidenceStrength = "strong"
        detail = f"{stifle_px:.1f}px ({ratio:.2f} of bbox)"
    elif ratio > 0.15:
        strength = "moderate"
        detail = f"{stifle_px:.1f}px ({ratio:.2f} of bbox)"
    else:
        strength = "weak"
        detail = f"{stifle_px:.1f}px ({ratio:.2f} of bbox)"

    return RearCueContribution(
        name="stifle_separation",
        available=True,
        contribution=contribution,
        strength=strength,
        detail=detail,
    )


def _evaluate_tail_head_relationship(features: "ViewFeatures") -> RearCueContribution:
    if features.tail_withers_visible and features.head_landmarks_visible == 0:
        return RearCueContribution(
            name="tail_head_relationship",
            available=True,
            contribution=0.85,
            strength="strong",
            detail="tail/withers without head",
        )

    if features.tail_visible:
        return RearCueContribution(
            name="tail_head_relationship",
            available=True,
            contribution=0.2,
            strength="weak",
            detail="tail visible with partial head",
        )

    return RearCueContribution(
        name="tail_head_relationship",
        available=False,
        contribution=0.0,
        strength="unavailable",
        detail="tail not visible",
    )


def _evaluate_symmetry(features: "ViewFeatures") -> RearCueContribution:
    if not features.bilateral_symmetry.available:
        return RearCueContribution(
            name="symmetry",
            available=False,
            contribution=0.0,
            strength="unavailable",
            detail="no bilateral pairs",
        )

    symmetry = features.bilateral_symmetry.value or 0.0
    contribution = _sigmoid(symmetry, center=0.75, scale=0.08) * 0.35

    if symmetry >= 0.8:
        strength: EvidenceStrength = "strong"
    elif symmetry >= 0.65:
        strength = "moderate"
    elif symmetry >= 0.5:
        strength = "weak"
    else:
        strength = "weak"
        contribution = contribution * 0.5

    return RearCueContribution(
        name="symmetry",
        available=True,
        contribution=contribution,
        strength=strength,
        detail=f"{symmetry:.2f}",
    )


def _evaluate_body_geometry(features: "ViewFeatures") -> RearCueContribution:
    if not (features.body_horizontal_span.available and features.body_vertical_span.available):
        return RearCueContribution(
            name="body_geometry",
            available=False,
            contribution=0.0,
            strength="unavailable",
            detail="body axis unavailable",
        )

    horizontal = features.body_horizontal_span.value or 0.0
    vertical = max(features.body_vertical_span.value or 1.0, 1.0)
    compactness = horizontal / vertical
    compact_factor = 1.0 - _sigmoid(compactness, center=0.95, scale=0.12)

    if compact_factor < 0.1:
        return RearCueContribution(
            name="body_geometry",
            available=True,
            contribution=0.0,
            strength="weak",
            detail=f"elongated body width/length={compactness:.2f}",
        )

    contribution = compact_factor * 0.35
    strength: EvidenceStrength = "moderate" if compactness < 0.95 else "weak"
    return RearCueContribution(
        name="body_geometry",
        available=True,
        contribution=contribution,
        strength=strength,
        detail=f"width/length={compactness:.2f}",
    )


def evaluate_rear_cues(features: "ViewFeatures") -> tuple[RearCueContribution, ...]:
    return (
        _evaluate_stifle_separation(features),
        _evaluate_tail_head_relationship(features),
        _evaluate_symmetry(features),
        _evaluate_body_geometry(features),
    )


def _aggregate_available_contributions(cues: tuple[RearCueContribution, ...]) -> float:
    total = sum(cue.contribution for cue in cues if cue.available)
    return _clamp01(total)


def _apply_head_penalty(
    features: "ViewFeatures",
    rear_score: float,
    cues: tuple[RearCueContribution, ...],
) -> tuple[float, bool]:
    tail_cue = next(cue for cue in cues if cue.name == "tail_head_relationship")

    if (
        features.head_landmarks_visible >= 2
        and features.nose_centered.available
        and features.nose_centered.value == 1.0
    ):
        return rear_score * 0.35, True

    if features.head_landmarks_visible >= 1 and tail_cue.contribution >= 0.85:
        return max(rear_score, tail_cue.contribution), False

    return rear_score, False


def compute_rear_score_breakdown(features: "ViewFeatures") -> RearScoreBreakdown:
    """Score rear view from observable cues only."""

    cues = evaluate_rear_cues(features)
    raw_score = _aggregate_available_contributions(cues)
    rear_score, head_penalty_applied = _apply_head_penalty(features, raw_score, cues)

    cue_evidence = {
        cue.name: CueEvidence(cue.strength, cue.detail)
        for cue in cues
    }
    evidence_strength = _overall_from_cues(cue_evidence, anchor_cues=("stifle_separation",))

    evidence = {
        cue.name: {
            "available": cue.available,
            "contribution": round(cue.contribution, 4) if cue.available else 0.0,
        }
        for cue in cues
    }

    return RearScoreBreakdown(
        rear_score=rear_score,
        evidence_strength=evidence_strength,
        evidence=evidence,
        cues=cues,
        head_penalty_applied=head_penalty_applied,
    )
