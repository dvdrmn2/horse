from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from config.landmark_spec import ViewName

AnalysisLevel = Literal["high", "moderate", "low", "insufficient"]


@dataclass(frozen=True)
class ConfidenceDimensions:
    recording_quality: float
    pose_confidence: float
    view_confidence: float
    tracking_confidence: float | None = None
    gait_confidence: float | None = None

    def available_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {
            "recording_quality": self.recording_quality,
            "pose_confidence": self.pose_confidence,
            "view_confidence": self.view_confidence,
        }
        if self.tracking_confidence is not None:
            scores["tracking_confidence"] = self.tracking_confidence
        if self.gait_confidence is not None:
            scores["gait_confidence"] = self.gait_confidence
        return scores

    def as_dict(self) -> dict[str, float | None]:
        return {
            "recording_quality": round(self.recording_quality, 1),
            "pose_confidence": round(self.pose_confidence, 1),
            "view_confidence": round(self.view_confidence, 1),
            "tracking_confidence": (
                round(self.tracking_confidence, 1) if self.tracking_confidence is not None else None
            ),
            "gait_confidence": (
                round(self.gait_confidence, 1) if self.gait_confidence is not None else None
            ),
        }


@dataclass(frozen=True)
class PrimaryLimitation:
    dimension: str
    score: float
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 1),
            "message": self.message,
        }


@dataclass(frozen=True)
class AnalysisConfidenceReport:
    level: AnalysisLevel
    label: str
    dimensions: ConfidenceDimensions
    primary_limitation: PrimaryLimitation
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "label": self.label,
            "dimensions": self.dimensions.as_dict(),
            "primary_limitation": self.primary_limitation.as_dict(),
            "notes": list(self.notes),
        }


def _level_from_scores(scores: dict[str, float]) -> tuple[AnalysisLevel, str]:
    if not scores:
        return "insufficient", "INSUFFICIENT"

    combined = min(
        min(scores.values()),
        sum(scores.values()) / len(scores),
    )

    if combined >= 80.0:
        return "high", "HIGH"
    if combined >= 60.0:
        return "moderate", "MODERATE"
    if combined >= 40.0:
        return "low", "LOW"
    return "insufficient", "INSUFFICIENT"


def _recording_limitation(factors: dict[str, float]) -> str:
    ordered = [
        ("motion_blur", "motion blur"),
        ("camera_motion", "camera movement"),
        ("horse_visibility", "horse visibility"),
        ("occlusion", "occlusion"),
        ("resolution", "resolution"),
    ]
    weakest_name = None
    weakest_score = 101.0
    for key, label in ordered:
        score = factors.get(key, 100.0)
        if score < weakest_score:
            weakest_score = score
            weakest_name = label
    if weakest_name and weakest_score < 70.0:
        return weakest_name
    return "recording limitations"


def _view_limitation(dominant_view: ViewName, distribution: dict[str, float]) -> str:
    if not distribution:
        return "view classification unavailable"

    ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    dominant = ordered[0]
    if dominant[1] >= 0.85:
        return f"stable {dominant[0]} view"

    if len(ordered) > 1 and ordered[1][1] >= 0.15:
        return f"{dominant[0]}/{ordered[1][0]} ambiguity"

    return f"limited {dominant[0]} view confidence"


def _dimension_limitation(
    dimension: str,
    *,
    recording_factors: dict[str, float] | None = None,
    dominant_view: ViewName = "unknown",
    view_distribution: dict[str, float] | None = None,
    track_count: int = 1,
) -> str:
    if dimension == "recording_quality":
        return _recording_limitation(recording_factors or {})
    if dimension == "pose_confidence":
        return "pose extraction confidence is limited"
    if dimension == "view_confidence":
        return _view_limitation(dominant_view, view_distribution or {})
    if dimension == "tracking_confidence":
        if track_count > 1:
            return "multi-horse or fragmented tracking"
        return "tracking continuity is limited"
    if dimension == "gait_confidence":
        return "gait classification not yet available"
    return "confidence is limited"


def build_analysis_confidence(
    dimensions: ConfidenceDimensions,
    *,
    recording_factors: dict[str, float] | None = None,
    dominant_view: ViewName = "unknown",
    view_distribution: dict[str, float] | None = None,
    track_count: int = 1,
) -> AnalysisConfidenceReport:
    """Build a multi-dimensional confidence report with an explicit primary limitation."""

    scores = dimensions.available_scores()
    level, label = _level_from_scores(scores)

    notes: list[str] = []
    factors = recording_factors or {}
    if factors.get("motion_blur", 100.0) < 60.0:
        notes.append("motion blur detected")
    if factors.get("camera_motion", 100.0) < 60.0:
        notes.append("camera movement detected")
    if factors.get("horse_visibility", 100.0) < 50.0:
        notes.append("horse occupies a small or clipped portion of the frame")
    if factors.get("occlusion", 100.0) < 55.0:
        notes.append("occlusion or partial landmark visibility")
    if dimensions.pose_confidence < 60.0:
        notes.append("pose extraction confidence is limited")
    if dimensions.view_confidence < 70.0:
        notes.append(_view_limitation(dominant_view, view_distribution or {}))
    if dimensions.tracking_confidence is not None and dimensions.tracking_confidence < 70.0:
        notes.append("tracking continuity is limited")

    weakest_dimension = min(scores, key=scores.get)
    limitation_message = _dimension_limitation(
        weakest_dimension,
        recording_factors=factors,
        dominant_view=dominant_view,
        view_distribution=view_distribution,
        track_count=track_count,
    )
    human_dimension = weakest_dimension.replace("_", " ").title()
    primary = PrimaryLimitation(
        dimension=weakest_dimension,
        score=scores[weakest_dimension],
        message=f"{human_dimension} — {limitation_message}",
    )

    return AnalysisConfidenceReport(
        level=level,
        label=label,
        dimensions=dimensions,
        primary_limitation=primary,
        notes=tuple(notes),
    )


# Backward-compatible alias for earlier callers.
AnalysisConfidence = AnalysisConfidenceReport


def classify_analysis_confidence(
    recording_quality: float,
    pose_confidence: float,
    *,
    factor_notes: dict[str, float] | None = None,
    view_confidence: float | None = None,
    tracking_confidence: float | None = None,
    dominant_view: ViewName = "unknown",
    view_distribution: dict[str, float] | None = None,
    track_count: int = 1,
) -> AnalysisConfidenceReport:
    return build_analysis_confidence(
        ConfidenceDimensions(
            recording_quality=recording_quality,
            pose_confidence=pose_confidence,
            view_confidence=view_confidence if view_confidence is not None else 100.0,
            tracking_confidence=tracking_confidence,
            gait_confidence=None,
        ),
        recording_factors=factor_notes,
        dominant_view=dominant_view,
        view_distribution=view_distribution,
        track_count=track_count,
    )
