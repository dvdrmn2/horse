from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AnalysisLevel = Literal["high", "moderate", "low", "insufficient"]


@dataclass(frozen=True)
class AnalysisConfidence:
    level: AnalysisLevel
    label: str
    recording_quality: float
    pose_confidence: float
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "label": self.label,
            "recording_quality": self.recording_quality,
            "pose_confidence": self.pose_confidence,
            "notes": list(self.notes),
        }


def classify_analysis_confidence(
    recording_quality: float,
    pose_confidence: float,
    *,
    factor_notes: dict[str, float] | None = None,
) -> AnalysisConfidence:
    """Combine recording and pose scores into an overall analysis confidence."""

    notes: list[str] = []
    factors = factor_notes or {}

    if factors.get("motion_blur", 100.0) < 60.0:
        notes.append("motion blur detected")
    if factors.get("camera_motion", 100.0) < 60.0:
        notes.append("camera movement detected")
    if factors.get("horse_visibility", 100.0) < 50.0:
        notes.append("horse occupies a small or clipped portion of the frame")
    if factors.get("occlusion", 100.0) < 55.0:
        notes.append("occlusion or partial landmark visibility")
    if pose_confidence < 60.0:
        notes.append("pose extraction confidence is limited")

    # Conservative combination: weak recording or weak pose drags the session down.
    combined = min(
        recording_quality,
        pose_confidence,
        (recording_quality * 0.55) + (pose_confidence * 0.45),
    )

    if combined >= 80.0:
        level: AnalysisLevel = "high"
        label = "High"
    elif combined >= 60.0:
        level = "moderate"
        label = "Moderate"
    elif combined >= 40.0:
        level = "low"
        label = "Low"
    else:
        level = "insufficient"
        label = "Insufficient"

    if notes and level != "high":
        label = f"{label} — {notes[0]}"

    return AnalysisConfidence(
        level=level,
        label=label,
        recording_quality=recording_quality,
        pose_confidence=pose_confidence,
        notes=tuple(notes),
    )
