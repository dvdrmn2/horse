"""Per-landmark tracking profiles for QC and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MotionLevel = Literal["static", "moderate", "dynamic"]
ToleranceLevel = Literal["low", "medium", "high"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class LandmarkProfile:
    motion: MotionLevel
    occlusion_tolerance: ToleranceLevel
    identity_swap_risk: RiskLevel


LANDMARK_PROFILES: dict[str, LandmarkProfile] = {
    "nose": LandmarkProfile("static", "low", "low"),
    "left_eye": LandmarkProfile("static", "low", "medium"),
    "right_eye": LandmarkProfile("static", "low", "medium"),
    "poll": LandmarkProfile("static", "low", "low"),
    "neck": LandmarkProfile("moderate", "medium", "low"),
    "withers": LandmarkProfile("static", "low", "low"),
    "spine_mid": LandmarkProfile("static", "medium", "low"),
    "croup": LandmarkProfile("static", "medium", "low"),
    "tail_head": LandmarkProfile("moderate", "medium", "low"),
    "left_shoulder": LandmarkProfile("moderate", "medium", "medium"),
    "right_shoulder": LandmarkProfile("moderate", "medium", "medium"),
    "left_elbow": LandmarkProfile("moderate", "medium", "medium"),
    "right_elbow": LandmarkProfile("moderate", "medium", "medium"),
    "left_carpus": LandmarkProfile("dynamic", "high", "high"),
    "right_carpus": LandmarkProfile("dynamic", "high", "high"),
    "left_fetlock": LandmarkProfile("dynamic", "high", "high"),
    "right_fetlock": LandmarkProfile("dynamic", "high", "high"),
    "left_front_hoof": LandmarkProfile("dynamic", "high", "high"),
    "right_front_hoof": LandmarkProfile("dynamic", "high", "high"),
    "left_hip": LandmarkProfile("moderate", "medium", "medium"),
    "right_hip": LandmarkProfile("moderate", "medium", "medium"),
    "left_stifle": LandmarkProfile("moderate", "medium", "medium"),
    "right_stifle": LandmarkProfile("moderate", "medium", "medium"),
    "left_hock": LandmarkProfile("dynamic", "high", "high"),
    "right_hock": LandmarkProfile("dynamic", "high", "high"),
    "left_hind_fetlock": LandmarkProfile("dynamic", "high", "high"),
    "right_hind_fetlock": LandmarkProfile("dynamic", "high", "high"),
    "left_hind_hoof": LandmarkProfile("dynamic", "high", "high"),
    "right_hind_hoof": LandmarkProfile("dynamic", "high", "high"),
}


def get_landmark_profile(name: str) -> LandmarkProfile:
    if name in LANDMARK_PROFILES:
        return LANDMARK_PROFILES[name]
    if name.endswith("_hoof") or name.endswith("_fetlock"):
        return LandmarkProfile("dynamic", "high", "high")
    if name.endswith("_carpus") or name.endswith("_hock"):
        return LandmarkProfile("dynamic", "high", "high")
    return LandmarkProfile("moderate", "medium", "medium")
