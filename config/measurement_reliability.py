"""Measurement-specific reliability priorities.

The session-level recording_quality.overall score is a human-facing summary using
default weights in config.recording_protocol. Individual measurements should use
their own factor priorities when weighting frame usability.
"""

from __future__ import annotations

# Future hook: measurement name -> ordered recording factor priorities.
MEASUREMENT_RECORDING_PRIORITIES: dict[str, tuple[str, ...]] = {
    "hoof_trajectory": ("motion_blur", "occlusion"),
    "joint_angle": ("resolution", "motion_blur"),
    "symmetry": ("occlusion", "camera_motion"),
    "stride_length": ("camera_motion", "horse_visibility"),
    "head_carriage": ("resolution", "motion_blur"),
    "hind_limb_symmetry": ("occlusion", "motion_blur"),
    "temporal_gait": ("motion_blur", "camera_motion"),
}

# Future hook: measurement name -> confidence dimensions to emphasize.
MEASUREMENT_CONFIDENCE_PRIORITIES: dict[str, tuple[str, ...]] = {
    "hoof_trajectory": ("pose_confidence", "recording_quality"),
    "joint_angle": ("pose_confidence", "view_confidence"),
    "symmetry": ("view_confidence", "pose_confidence"),
    "stride_length": ("view_confidence", "tracking_confidence"),
    "head_carriage": ("view_confidence", "recording_quality"),
    "hind_limb_symmetry": ("view_confidence", "pose_confidence"),
    "temporal_gait": ("tracking_confidence", "recording_quality"),
}
