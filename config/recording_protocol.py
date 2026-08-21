"""Recommended recording protocol for trainer-submitted workout videos.

Hard requirements are minimal; the system scores whatever is submitted and
reports analysis confidence rather than rejecting uploads.
"""

from __future__ import annotations

RECOMMENDED_PROTOCOL = {
    "required": [
        "No digital zoom",
        "No rotation or orientation changes",
        "Keep the entire horse visible in frame",
        "Do not intentionally slow down or speed up the video",
    ],
    "strongly_preferred": [
        "Camera roughly perpendicular to the horse's path for side-view analysis",
        "Camera relatively stationary or smooth panning only",
        "Good, even lighting",
        "Horse occupies a reasonable portion of the frame",
        "Minimal obstruction or occlusion",
        "Avoid shooting through fences when possible",
        "Camera at approximately mid-body height",
    ],
    "trainer_message": (
        "For the most accurate analysis, record from a stationary camera perpendicular "
        "to the horse's path, without zooming or rotating. We'll analyze whatever you "
        "submit and report how confident we are in the results."
    ),
}

# Factor weights for overall recording quality (sum to 1.0).
RECORDING_QUALITY_WEIGHTS = {
    "motion_blur": 0.30,
    "horse_visibility": 0.25,
    "occlusion": 0.20,
    "camera_motion": 0.15,
    "resolution": 0.10,
}
