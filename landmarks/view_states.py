"""View taxonomy: measurable views vs classification states."""

from __future__ import annotations

from config.landmark_spec import ViewName

# Discrete camera views the pipeline can estimate.
MEASURABLE_VIEWS: tuple[ViewName, ...] = ("side", "front", "rear", "oblique")

# Confidence states — not views. Downstream measurement code must gate on these.
VIEW_CLASSIFICATION_STATES = ("confident", "ambiguous", "insufficient")


def is_measurable_view(view: str) -> bool:
    return view in MEASURABLE_VIEWS


def effective_view_for_measurement(view: str, classification: str) -> ViewName:
    """Return the view used for view-dependent landmark scoring.

    Insufficient and ambiguous frames must not be treated as a reliable view,
    even if a distribution peak exists.
    """

    if classification != "confident":
        return "unknown"
    if not is_measurable_view(view):
        return "unknown"
    return view  # type: ignore[return-value]


def view_is_usable_for_measurement(classification: str) -> bool:
    return classification == "confident"
