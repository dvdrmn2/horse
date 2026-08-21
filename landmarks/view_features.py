from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class FeatureValue:
    """A measured value with explicit observability.

    ``available=False`` means we cannot observe this feature — not that it is zero.
    """

    value: float | None = None
    available: bool = False

    @classmethod
    def unavailable(cls) -> FeatureValue:
        return cls(value=None, available=False)

    @classmethod
    def measured(cls, value: float) -> FeatureValue:
        return cls(value=float(value), available=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "available": self.available,
        }

    def display(self) -> str:
        if not self.available:
            return "unavailable"
        return f"{self.value:.4f}"


EvidenceStrength = Literal["strong", "moderate", "weak", "unavailable"]
