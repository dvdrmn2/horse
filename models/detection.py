from dataclasses import dataclass

@dataclass
class Detection:
    bbox: list[float]
    confidence: float
    class_id: int
