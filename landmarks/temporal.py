"""Temporal motion analysis and landmark-aware quality control helpers."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Iterable, Sequence

from landmarks.body_frame import BodyFrame
from config.landmark_spec import LANDMARK_DEFINITIONS


class MotionProfile(str, Enum):
    """How we expect a landmark to move over time."""

    STATIC = "static"
    MODERATE = "moderate"
    DYNAMIC = "dynamic"


STATIC_LANDMARKS = frozenset(
    {
        "nose",
        "left_eye",
        "right_eye",
        "poll",
        "withers",
        "spine_mid",
        "croup",
    }
)

MODERATE_LANDMARKS = frozenset(
    {
        "neck",
        "tail_head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hip",
        "right_hip",
        "left_stifle",
        "right_stifle",
    }
)

DYNAMIC_LANDMARKS = frozenset(
    {
        "left_carpus",
        "right_carpus",
        "left_fetlock",
        "right_fetlock",
        "left_front_hoof",
        "right_front_hoof",
        "left_hock",
        "right_hock",
        "left_hind_fetlock",
        "right_hind_fetlock",
        "left_hind_hoof",
        "right_hind_hoof",
    }
)


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    frame: int | None = None


@dataclass(frozen=True)
class KinematicSample:
    frame: int | None
    velocity: float
    acceleration: float
    jerk: float
    direction_deg: float | None
    prediction_residual: float


@dataclass(frozen=True)
class TrajectoryMetrics:
    frames: int
    motion_profile: MotionProfile
    mean_confidence: float | None
    mean_velocity: float | None
    median_velocity: float | None
    p95_velocity: float | None
    max_velocity: float | None
    mean_acceleration: float | None
    p95_acceleration: float | None
    max_acceleration: float | None
    direction_changes: int
    spike_count: int
    legacy_outlier_rate: float | None
    bbox_mean_velocity: float | None
    bbox_p95_velocity: float | None


def get_motion_profile(landmark_name: str) -> MotionProfile:
    from config.landmark_profiles import get_landmark_profile

    profile = get_landmark_profile(landmark_name)
    if profile.motion == "static":
        return MotionProfile.STATIC
    if profile.motion == "dynamic":
        return MotionProfile.DYNAMIC
    return MotionProfile.MODERATE


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def compensating_delta(
    x: float,
    y: float,
    last_x: float,
    last_y: float,
    bbox: Sequence[float] | None,
    last_bbox: Sequence[float] | None,
) -> tuple[float, float]:
    dx = x - last_x
    dy = y - last_y
    if bbox is None or last_bbox is None:
        return dx, dy

    cx0 = (last_bbox[0] + last_bbox[2]) / 2.0
    cy0 = (last_bbox[1] + last_bbox[3]) / 2.0
    cx1 = (bbox[0] + bbox[2]) / 2.0
    cy1 = (bbox[1] + bbox[3]) / 2.0
    return dx - (cx1 - cx0), dy - (cy1 - cy0)


def compensating_jump(
    x: float,
    y: float,
    last_x: float,
    last_y: float,
    bbox: Sequence[float] | None,
    last_bbox: Sequence[float] | None,
) -> float:
    dx, dy = compensating_delta(x, y, last_x, last_y, bbox, last_bbox)
    return math.hypot(dx, dy)


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def _angle_deg(vx: float, vy: float) -> float | None:
    if vx == 0 and vy == 0:
        return None
    return math.degrees(math.atan2(vy, vx))


def normalize_to_bbox(x: float, y: float, bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return ((x - x1) / width, (y - y1) / height)


def normalize_to_body_axis(
    x: float,
    y: float,
    withers: Point2D,
    tail: Point2D,
) -> tuple[float, float] | None:
    frame = BodyFrame.from_points((withers.x, withers.y), (tail.x, tail.y))
    if frame is None:
        return None
    return frame.to_body(x, y)


def compute_kinematics(points: Sequence[Point2D]) -> list[KinematicSample]:
    if len(points) < 2:
        return []

    samples: list[KinematicSample] = []
    velocities: list[float] = []
    directions: list[float | None] = []

    for index in range(1, len(points)):
        velocity = _distance(points[index], points[index - 1])
        velocities.append(velocity)

        vx = points[index].x - points[index - 1].x
        vy = points[index].y - points[index - 1].y
        directions.append(_angle_deg(vx, vy))

    for index in range(1, len(points)):
        velocity = velocities[index - 1]
        acceleration = abs(velocities[index - 1] - velocities[index - 2]) if index >= 2 else 0.0
        jerk = 0.0
        if index >= 3:
            prev_accel = abs(velocities[index - 2] - velocities[index - 3])
            jerk = abs(acceleration - prev_accel)

        prediction_residual = 0.0
        if index >= 2:
            predicted = Point2D(
                x=points[index - 1].x + (points[index - 1].x - points[index - 2].x),
                y=points[index - 1].y + (points[index - 1].y - points[index - 2].y),
                frame=points[index].frame,
            )
            prediction_residual = _distance(points[index], predicted)

        samples.append(
            KinematicSample(
                frame=points[index].frame,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                direction_deg=directions[index - 1],
                prediction_residual=prediction_residual,
            )
        )

    return samples


def count_direction_changes(samples: Sequence[KinematicSample], min_angle_deg: float = 75.0) -> int:
    directions = [sample.direction_deg for sample in samples if sample.direction_deg is not None]
    if len(directions) < 2:
        return 0

    changes = 0
    for index in range(1, len(directions)):
        delta = abs((directions[index] - directions[index - 1] + 180) % 360 - 180)
        if delta >= min_angle_deg:
            changes += 1
    return changes


def detect_spike_indices(
    samples: Sequence[KinematicSample],
    profile: MotionProfile,
) -> list[int]:
    """Return sample indices that look like implausible teleports, not fast limb motion."""

    if len(samples) < 3:
        return []

    residuals = [sample.prediction_residual for sample in samples if sample.prediction_residual > 0]
    accel_values = [sample.acceleration for sample in samples if sample.acceleration > 0]
    if not residuals:
        return []

    residual_floor = 12.0 if profile is MotionProfile.DYNAMIC else 8.0
    residual_multiplier = 4.0 if profile is MotionProfile.DYNAMIC else 2.5
    accel_multiplier = 4.5 if profile is MotionProfile.DYNAMIC else 3.0

    baseline_residual = max(median(residuals), residual_floor)
    baseline_accel = max(median(accel_values) if accel_values else 0.0, 4.0)

    spike_indices: list[int] = []
    for index, sample in enumerate(samples):
        residual_spike = sample.prediction_residual > max(
            baseline_residual * residual_multiplier,
            residual_floor * 2,
        )
        accel_spike = sample.acceleration > max(
            baseline_accel * accel_multiplier,
            baseline_accel + 20.0,
        )

        if profile is MotionProfile.STATIC:
            if residual_spike or sample.velocity > baseline_residual * 6:
                spike_indices.append(index)
            continue

        if profile is MotionProfile.MODERATE:
            if residual_spike and accel_spike:
                spike_indices.append(index)
            continue

        # Dynamic landmarks: require both a large misprediction and a local accel spike.
        if residual_spike and accel_spike and sample.velocity > 8.0:
            spike_indices.append(index)

    return spike_indices


def is_streaming_outlier(
    x: float,
    y: float,
    history: deque[tuple[float, float, float]],
    *,
    profile: MotionProfile,
    max_jump: float,
    bbox: Sequence[float] | None = None,
    last_bbox: Sequence[float] | None = None,
) -> bool:
    """Online outlier test used by the frame-by-frame quality controller."""

    if len(history) < 1:
        return False

    last_x, last_y, _ = history[-1]
    jump = compensating_jump(x, y, last_x, last_y, bbox, last_bbox)

    if profile is MotionProfile.STATIC:
        return jump > max_jump * 0.35

    if profile is MotionProfile.MODERATE:
        if len(history) < 2:
            return jump > max_jump * 0.55
        return _streaming_spike(
            history,
            x,
            y,
            jump,
            bbox=bbox,
            last_bbox=last_bbox,
            residual_multiplier=3.0,
            accel_multiplier=3.5,
        )

    if profile is MotionProfile.DYNAMIC:
        # Distal landmarks can move quickly and nonlinearly; reject only obvious teleports.
        if len(history) < 3:
            return False
        return _streaming_spike(
            history,
            x,
            y,
            jump,
            bbox=bbox,
            last_bbox=last_bbox,
            residual_multiplier=6.0,
            accel_multiplier=7.0,
            min_jump=20.0,
            min_residual=35.0,
        )

    if len(history) < 2:
        return False


def _streaming_spike(
    history: deque[tuple[float, float, float]],
    x: float,
    y: float,
    jump: float,
    *,
    bbox: Sequence[float] | None,
    last_bbox: Sequence[float] | None,
    residual_multiplier: float,
    accel_multiplier: float,
    min_jump: float = 8.0,
    min_residual: float = 18.0,
) -> bool:
    if len(history) < 2:
        return False

    x1, y1, _ = history[-1]
    x0, y0, _ = history[-2]
    pred_x = x1 + (x1 - x0)
    pred_y = y1 + (y1 - y0)
    residual = compensating_jump(x, y, pred_x, pred_y, bbox, last_bbox)

    dx, dy = compensating_delta(x, y, x1, y1, bbox, last_bbox)
    v_curr = (dx, dy)
    pdx, pdy = compensating_delta(x1, y1, x0, y0, last_bbox, None)
    v_prev = (pdx, pdy)
    acceleration = math.hypot(v_curr[0] - v_prev[0], v_curr[1] - v_prev[1])

    residuals = []
    accels = []
    hist = list(history)
    for index in range(2, len(hist)):
        px0, py0, _ = hist[index - 2]
        px1, py1, _ = hist[index - 1]
        px2, py2, _ = hist[index]
        pred_px = px1 + (px1 - px0)
        pred_py = py1 + (py1 - py0)
        residuals.append(math.hypot(px2 - pred_px, py2 - pred_py))
        vp = (px1 - px0, py1 - py0)
        vc = (px2 - px1, py2 - py1)
        accels.append(math.hypot(vc[0] - vp[0], vc[1] - vp[1]))

    baseline_residual = max(median(residuals) if residuals else 8.0, 8.0)
    baseline_accel = max(median(accels) if accels else 4.0, 4.0)

    residual_spike = residual > max(baseline_residual * residual_multiplier, min_residual)
    accel_spike = acceleration > max(baseline_accel * accel_multiplier, baseline_accel + 18.0)
    return residual_spike and accel_spike and jump > min_jump


def summarize_trajectory(
    points: Sequence[dict],
    landmark_name: str,
) -> TrajectoryMetrics:
    profile = get_motion_profile(landmark_name)
    if not points:
        return TrajectoryMetrics(
            frames=0,
            motion_profile=profile,
            mean_confidence=None,
            mean_velocity=None,
            median_velocity=None,
            p95_velocity=None,
            max_velocity=None,
            mean_acceleration=None,
            p95_acceleration=None,
            max_acceleration=None,
            direction_changes=0,
            spike_count=0,
            legacy_outlier_rate=None,
            bbox_mean_velocity=None,
            bbox_p95_velocity=None,
        )

    point_objs = [
        Point2D(x=float(point["x"]), y=float(point["y"]), frame=point.get("frame"))
        for point in points
    ]
    samples = compute_kinematics(point_objs)
    velocities = [sample.velocity for sample in samples]
    accelerations = [sample.acceleration for sample in samples]
    confidences = [point["confidence"] for point in points if point.get("confidence") is not None]
    outliers = [point for point in points if point.get("outlier")]
    spike_indices = detect_spike_indices(samples, profile)

    bbox_velocities: list[float] = []
    for index in range(1, len(points)):
        bbox_a = points[index - 1].get("bbox")
        bbox_b = points[index].get("bbox")
        if not bbox_a or not bbox_b:
            continue
        rel_a = normalize_to_bbox(points[index - 1]["x"], points[index - 1]["y"], bbox_a)
        rel_b = normalize_to_bbox(points[index]["x"], points[index]["y"], bbox_b)
        bbox_velocities.append(math.hypot(rel_b[0] - rel_a[0], rel_b[1] - rel_a[1]))

    return TrajectoryMetrics(
        frames=len(points),
        motion_profile=profile,
        mean_confidence=sum(confidences) / len(confidences) if confidences else None,
        mean_velocity=sum(velocities) / len(velocities) if velocities else None,
        median_velocity=median(velocities) if velocities else None,
        p95_velocity=_percentile(velocities, 0.95),
        max_velocity=max(velocities) if velocities else None,
        mean_acceleration=sum(accelerations) / len(accelerations) if accelerations else None,
        p95_acceleration=_percentile(accelerations, 0.95),
        max_acceleration=max(accelerations) if accelerations else None,
        direction_changes=count_direction_changes(samples),
        spike_count=len(spike_indices),
        legacy_outlier_rate=len(outliers) / len(points),
        bbox_mean_velocity=sum(bbox_velocities) / len(bbox_velocities) if bbox_velocities else None,
        bbox_p95_velocity=_percentile(bbox_velocities, 0.95),
    )


def format_metrics(metrics: TrajectoryMetrics) -> Iterable[str]:
    yield f"  motion profile      : {metrics.motion_profile.value}"
    if metrics.frames == 0:
        yield "  (no trajectory points)"
        return

    if metrics.mean_confidence is not None:
        yield f"  mean confidence     : {metrics.mean_confidence:.3f}"
    if metrics.mean_velocity is not None:
        yield f"  mean velocity       : {metrics.mean_velocity:.1f} px/frame"
    if metrics.median_velocity is not None:
        yield f"  median velocity     : {metrics.median_velocity:.1f} px/frame"
    if metrics.p95_velocity is not None:
        yield f"  p95 velocity        : {metrics.p95_velocity:.1f} px/frame"
    if metrics.max_velocity is not None:
        yield f"  max velocity        : {metrics.max_velocity:.1f} px/frame"
    if metrics.mean_acceleration is not None:
        yield f"  mean acceleration   : {metrics.mean_acceleration:.1f} px/frame²"
    if metrics.p95_acceleration is not None:
        yield f"  p95 acceleration    : {metrics.p95_acceleration:.1f} px/frame²"
    if metrics.max_acceleration is not None:
        yield f"  max acceleration    : {metrics.max_acceleration:.1f} px/frame²"
    yield f"  direction changes   : {metrics.direction_changes}"
    yield f"  spike count         : {metrics.spike_count}"
    if metrics.legacy_outlier_rate is not None:
        yield f"  qc outlier rate      : {metrics.legacy_outlier_rate:.0%}"
    if metrics.bbox_mean_velocity is not None:
        yield f"  bbox mean velocity  : {metrics.bbox_mean_velocity:.3f} horse-widths/frame"
    if metrics.bbox_p95_velocity is not None:
        yield f"  bbox p95 velocity   : {metrics.bbox_p95_velocity:.3f} horse-widths/frame"
