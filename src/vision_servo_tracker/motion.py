"""Turn fresh, selected-target offsets into bounded servo setpoints.

This module plans movement only; it does not communicate with a motor. Axis
direction and travel limits must be measured on the actual gimbal first.
"""

from dataclasses import dataclass
import math

from .target_lock import TargetState


@dataclass(frozen=True)
class AxisCalibration:
    center_us: int
    minimum_us: int
    maximum_us: int
    direction: int
    gain_us: float
    max_step_us: int

    def __post_init__(self) -> None:
        if not 1000 <= self.minimum_us < self.center_us < self.maximum_us <= 2000:
            raise ValueError("Setpoints must be ordered within 1000..2000 microseconds")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if not math.isfinite(self.gain_us) or self.gain_us <= 0:
            raise ValueError("gain_us must be positive and finite")
        if not 1 <= self.max_step_us <= 50:
            raise ValueError("max_step_us must be within 1..50")


@dataclass(frozen=True)
class MotionPlan:
    enabled: bool
    pan_us: int | None = None
    tilt_us: int | None = None
    reason: str = ""


class MotionPlanner:
    """Reject stale positions and release output whenever tracking is absent.

    ``now`` and ``frame_at`` use the same monotonic clock. A motor-side
    watchdog must separately turn outputs off if no plans arrive.
    """

    def __init__(self, pan: AxisCalibration, tilt: AxisCalibration,
                 *, deadzone: float = 0.10, max_age_s: float = 0.30):
        if not 0 <= deadzone < 1 or not 0 < max_age_s <= 1:
            raise ValueError("Invalid deadzone or maximum frame age")
        self.pan = pan
        self.tilt = tilt
        self.deadzone = deadzone
        self.max_age_s = max_age_s
        self._pan_us = pan.center_us
        self._tilt_us = tilt.center_us
        self._identity: tuple[int, str] | None = None
        self._last_frame_at: float | None = None

    def arm(self, state: TargetState, *, pan_us: int, tilt_us: int) -> None:
        """Arm only after a human has confirmed the actual starting pose."""
        if state.status != "tracking" or state.track_id is None:
            raise ValueError("A selected and visible target is required")
        if not self.pan.minimum_us <= pan_us <= self.pan.maximum_us:
            raise ValueError("Pan starting setpoint is outside calibrated travel")
        if not self.tilt.minimum_us <= tilt_us <= self.tilt.maximum_us:
            raise ValueError("Tilt starting setpoint is outside calibrated travel")
        self._pan_us = pan_us
        self._tilt_us = tilt_us
        self._identity = (state.track_id, state.class_name)
        self._last_frame_at = None

    def reset(self, reason: str) -> MotionPlan:
        self._identity = None
        self._last_frame_at = None
        return MotionPlan(False, reason=reason)

    @staticmethod
    def _advance(current: int, error: float, axis: AxisCalibration,
                 deadzone: float) -> int:
        if abs(error) <= deadzone:
            return current
        requested = round(error * axis.gain_us * axis.direction)
        step = max(-axis.max_step_us, min(axis.max_step_us, requested))
        return max(axis.minimum_us, min(axis.maximum_us, current + step))

    def update(self, state: TargetState, *, frame_at: float, now: float) -> MotionPlan:
        if not math.isfinite(frame_at) or not math.isfinite(now):
            return self.reset("invalid_time")
        if frame_at > now or now - frame_at > self.max_age_s:
            return self.reset("stale_frame")
        if state.status != "tracking" or state.track_id is None or state.offset_normalized is None:
            return self.reset("target_unavailable")
        dx, dy = state.offset_normalized
        if not all(math.isfinite(value) and abs(value) <= 1 for value in (dx, dy)):
            return self.reset("invalid_offset")
        identity = (state.track_id, state.class_name)
        if self._identity != identity:
            return self.reset("target_not_armed")
        if self._last_frame_at is not None and frame_at <= self._last_frame_at:
            return MotionPlan(False, reason="repeated_frame")
        self._last_frame_at = frame_at
        self._pan_us = self._advance(self._pan_us, dx, self.pan, self.deadzone)
        self._tilt_us = self._advance(self._tilt_us, dy, self.tilt, self.deadzone)
        return MotionPlan(True, self._pan_us, self._tilt_us, "tracking")
