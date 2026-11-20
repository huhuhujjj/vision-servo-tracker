"""目标选择与偏差计算。此模块不依赖摄像头、模型或舵机驱动。"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Detection:
    track_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def valid(self) -> bool:
        x1, y1, x2, y2 = self.xyxy
        return (all(math.isfinite(v) for v in self.xyxy)
                and x2 > x1 and y2 > y1)


@dataclass(frozen=True)
class TargetState:
    status: str
    track_id: int | None
    class_name: str
    box: tuple[float, float, float, float] | None = None
    offset_px: tuple[float, float] | None = None
    offset_normalized: tuple[float, float] | None = None
    centered: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class TargetLock:
    """保留用户选中的临时 ID 和类别，不自动改选其他对象。

    检测框仍按图像坐标表示，偏差则定义为右正、上正。
    同 ID 与类别重新出现时可恢复位置输出，但这不保证物理身份一致。
    """

    def __init__(self, target_class: str = "any", deadzone: float = 0.1):
        if not 0 <= deadzone < 1:
            raise ValueError("deadzone must be between 0 (inclusive) and 1")
        self.target_class = target_class
        self.deadzone = deadzone
        self.track_id: int | None = None
        self.selected_class: str | None = None

    def release(self) -> None:
        """清除选择；后续帧不会自动锁定其他目标。"""
        self.track_id = None
        self.selected_class = None

    def candidates(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections
                if (self.target_class == "any" or d.class_name == self.target_class) and d.valid]

    def select_at(self, detections: list[Detection], x: int, y: int) -> bool:
        hits = [d for d in self.candidates(detections)
                if d.xyxy[0] <= x <= d.xyxy[2]
                and d.xyxy[1] <= y <= d.xyxy[3]]
        if not hits:
            return False
        # 手机框常被人物框包围，优先取较小的框，避免点手机时选中人物。
        selected = min(hits, key=lambda d: (
            (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]), d.track_id))
        self.track_id = selected.track_id
        self.selected_class = selected.class_name
        return True

    def select_nearest(self, detections: list[Detection], width: int, height: int) -> bool:
        if self.track_id is not None:
            return False
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        choices = self.candidates(detections)
        if not choices:
            return False
        selected = min(choices, key=lambda d: (
            ((d.center[0] - width / 2) / (width / 2)) ** 2
            + ((d.center[1] - height / 2) / (height / 2)) ** 2,
            d.track_id))
        self.track_id = selected.track_id
        self.selected_class = selected.class_name
        return True

    def observe(self, detections: list[Detection], width: int, height: int) -> TargetState:
        """根据当前帧生成状态。丢失或类别变化时，不保留旧的位置值。"""
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.track_id is None:
            return TargetState("idle", None, self.target_class)
        same_id = [d for d in detections if d.valid and d.track_id == self.track_id]
        matches = [d for d in same_id if d.class_name == self.selected_class]
        if not matches:
            reason = "class_changed" if same_id else "missing_track"
            return TargetState("lost", self.track_id, self.selected_class, reason=reason)
        selected = max(matches, key=lambda d: d.confidence)
        x1, y1, x2, y2 = selected.xyxy
        box = (max(0.0, min(width, x1)), max(0.0, min(height, y1)),
               max(0.0, min(width, x2)), max(0.0, min(height, y2)))
        if box[2] <= box[0] or box[3] <= box[1]:
            return TargetState("lost", self.track_id, self.selected_class, reason="invalid_box")
        dx = (box[0] + box[2]) / 2 - width / 2
        # 图像 y 轴向下；减号让输出遵守“向上为正”的控制坐标约定。
        dy = height / 2 - (box[1] + box[3]) / 2
        normalized = (dx / (width / 2), dy / (height / 2))
        return TargetState(
            "tracking", self.track_id, self.selected_class, box,
            (dx, dy), normalized,
            abs(normalized[0]) <= self.deadzone and abs(normalized[1]) <= self.deadzone,
        )
