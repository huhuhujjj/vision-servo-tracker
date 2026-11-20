"""主机端实时检测窗口：采集、推理、目标点选与运行摘要。"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path
from queue import SimpleQueue
import sys
from time import perf_counter

import cv2
from ultralytics import YOLO
from .target_lock import Detection, TargetLock, TargetState


CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MODEL_NAME = "yolo26n.pt"
WINDOW_NAME = "C922 | YOLO26n + ByteTrack"
DEFAULT_MODEL = Path("models") / MODEL_NAME


def tracked_detections(result) -> list[Detection]:
    """只向点选模块提供已被跟踪器赋予 ID 的检测框。"""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    return [Detection(int(tid), str(result.names[int(cls)]), tuple(map(float, xyxy)), float(conf))
            for tid, cls, xyxy, conf in zip(
                boxes.id.int().cpu().tolist(), boxes.cls.cpu().tolist(),
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist())]


def render_tracking(result, state: TargetState, active_ids: int, deadzone: float,
                    mirror: bool = False):
    image = result.plot()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    color = (0, 220, 0) if state.centered else (160, 160, 160)
    cv2.rectangle(image,
                  (int(width * (1 - deadzone) / 2), int(height * (1 - deadzone) / 2)),
                  (int(width * (1 + deadzone) / 2), int(height * (1 + deadzone) / 2)),
                  color, 1)
    cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 24, 2)
    if state.status == "tracking":
        x1, y1, x2, y2 = map(round, state.box)
        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.circle(image, target_center, 6, (0, 255, 255), -1)
        cv2.line(image, center, target_center, (0, 255, 255), 2)
        dx, dy = state.offset_px
        nx, ny = state.offset_normalized
        status = "CENTERED" if state.centered else "TRACKING"
        title = f"{status} | {state.class_name} id:{state.track_id} | Active IDs: {active_ids}"
        detail = f"Offset: dx={dx:+.0f}px dy={dy:+.0f}px | x={nx:+.2f} y={ny:+.2f}"
    elif state.status == "lost":
        title = f"LOST | {state.class_name} id:{state.track_id} | No position output"
        detail = ("Detector class changed for this ID. Click the intended box to select again."
                  if state.reason == "class_changed" else
                  "Selected ID is missing. Wait for it, or click a box / press R to select again.")
    else:
        selection_class = "ANY DETECTED OBJECT" if state.class_name == "any" else state.class_name
        title = f"SELECT TARGET: {selection_class} | Active IDs: {active_ids}"
        detail = "Click a tracked target, or press SPACE to select the one nearest the center."
    image = cv2.copyMakeBorder(image, 0, 108, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    text_scale = min(0.65, width / 1800)
    for text, y, text_color in (
        (title, height + 27, (0, 255, 255)),
        (detail, height + 57, (255, 255, 255)),
        (f"Mirror: {'ON' if mirror else 'OFF'} | SPACE: select | R: release | Q: quit | +x right, +y up", height + 87,
         (200, 200, 200)),
    ):
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1,
                    cv2.LINE_AA)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="C922 + YOLO26n + ByteTrack")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Local model file.")
    parser.add_argument("--device", default="0", help="Inference device: 0 for CUDA, cpu for CPU.")
    parser.add_argument("--report-dir", type=Path, default=Path("runs"))
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--verify-frames", type=int, default=0,
                        help="Verify a fixed number of frames without opening a window.")
    parser.add_argument("--target-class", default="any",
                        help="Selectable COCO class name, or 'any' for all detected classes (default).")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference input size (default: 960).")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Detection threshold; 0.1 allows ByteTrack low-confidence recovery.")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                        help="Mirror camera frames horizontally before detection (default: on).")
    args = parser.parse_args()
    if args.verify_frames < 0:
        parser.error("--verify-frames must be nonnegative")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32")
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and at most 1")
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"找不到模型：{model_path}。先运行 python -m vision_servo_tracker.models。")
        return 1
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        camera.release()
        print(
            f"打不开摄像头索引 {args.camera}。请确认 C922 已插好、没有被其他应用占用。"
        )
        return 1

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 30)
    report = {
        "started_at": datetime.now().astimezone().isoformat(),
        "camera_index": args.camera,
        "model": model_path.name,
        "device": args.device,
        "tracker": "bytetrack.yaml",
        "imgsz": args.imgsz,
        "conf": args.conf,
        "mirror": args.mirror,
        "coordinate_space": "mirrored_image" if args.mirror else "camera_image",
        "offset_axes": {"x_positive": "right", "y_positive": "up"},
        "frames": 0,
        "frames_with_tracks": 0,
        "adjacent_frames_with_shared_ids": 0,
        "status": "running",
    }
    classes = Counter()
    track_ids_seen = set()
    previous_ids = set()
    inference_count = 0
    inference_seconds = 0.0
    target_lock = TargetLock(args.target_class)
    target_samples = deque(maxlen=120)
    selections = deque(maxlen=100)
    clicks = SimpleQueue()
    pending_key = -1
    report.update(target_class=args.target_class, target_selection_count=0,
                  target_visible_frames=0, target_lost_frames=0, target_centered_frames=0)
    run_started = perf_counter()

    try:
        import torch
        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("当前环境无法使用 NVIDIA 显卡，请先检查 GPU 运行环境。")
        report["gpu"] = (torch.cuda.get_device_name(int(args.device))
                         if args.device != "cpu" else None)
        report["torch"] = torch.__version__
        model = YOLO(str(model_path))
        if args.target_class != "any" and args.target_class not in model.names.values():
            raise ValueError(f"模型没有类别 {args.target_class}，请使用 COCO 类别英文名称。")
        print(f"使用摄像头索引 {args.camera}，模型 {MODEL_NAME}，跟踪器 ByteTrack。")
        print(f"左右镜像：{'已开启' if args.mirror else '已关闭'}；识别、显示和偏差都以当前画面为准。")
        if args.verify_frames:
            print(f"开始读取并处理 {args.verify_frames} 帧，验证完成后自动关闭摄像头。")
        else:
            print("画面窗口中会显示识别框和临时跟踪 ID；按 q 或点击窗口右上角关闭。")
            selectable = "所有已识别类别" if args.target_class == "any" else args.target_class
            print(f"允许选择：{selectable}。点击目标自己的识别框；框重叠时优先选中较小的框。")
            print("点人物框可跟随人物，点 cell phone 框可跟随手机。空格选中靠近画面中心的目标。")
            print(f"识别输入尺寸：{args.imgsz}，检测阈值：{args.conf}。")
            print("黄色线连接目标和画面中心；dx 正数表示目标在右，dy 正数表示目标在上。")
            print("按 r 解除锁定。目标丢失时保留原编号并清空位置，不自动切换其他编号。")
        if not args.verify_frames:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 828)

            def on_mouse(event, x, y, _flags, _param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # 点击对应刚显示的检测框，不能用下一帧恰好经过此位置的对象替换。
                    clicks.put((list(detections), x, y))

            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("没有读到摄像头画面。请检查 C922 连接和摄像头占用情况。")

            # 在推理前镜像一次，显示、检测框与点击坐标才能保持一致。
            if args.mirror:
                frame = cv2.flip(frame, 1)

            inference_started = perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            elapsed = perf_counter() - inference_started
            # 首帧包含模型预热开销，不计入后续帧的平均调用耗时。
            if report["frames"] > 0:
                inference_count += 1
                inference_seconds += elapsed
            report["frames"] += 1
            report["frame_size"] = [int(frame.shape[1]), int(frame.shape[0])]
            ids = set()
            if result.boxes is not None:
                classes.update(model.names[int(c)] for c in result.boxes.cls.cpu().tolist())
                if result.boxes.id is not None:
                    ids = set(result.boxes.id.int().cpu().tolist())
            if ids:
                report["frames_with_tracks"] += 1
                track_ids_seen.update(ids)
            if ids & previous_ids:
                report["adjacent_frames_with_shared_ids"] += 1
            previous_ids = ids
            detections = tracked_detections(result)
            height, width = frame.shape[:2]
            selected = False
            if pending_key in (ord("r"), ord("R")):
                target_lock.release()
            elif pending_key == ord(" "):
                selected = target_lock.select_nearest(detections, width, height)
            while not clicks.empty():
                clicked_detections, x, y = clicks.get()
                if 0 <= x < width and 0 <= y < height:
                    selected = target_lock.select_at(clicked_detections, x, y) or selected
            if selected:
                report["target_selection_count"] += 1
                selections.append({"frame": report["frames"], "track_id": target_lock.track_id})
            state = target_lock.observe(detections, width, height)
            report["last_target"] = state.to_dict()
            if state.status == "tracking":
                report["target_visible_frames"] += 1
                report["target_centered_frames"] += int(state.centered)
            elif state.status == "lost":
                report["target_lost_frames"] += 1
            if state.status != "idle":
                target_samples.append({"frame": report["frames"], **state.to_dict()})
            if args.verify_frames and report["frames"] >= args.verify_frames:
                report["status"] = "verification_finished"
                return 0
            if not args.verify_frames:
                annotated = render_tracking(result, state, len(ids), target_lock.deadzone, args.mirror)
                cv2.imshow(WINDOW_NAME, annotated)
                pending_key = cv2.waitKey(1) & 0xFF
                if (pending_key in (ord("q"), ord("Q"))
                        or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1):
                    report["status"] = "stopped_by_user"
                    return 0
    except KeyboardInterrupt:
        report["status"] = "stopped_by_user"
        return 0
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        print(f"启动或运行失败：{exc}")
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        report["ended_at"] = datetime.now().astimezone().isoformat()
        report["elapsed_seconds"] = round(perf_counter() - run_started, 3)
        report["class_detections_across_frames"] = dict(classes)
        report["track_ids"] = sorted(track_ids_seen)
        report["target_selections_recent"] = list(selections)
        report["target_samples_recent"] = list(target_samples)
        if inference_count:
            report["mean_track_call_ms_excluding_first"] = round(
                inference_seconds / inference_count * 1000, 2)
        report_dir = args.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"session_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"运行记录已保存：{report_path}")



if __name__ == "__main__":
    sys.exit(main())
