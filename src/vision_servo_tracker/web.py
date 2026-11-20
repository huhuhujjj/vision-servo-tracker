"""树莓派网页端：后台推理、JPEG预览与按显示帧选择目标。"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .target_lock import Detection, TargetLock


class Shared:
    """推理线程与 HTTP 请求共享的状态；读写统一在 lock 内完成。"""
    def __init__(self):
        self.lock = threading.Lock()
        self.target = TargetLock()
        self.snapshots = OrderedDict()
        self.jpeg = None
        self.sequence = 0
        self.updated = 0.0
        self.version = 0
        self.error = "正在加载模型和摄像头"
        self.state = {}
        self.token = secrets.token_urlsafe(24)
        self.stop = threading.Event()

    def publish(self, jpeg, detections, width, height, fps):
        with self.lock:
            self.sequence += 1
            self.version += 1
            now = time.monotonic()
            self.snapshots[self.sequence] = (now, detections, width, height)
            while len(self.snapshots) > 30:
                self.snapshots.popitem(last=False)
            self.jpeg = jpeg
            self.updated = now
            self.error = ""
            self.state = {**self.target.observe(detections, width, height).to_dict(),
                          "version": self.version,
                          "frame_id": self.sequence, "fps": round(fps, 1),
                          "axes": {"x_positive": "right", "y_positive": "up"}}

    def status(self):
        # 相机卡住或线程出错时，旧偏差不能继续作为有效位置输出。
        result = dict(self.state)
        if self.error or time.monotonic() - self.updated > 3:
            result.update(status="offline", box=None, offset_px=None, offset_normalized=None,
                          centered=False, reason=self.error or "画面更新超时")
        return result

    def command(self, data):
        with self.lock:
            if data.get("action") not in ("select", "release"):
                raise ValueError("操作无效")
            if data.get("action") == "release":
                self.target.release()
            else:
                frame_id = data.get("frame_id")
                snapshot = self.snapshots.get(frame_id) if type(frame_id) is int else None
                if snapshot is None or time.monotonic() - snapshot[0] > 3 or self.error:
                    raise ValueError("画面已过期，请等待新画面后重新点击")
                _, detections, width, height = snapshot
                x, y = data.get("x"), data.get("y")
                if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                           for v in (x, y)):
                    raise ValueError("点击坐标无效")
                if not self.target.select_at(detections, x * width, y * height):
                    raise ValueError("请点击带编号的识别框")
            self.version += 1
            self.state["version"] = self.version
            if self.snapshots:
                _, detections, width, height = next(reversed(self.snapshots.values()))
                self.state.update(self.target.observe(detections, width, height).to_dict())


class LatestCamera:
    """持续读取相机，只保留最新帧，避免 CPU 推理变慢时积压视频。"""
    def __init__(self, camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.camera.read()
                if not ok:
                    raise RuntimeError("C922 读取失败，请检查 USB 连接和摄像头占用")
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def read(self, previous):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous or self.error or self.stop.is_set(), timeout=5)
            if self.stop.is_set():
                raise RuntimeError("摄像头读取已停止")
            if self.error:
                raise self.error
            if self.sequence == previous:
                raise RuntimeError("C922 读取超时")
            return self.sequence, self.frame.copy()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
        # 不与仍阻塞在 read() 的线程同时释放句柄，避免访问冲突。
        if not self.thread.is_alive():
            self.camera.release()


def worker(shared, args):
    camera = None
    reader = None
    try:
        import cv2
        from ultralytics import YOLO
        cv2.setNumThreads(2)
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在：{model_path}")
        model = YOLO(str(model_path), task="detect")
        camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not camera.isOpened():
            raise RuntimeError("打不开 C922，请先停止 ustreamer 并确认 USB 连接")
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        reader = LatestCamera(camera)
        sequence = 0
        while not shared.stop.is_set():
            started = time.monotonic()
            sequence, frame = reader.read(sequence)
            if args.mirror:
                frame = cv2.flip(frame, 1)
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 imgsz=args.imgsz, conf=0.1, device="cpu", verbose=False)[0]
            detections = []
            if result.boxes is not None and result.boxes.id is not None:
                b = result.boxes
                detections = [Detection(int(t), result.names[int(c)], tuple(map(float, xy)), float(p))
                              for t, c, xy, p in zip(b.id.cpu().tolist(), b.cls.cpu().tolist(),
                                                     b.xyxy.cpu().tolist(), b.conf.cpu().tolist())]
            height, width = frame.shape[:2]
            image = result.plot()
            cv2.drawMarker(image, (width // 2, height // 2), (255, 255, 255),
                           cv2.MARKER_CROSS, 24, 2)
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise RuntimeError("无法生成网页画面")
            shared.publish(jpeg.tobytes(), detections, width, height,
                           1 / max(0.001, time.monotonic() - started))
    except Exception as exc:
        with shared.lock:
            shared.error = str(exc)
        print(f"识别停止：{exc}", flush=True)
    finally:
        if reader:
            reader.close()
        elif camera:
            camera.release()


def handler_for(shared):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_body(self, status, body, content_type, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == "/":
                    page = Path(__file__).with_name("static").joinpath("index.html").read_text(encoding="utf-8")
                    self.send_body(200, page.replace("__TOKEN__", shared.token).encode(), "text/html; charset=utf-8")
                elif self.path in ("/app.js", "/style.css"):
                    static_file = Path(__file__).with_name("static") / self.path[1:]
                    mime = "text/javascript" if self.path.endswith(".js") else "text/css"
                    self.send_body(200, static_file.read_bytes(), mime + "; charset=utf-8")
                elif self.path == "/frame":
                    with shared.lock:
                        jpeg, frame_id = shared.jpeg, shared.sequence
                        status = shared.status()
                        # JSON 中的中文会转义，保证响应头只包含 ASCII。
                        state_header = json.dumps(status, ensure_ascii=True)
                    if jpeg is None or status["status"] == "offline":
                        self.send_body(503, json.dumps(status, ensure_ascii=False).encode(), "application/json")
                    else:
                        self.send_body(200, jpeg, "image/jpeg", {"X-Frame-ID": frame_id, "X-Tracking-State": state_header})
                elif self.path == "/status":
                    with shared.lock:
                        status = shared.status()
                    self.send_body(200, json.dumps(status, ensure_ascii=False).encode(), "application/json")
                else:
                    self.send_body(404, b"Not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                if self.path != "/command" or not secrets.compare_digest(
                        self.headers.get("X-Control-Token", ""), shared.token):
                    self.send_body(403, b"Forbidden", "text/plain")
                    return
                self.connection.settimeout(5)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("请求长度无效")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or data.get("action") not in ("select", "release"):
                    raise ValueError("操作无效")
                shared.command(data)
                with shared.lock:
                    status = shared.status()
                self.send_body(200, json.dumps(status, ensure_ascii=False).encode(), "application/json")
            except (TimeoutError, OSError):
                # 断开的客户端不应阻塞后续请求。
                return
            except (ValueError, TypeError) as exc:
                self.send_body(400, json.dumps({"error": str(exc)}, ensure_ascii=False).encode(), "application/json")
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo26n_ncnn_model")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("imgsz 必须为 32 的正整数倍，并与 NCNN 导出尺寸相同")
    shared = Shared()
    # 只监听回环地址；远程访问通过 SSH 转发，不暴露摄像头到公网。
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(shared))
    thread = threading.Thread(target=worker, args=(shared, args), daemon=True)
    thread.start()
    print(f"网页服务已启动：127.0.0.1:{args.port}，Ctrl+C 停止", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shared.stop.set()
        server.server_close()
        thread.join(timeout=8)


if __name__ == "__main__":
    main()
