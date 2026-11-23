"""HTTP MJPEG取流；读取线程只保留最新JPEG，断线后重新建立连接。"""
from dataclasses import dataclass
import threading
import time
import urllib.parse
import urllib.request


@dataclass(frozen=True)
class Frame:
    sequence: int
    generation: int
    received_at: float
    jpeg: bytes


class JpegBuffer:
    """按JPEG边界提取完整帧，不依赖网络分块恰好与帧对齐。"""
    def __init__(self, limit=8 * 1024 * 1024):
        self.data = bytearray()
        self.limit = limit

    def feed(self, chunk):
        self.data.extend(chunk)
        frames = []
        while True:
            start = self.data.find(b"\xff\xd8")
            if start < 0:
                self.data[:] = self.data[-1:]
                return frames
            if start:
                del self.data[:start]
            end = self.data.find(b"\xff\xd9", 2)
            if end < 0:
                if len(self.data) > self.limit:
                    self.data.clear()
                    raise ValueError("MJPEG帧超过大小限制")
                return frames
            if end + 2 > self.limit:
                self.data.clear()
                raise ValueError("MJPEG帧超过大小限制")
            frames.append(bytes(self.data[:end + 2]))
            del self.data[:end + 2]


class MjpegReader:
    def __init__(self, url, timeout=3.0):
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            raise ValueError("网络源必须是HTTP或HTTPS MJPEG地址")
        self.url = url
        self.timeout = timeout
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.sequence = 0
        self.generation = 0

        self.online = False
        self.error = "等待视频连接"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                request = urllib.request.Request(self.url, headers={"User-Agent": "vision-servo-tracker"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if "multipart/" not in response.headers.get("Content-Type", "").lower():
                        raise ValueError("地址没有返回MJPEG视频流")
                    parser = JpegBuffer()
                    with self.condition:
                        self.generation += 1
                    while not self.stop.is_set():
                        chunk = response.read1(65536)
                        if not chunk:
                            raise ConnectionError("视频连接关闭")
                        for jpeg in parser.feed(chunk):
                            with self.condition:
                                self.sequence += 1
                                self.latest = Frame(self.sequence, self.generation, time.monotonic(), jpeg)
                                self.online = True
                                self.error = ""
                                self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    self.online = False
                    self.latest = None
                    # 不把可能含凭据的URL写进日志。
                    self.error = type(exc).__name__
                    self.condition.notify_all()
                self.stop.wait(1)

    def get(self, previous, wait=0.03):
        with self.condition:
            self.condition.wait_for(
                lambda: self.stop.is_set() or (self.latest and self.latest.sequence != previous),
                timeout=wait)
            if (self.latest and self.latest.sequence != previous
                    and time.monotonic() - self.latest.received_at <= self.timeout):
                return self.latest
            return None

    def healthy(self):
        with self.condition:
            return bool(self.online and self.latest
                        and time.monotonic() - self.latest.received_at <= self.timeout)

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=self.timeout + 2)


class NetworkCamera:
    """为主机检测循环提供与本地相机一致的read/release接口。"""
    def __init__(self, url, timeout=3):
        self.reader = MjpegReader(url, timeout)
        self.sequence = 0
        self.generation = 0
        self.decode_failed = False

    def read(self):
        packet = self.reader.get(self.sequence)
        if packet is None:
            return False, None
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(packet.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.sequence = packet.sequence
        self.generation = packet.generation
        self.decode_failed = image is None
        return image is not None, image

    def healthy(self):
        return not self.decode_failed and self.reader.healthy()

    def release(self):
        self.reader.close()
