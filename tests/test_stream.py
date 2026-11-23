import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vision_servo_tracker.stream import JpegBuffer, MjpegReader


class JpegTests(unittest.TestCase):
    def test_markers_can_span_network_chunks(self):
        parser = JpegBuffer()
        self.assertEqual(parser.feed(b"header\r\n\xff"), [])
        self.assertEqual(parser.feed(b"\xd8abc\xff"), [])
        self.assertEqual(parser.feed(b"\xd9\r\n"), [b"\xff\xd8abc\xff\xd9"])

    def test_multiple_frames_in_one_chunk(self):
        frame = b"\xff\xd8test\xff\xd9"
        self.assertEqual(JpegBuffer().feed(frame + b"boundary" + frame), [frame, frame])

    def test_incomplete_oversized_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            JpegBuffer(limit=10).feed(b"\xff\xd8" + b"x" * 11)

    def test_complete_oversized_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            JpegBuffer(limit=10).feed(b"\xff\xd8" + b"x" * 11 + b"\xff\xd9")


class ReaderTests(unittest.TestCase):
    def test_invalid_source_scheme_is_rejected(self):
        with self.assertRaises(ValueError):
            MjpegReader("file:///private/file")

    def test_reconnect_changes_generation_and_retains_only_latest_packet(self):
        disconnect = threading.Event()
        finish = threading.Event()
        counter = [0]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                counter[0] += 1
                generation = counter[0]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    self.wfile.write(b"--frame\r\n\r\n\xff\xd8" + str(generation).encode() + b"\xff\xd9\r\n")
                    self.wfile.flush()
                    (disconnect if generation == 1 else finish).wait(5)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        reader = MjpegReader(f"http://127.0.0.1:{server.server_port}/stream", timeout=2)
        try:
            first = reader.get(0, wait=3)
            self.assertIsNotNone(first)
            self.assertEqual(first.jpeg, b"\xff\xd81\xff\xd9")
            self.assertTrue(reader.healthy())
            disconnect.set()
            deadline = time.monotonic() + 4
            second = None
            while time.monotonic() < deadline:
                second = reader.get(first.sequence, wait=0.1)
                if second:
                    break
            self.assertIsNotNone(second)
            self.assertGreater(second.generation, first.generation)
            self.assertEqual(second.jpeg, b"\xff\xd82\xff\xd9")
        finally:
            finish.set()
            disconnect.set()
            reader.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
