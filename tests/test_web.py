"""不用模型和摄像头，检查网页选框协议及失效位置的处理。"""
import http.client
import json
import threading
import unittest
from unittest.mock import patch

from http.server import ThreadingHTTPServer
from vision_servo_tracker.target_lock import Detection
from vision_servo_tracker.web import Shared, handler_for


def phone(track_id=2, box=(20, 20, 40, 40)):
    return Detection(track_id, "cell phone", box, 0.9)


class SharedStateTests(unittest.TestCase):
    def test_click_uses_displayed_frame_not_a_new_object_at_same_position(self):
        state = Shared()
        state.publish(b"old", [phone(2)], 100, 100, 5)
        state.publish(b"new", [phone(9)], 100, 100, 5)
        state.command({"action": "select", "frame_id": 1, "x": 0.3, "y": 0.3})
        self.assertEqual(state.target.track_id, 2)
        self.assertEqual(state.status()["status"], "lost")
        self.assertIsNone(state.status()["offset_px"])

    def test_expired_frame_cannot_change_selection(self):
        state = Shared()
        with patch("vision_servo_tracker.web.time.monotonic", return_value=10):
            state.publish(b"frame", [phone()], 100, 100, 5)
        with patch("vision_servo_tracker.web.time.monotonic", return_value=14):
            with self.assertRaises(ValueError):
                state.command({"action": "select", "frame_id": 1, "x": 0.3, "y": 0.3})
        self.assertIsNone(state.target.track_id)

    def test_stream_timeout_clears_old_coordinates(self):
        state = Shared()
        with patch("vision_servo_tracker.web.time.monotonic", return_value=10):
            state.publish(b"frame", [phone()], 100, 100, 5)
            state.command({"action": "select", "frame_id": 1, "x": 0.3, "y": 0.3})
            self.assertEqual(state.status()["offset_px"], (-20, 20))
        with patch("vision_servo_tracker.web.time.monotonic", return_value=14):
            status = state.status()
        self.assertEqual(status["status"], "offline")
        self.assertIsNone(status["box"])
        self.assertIsNone(status["offset_px"])
        self.assertIsNone(status["offset_normalized"])

    def test_release_works_even_when_camera_failed(self):
        state = Shared()
        state.publish(b"frame", [phone()], 100, 100, 5)
        state.command({"action": "select", "frame_id": 1, "x": 0.3, "y": 0.3})
        state.error = "camera failed"
        state.command({"action": "release"})
        self.assertIsNone(state.target.track_id)
        self.assertIsNone(state.status()["offset_px"])

    def test_nonfinite_or_boolean_coordinates_are_rejected(self):
        state = Shared()
        state.publish(b"frame", [phone()], 100, 100, 5)
        for invalid in (float("nan"), float("inf"), True, "0.3", -0.1, 1.1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                state.command({"action": "select", "frame_id": 1, "x": invalid, "y": 0.3})

    def test_frame_history_is_bounded(self):
        state = Shared()
        for _ in range(35):
            state.publish(b"frame", [phone()], 100, 100, 5)
        self.assertEqual(len(state.snapshots), 30)
        self.assertNotIn(1, state.snapshots)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.state = Shared()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_preview_carries_state_for_the_same_frame(self):
        self.state.publish(b"jpeg-test", [phone()], 100, 100, 5)
        status, headers, body = self.request("GET", "/frame")
        state = json.loads(headers["X-Tracking-State"])
        self.assertEqual(status, 200)
        self.assertEqual(body, b"jpeg-test")
        self.assertEqual(state["frame_id"], int(headers["X-Frame-ID"]))

    def test_page_inserts_token_and_static_resources_are_available(self):
        status, _, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(self.state.token.encode(), body)
        self.assertNotIn(b"__TOKEN__", body)
        self.assertEqual(self.request("GET", "/app.js")[0], 200)
        self.assertEqual(self.request("GET", "/style.css")[0], 200)

    def test_unauthenticated_control_request_does_not_select(self):
        self.state.publish(b"frame", [phone()], 100, 100, 5)
        status, _, _ = self.request("POST", "/command", json.dumps({"action": "release"}))
        self.assertEqual(status, 403)
        self.assertIsNone(self.state.target.track_id)

    def test_control_selects_phone_and_returns_updated_state(self):
        self.state.publish(b"frame", [phone()], 100, 100, 5)
        payload = json.dumps({"action": "select", "frame_id": 1, "x": 0.3, "y": 0.3})
        status, _, body = self.request("POST", "/command", payload,
            {"Content-Type": "application/json", "X-Control-Token": self.state.token})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["offset_px"], [-20, 20])

    def test_invalid_json_is_a_client_error(self):
        status, _, _ = self.request("POST", "/command", "{invalid",
            {"X-Control-Token": self.state.token})
        self.assertEqual(status, 400)

    def test_initializing_stream_returns_unavailable_without_old_frame(self):
        status, _, body = self.request("GET", "/frame")
        self.assertEqual(status, 503)
        self.assertIsNone(json.loads(body)["offset_px"])


if __name__ == "__main__":
    unittest.main()
