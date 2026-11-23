import unittest

from vision_servo_tracker.motion import AxisCalibration, MotionPlanner
from vision_servo_tracker.target_lock import TargetState


PAN = AxisCalibration(1500, 1450, 1550, 1, 100, 10)
TILT = AxisCalibration(1500, 1450, 1550, -1, 100, 10)


def selected(track_id=7, offset=(0.5, -0.5)):
    return TargetState("tracking", track_id, "person", offset_normalized=offset)


class MotionSafetyTests(unittest.TestCase):
    def test_requires_explicit_arming_and_limits_each_step(self):
        planner = MotionPlanner(PAN, TILT)
        self.assertFalse(planner.update(selected(), frame_at=1.0, now=1.1).enabled)
        planner.arm(selected(), pan_us=1500, tilt_us=1500)
        first = planner.update(selected(), frame_at=1.0, now=1.1)
        self.assertEqual((first.pan_us, first.tilt_us), (1510, 1510))
        for index in range(12):
            latest = planner.update(selected(offset=(1.0, -1.0)),
                                    frame_at=2.0 + index, now=2.01 + index)
        self.assertEqual((latest.pan_us, latest.tilt_us), (1550, 1550))

    def test_lost_stale_and_new_target_disable_output(self):
        planner = MotionPlanner(PAN, TILT)
        planner.arm(selected(), pan_us=1500, tilt_us=1500)
        self.assertFalse(planner.update(TargetState("lost", 7, "person"),
                                        frame_at=1.0, now=1.1).enabled)
        self.assertFalse(planner.update(selected(), frame_at=1.2, now=1.3).enabled)
        planner.arm(selected(), pan_us=1500, tilt_us=1500)
        self.assertEqual(planner.update(selected(), frame_at=1.0, now=1.4).reason,
                         "stale_frame")
        planner.arm(selected(), pan_us=1500, tilt_us=1500)
        self.assertEqual(planner.update(selected(track_id=8), frame_at=2.0, now=2.1).reason,
                         "target_not_armed")

    def test_center_deadzone_holds_and_bad_values_are_rejected(self):
        planner = MotionPlanner(PAN, TILT)
        planner.arm(selected(), pan_us=1500, tilt_us=1500)
        held = planner.update(selected(offset=(0.05, -0.05)), frame_at=1.0, now=1.1)
        self.assertEqual((held.pan_us, held.tilt_us), (1500, 1500))
        self.assertEqual(planner.update(selected(offset=(float("nan"), 0)),
                                        frame_at=2.0, now=2.1).reason, "invalid_offset")
        with self.assertRaises(ValueError):
            planner.arm(selected(), pan_us=1600, tilt_us=1500)
        with self.assertRaises(ValueError):
            AxisCalibration(1500, 1450, 1550, 0, 100, 10)

    def test_one_video_frame_cannot_issue_repeated_movement(self):
        planner = MotionPlanner(PAN, TILT)
        planner.arm(selected(), pan_us=1500, tilt_us=1500)
        self.assertEqual(planner.update(selected(), frame_at=1.0, now=1.1).pan_us, 1510)
        self.assertEqual(planner.update(selected(), frame_at=1.0, now=1.2).reason,
                         "repeated_frame")
        self.assertEqual(planner.update(selected(), frame_at=1.3, now=1.4).pan_us, 1520)


if __name__ == "__main__":
    unittest.main()
