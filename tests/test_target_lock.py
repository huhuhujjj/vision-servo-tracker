import unittest

from vision_servo_tracker.target_lock import Detection, TargetLock


def person(tid, box, name="person"):
    return Detection(tid, name, box, 0.9)


class TargetLockTests(unittest.TestCase):
    def test_requires_selection_and_ignores_other_classes(self):
        lock = TargetLock("person")
        detections = [person(1, (0, 0, 100, 100), "cup")]
        self.assertFalse(lock.select_nearest(detections, 100, 100))
        self.assertEqual(lock.observe(detections, 100, 100).status, "idle")

    def test_nearest_selection_does_not_switch_to_a_new_closer_person(self):
        lock = TargetLock()
        self.assertTrue(lock.select_nearest([
            person(1, (0, 0, 20, 20)), person(2, (30, 30, 70, 70))], 100, 100))
        self.assertEqual(lock.track_id, 2)
        self.assertFalse(lock.select_nearest([person(3, (40, 40, 60, 60))], 100, 100))
        self.assertEqual(lock.track_id, 2)

    def test_missing_id_clears_offsets_and_does_not_follow_another_id(self):
        lock = TargetLock()
        lock.select_at([person(8, (10, 10, 30, 30))], 20, 20)
        present = lock.observe([person(8, (10, 10, 30, 30))], 100, 100)
        self.assertEqual(present.offset_px, (-30, 30))
        lost = lock.observe([person(9, (10, 10, 30, 30))], 100, 100)
        self.assertEqual(lost.status, "lost")
        self.assertEqual(lost.track_id, 8)
        self.assertIsNone(lost.offset_px)
        self.assertIsNone(lost.offset_normalized)
        self.assertFalse(lost.centered)

    def test_same_id_reappears_and_release_requires_new_selection(self):
        lock = TargetLock()
        target = person(4, (40, 40, 60, 60))
        lock.select_at([target], 50, 50)
        self.assertEqual(lock.observe([], 100, 100).status, "lost")
        self.assertTrue(lock.observe([target], 100, 100).centered)
        lock.release()
        self.assertEqual(lock.observe([target], 100, 100).status, "idle")

    def test_offsets_are_right_and_up_positive_and_resolution_independent(self):
        lock = TargetLock()
        target = person(4, (120, 60, 180, 90))
        lock.select_at([target], 150, 75)
        state = lock.observe([target], 200, 100)
        self.assertEqual(state.offset_px, (50, -25))
        self.assertEqual(state.offset_normalized, (0.5, -0.5))
        self.assertFalse(state.centered)

    def test_click_requires_matching_class_and_prefers_tighter_box(self):
        lock = TargetLock("person")
        self.assertTrue(lock.select_at([
            person(1, (0, 0, 100, 100)), person(2, (20, 20, 40, 40)),
            person(3, (24, 24, 26, 26), "cup")], 25, 25))
        self.assertEqual(lock.track_id, 2)
        self.assertFalse(lock.select_at([], 25, 25))
        self.assertEqual(lock.track_id, 2)

    def test_wrong_class_same_id_cannot_resume_target(self):
        lock = TargetLock()
        lock.select_at([person(1, (10, 10, 30, 30))], 20, 20)
        self.assertEqual(lock.observe([person(1, (10, 10, 30, 30), "cup")], 100, 100).status,
                         "lost")

    def test_invalid_box_is_not_selected_and_invalid_dimensions_fail(self):
        lock = TargetLock()
        self.assertFalse(lock.select_nearest([person(1, (0, 0, float("nan"), 10))], 100, 100))
        with self.assertRaises(ValueError):
            lock.observe([], 0, 100)

    def test_phone_inside_person_box_is_selected_when_clicked(self):
        lock = TargetLock()
        detections = [person(1, (0, 0, 100, 100)),
                      person(2, (25, 30, 40, 60), "cell phone")]
        self.assertTrue(lock.select_at(detections, 32, 40))
        state = lock.observe(detections, 100, 100)
        self.assertEqual(state.track_id, 2)
        self.assertEqual(state.class_name, "cell phone")
        self.assertEqual(state.status, "tracking")

    def test_explicit_click_can_switch_from_person_to_phone(self):
        lock = TargetLock()
        detections = [person(1, (0, 0, 100, 100)),
                      person(2, (25, 30, 40, 60), "cell phone")]
        lock.select_at(detections, 80, 80)
        self.assertEqual(lock.selected_class, "person")
        lock.select_at(detections, 32, 40)
        self.assertEqual(lock.track_id, 2)
        self.assertEqual(lock.selected_class, "cell phone")

    def test_phone_disappearance_does_not_automatically_switch_to_person(self):
        lock = TargetLock()
        lock.select_at([person(2, (25, 30, 40, 60), "cell phone")], 32, 40)
        state = lock.observe([person(1, (0, 0, 100, 100))], 100, 100)
        self.assertEqual(state.status, "lost")
        self.assertEqual(state.track_id, 2)
        self.assertEqual(state.class_name, "cell phone")
        self.assertIsNone(state.offset_px)
        self.assertEqual(state.reason, "missing_track")

    def test_phone_class_change_is_reported_without_position_output(self):
        lock = TargetLock()
        lock.select_at([person(2, (25, 30, 40, 60), "cell phone")], 32, 40)
        state = lock.observe([person(2, (25, 30, 40, 60), "remote")], 100, 100)
        self.assertEqual(state.status, "lost")
        self.assertEqual(state.reason, "class_changed")
        self.assertIsNone(state.offset_px)
        lock.release()
        self.assertIsNone(lock.selected_class)
        self.assertEqual(lock.observe([], 100, 100).status, "idle")


if __name__ == "__main__":
    unittest.main()
