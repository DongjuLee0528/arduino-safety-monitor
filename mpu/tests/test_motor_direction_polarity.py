import os
import re
import unittest


def _repo_path(*parts):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", *parts))


def _read(*parts):
    with open(_repo_path(*parts), encoding="utf-8") as f:
        return f.read()


def _method_body(src, name):
    match = re.search(rf"void {name}\(\) \{{(?P<body>.*?)\n    \}}", src, re.S)
    if not match:
        raise AssertionError(f"{name}() not found")
    return match.group("body")


def _drive(forward, reversed_):
    drive_forward = not forward if reversed_ else forward
    return ("HIGH", "LOW") if drive_forward else ("LOW", "HIGH")


class TestMotorDirectionPolarity(unittest.TestCase):
    def setUp(self):
        self.config = _read("arduino", "config.h")
        self.motor = _read("arduino", "motor.h")

    def test_left_motor_reversed_config_enabled(self):
        self.assertIn("#define LEFT_MOTOR_REVERSED    true", self.config)

    def test_direction_polarity_is_centralized(self):
        self.assertIn("void writeSide(int forwardPin, int backwardPin, bool forward, bool reversed)", self.motor)
        self.assertIn("bool driveForward = reversed ? !forward : forward;", self.motor)
        self.assertEqual(self.motor.count("LEFT_MOTOR_REVERSED"), 5)

    def test_motion_methods_use_left_inversion_and_right_unmodified(self):
        expected = {
            "forward": (
                "writeSide(_lf, _lb, true,  LEFT_MOTOR_REVERSED);",
                "writeSide(_rf, _rb, true,  false);",
            ),
            "backward": (
                "writeSide(_lf, _lb, false, LEFT_MOTOR_REVERSED);",
                "writeSide(_rf, _rb, false, false);",
            ),
            "turnLeft": (
                "writeSide(_lf, _lb, false, LEFT_MOTOR_REVERSED);",
                "writeSide(_rf, _rb, true,  false);",
            ),
            "turnRight": (
                "writeSide(_lf, _lb, true,  LEFT_MOTOR_REVERSED);",
                "writeSide(_rf, _rb, false, false);",
            ),
        }
        for method, lines in expected.items():
            body = _method_body(self.motor, method)
            for line in lines:
                self.assertIn(line, body)

    def test_truth_table_with_left_inversion_off_matches_original(self):
        self.assertEqual(_drive(True, False), ("HIGH", "LOW"))
        self.assertEqual(_drive(False, False), ("LOW", "HIGH"))

    def test_truth_table_with_left_inversion_on_reverses_left_only(self):
        table = {
            "forward":  {"left": _drive(True, True),  "right": _drive(True, False)},
            "backward": {"left": _drive(False, True), "right": _drive(False, False)},
            "left":     {"left": _drive(False, True), "right": _drive(True, False)},
            "right":    {"left": _drive(True, True),  "right": _drive(False, False)},
        }
        self.assertEqual(table["forward"],  {"left": ("LOW", "HIGH"), "right": ("HIGH", "LOW")})
        self.assertEqual(table["backward"], {"left": ("HIGH", "LOW"), "right": ("LOW", "HIGH")})
        self.assertEqual(table["left"],     {"left": ("HIGH", "LOW"), "right": ("HIGH", "LOW")})
        self.assertEqual(table["right"],    {"left": ("LOW", "HIGH"), "right": ("LOW", "HIGH")})

    def test_stop_safe_state_unchanged(self):
        body = _method_body(self.motor, "stop")
        self.assertIn("_speed = MOTOR_SPEED_MIN;", body)
        self.assertEqual(body.count("digitalWrite"), 4)
        self.assertEqual(body.count("HIGH"), 4)
        self.assertIn("analogWrite(_lpwm, MOTOR_SPEED_MIN);", body)
        self.assertIn("analogWrite(_rpwm, MOTOR_SPEED_MIN);", body)
        self.assertNotIn("writeSide", body)


if __name__ == "__main__":
    unittest.main()
