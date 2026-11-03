#!/usr/bin/env python3
"""Ensure invalid numerical results cannot pass the validation gates."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from validate import matrix_errors, validate_boys


class ValidationGuards(unittest.TestCase):
    def test_known_matrix_errors(self):
        identity = np.eye(2)
        actual = (np.array([[1.0, 0.25], [0.25, 1.0]]),
                  2 * identity, -identity)
        reference = (identity, identity, -2 * identity)
        absolute, relative = matrix_errors(actual, reference)
        np.testing.assert_allclose(absolute, [0.25, 1.0, 1.0], rtol = 0, atol = 0)
        np.testing.assert_allclose(relative, [0.25, 1.0, 0.5], rtol = 1e-15, atol = 0)

    def test_nonfinite_matrices(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for side in (0, 1):
                with self.subTest(value = value, side = side):
                    arguments = [tuple(np.eye(2) for _ in range(3)) for _ in range(2)]
                    arguments[side][1][0, 1] = value
                    with self.assertRaisesRegex(RuntimeError, "Nonfinite matrix"):
                        matrix_errors(*arguments)

    def test_matrix_dimensions(self):
        valid = tuple(np.eye(2) for _ in range(3))
        malformed = [
            (np.eye(3), *valid[1:]),
            (np.ones((2, 3)), *valid[1:]),
            (np.ones(2), *valid[1:]),
            tuple(np.empty((0, 0)) for _ in range(3)),
            valid[:2],
        ]
        for matrices in malformed:
            with self.subTest(shapes = [matrix.shape for matrix in matrices]):
                with self.assertRaises(RuntimeError):
                    matrix_errors(matrices, valid)
        for shape in ((2,), (2, 3), (0, 0)):
            with self.subTest(equal_invalid_shape = shape):
                bad = tuple(np.ones(shape) for _ in range(3))
                with self.assertRaises(RuntimeError):
                    matrix_errors(bad, bad)

    def test_probe_rejects_nonfinite_or_incomplete_moments(self):
        # Exercise the subprocess boundary, rather than substitute validator
        # internals: a faulty executable must never produce a passing report.
        for token in ("nan", "inf", "-inf", "incomplete"):
            with self.subTest(token = token), tempfile.TemporaryDirectory() as folder:
                probe = Path(folder) / "faulty_probe"
                body = ("print('0.5')\n" if token == "incomplete" else
                        f"for line in sys.stdin:\n    print({token!r})\n")
                probe.write_text(f"#!{sys.executable}\nimport sys\n" + body)
                probe.chmod(0o700)
                output = Path(folder) / "should_not_pass.csv"
                with self.assertRaises(RuntimeError):
                    validate_boys(probe, seed = 20260615, out = output)
                self.assertFalse(output.exists())

    def test_cli_rejects_invalid_tolerances_before_loading_library(self):
        arguments = [
            ("validate.py", "--matrix-tolerance"),
            ("validate.py", "--boys-tolerance"),
            ("validate_matrix.py", "--matrix-tolerance"),
        ]
        for script, option in arguments:
            for value in ("nan", "inf", "-inf", "0", "-1"):
                with self.subTest(script = script, option = option, value = value):
                    completed = subprocess.run(
                        [sys.executable, "-B", str(SCRIPTS / script), f"{option}={value}"],
                        text=True, capture_output=True,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("must be finite and positive", completed.stderr)


if __name__ == "__main__":
    unittest.main()
