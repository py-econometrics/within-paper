from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "modular"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_tolerance import (  # noqa: E402
    METHOD_BY_KEY,
    coefficient_error_se,
    residual_error,
)
from plot_tolerance import aggregate_results, tolerance_figure  # noqa: E402


class ToleranceMetricTests(unittest.TestCase):
    def test_coefficient_error_is_in_reference_se_units(self) -> None:
        self.assertAlmostEqual(coefficient_error_se(1.01, 1.0, 0.02), 0.5)

    def test_residual_error_uses_reference_residual_norm(self) -> None:
        reference = np.array([3.0, 4.0])
        residual = np.array([6.0, 8.0])
        self.assertAlmostEqual(residual_error(residual, reference), 1.0)

    def test_residual_error_rejects_different_samples(self) -> None:
        with self.assertRaises(ValueError):
            residual_error(np.ones(2), np.ones(3))

    def test_default_grids_include_default_tolerances(self) -> None:
        self.assertEqual(
            set(METHOD_BY_KEY),
            {
                "lsmr_off",
                "lsmr_diagonal",
                "lsmr_additive",
                "pyfixest_map",
                "r_fixest",
                "julia_fem",
            },
        )
        for method in METHOD_BY_KEY.values():
            self.assertIn(method.default_tolerance, method.tolerances)


class TolerancePlotTests(unittest.TestCase):
    @staticmethod
    def _raw_results() -> pd.DataFrame:
        rows = []
        for design in ("akm_mobility_1", "akm_mobility_3", "akm_mobility_5"):
            for method_number, method in enumerate(METHOD_BY_KEY.values(), start=1):
                for tolerance_number, tolerance in enumerate(
                    method.tolerances[:2], start=1
                ):
                    for repetition in (1, 2, 3):
                        rows.append(
                            {
                                "design": design,
                                "method": method.key,
                                "label": method.label,
                                "tolerance": tolerance,
                                "default_tolerance": method.default_tolerance,
                                "time_s": method_number * tolerance_number
                                + repetition / 10,
                                "success": True,
                                "coefficient_error_se": tolerance * method_number,
                                "residual_error": tolerance * tolerance_number,
                            }
                        )
        return pd.DataFrame(rows)

    def test_aggregation_returns_one_row_per_setting(self) -> None:
        points = aggregate_results(self._raw_results())
        expected = 3 * sum(
            min(2, len(method.tolerances)) for method in METHOD_BY_KEY.values()
        )
        self.assertEqual(len(points), expected)
        self.assertTrue((points["n_success"] == 3).all())

    def test_figure_writes_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tolerance.svg"
            tolerance_figure(self._raw_results(), output)
            self.assertTrue(output.exists())
            self.assertIn("<svg", output.read_text(encoding="utf-8")[:500])


if __name__ == "__main__":
    unittest.main()
