from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.tolerance.measure import TOLERANCES, coefficient_error_se, residual_error
from scripts.benchmark_methods import METHODS, linestyle, style
from scripts.make_figures import CROSS_PACKAGE_BACKENDS, CROSSOVER_FILES, MECHANISM_BACKENDS
from scripts.plot_tolerance import METHOD_ORDER, aggregate_results, tolerance_figure


class ToleranceMetricTests(unittest.TestCase):
    def test_coefficient_error_is_in_reference_se_units(self) -> None:
        self.assertAlmostEqual(coefficient_error_se(1.01, 1.0, 0.02), 0.5)

    def test_residual_error_uses_reference_residual_norm(self) -> None:
        self.assertAlmostEqual(
            residual_error(np.array([6.0, 8.0]), np.array([3.0, 4.0])), 1.0
        )

    def test_residual_error_rejects_different_samples(self) -> None:
        with self.assertRaises(ValueError):
            residual_error(np.ones(2), np.ones(3))

    def test_default_tolerance_is_in_each_grid(self) -> None:
        for default, grid in TOLERANCES.values():
            self.assertIn(default, grid)


class TolerancePlotTests(unittest.TestCase):
    @staticmethod
    def _raw_results() -> pd.DataFrame:
        rows = []
        for design in ("akm_mobility_1", "akm_mobility_3", "akm_mobility_5"):
            for method_number, (backend, (default, grid)) in enumerate(
                TOLERANCES.items(), start=1
            ):
                for tolerance_number, tolerance in enumerate(grid[:2], start=1):
                    for repetition in range(3):
                        rows.append(
                            {
                                "design": design,
                                "backend": backend,
                                "tolerance": tolerance,
                                "default_tolerance": default,
                                "runtime_s": method_number * tolerance_number + repetition / 10,
                                "converged": True,
                                "coefficient_error_se": tolerance * method_number,
                                "residual_error": tolerance * tolerance_number,
                            }
                        )
        return pd.DataFrame(rows)

    def test_aggregation_returns_one_row_per_setting(self) -> None:
        points = aggregate_results(self._raw_results())
        self.assertEqual(len(points), 3 * 2 * len(TOLERANCES))
        self.assertTrue((points["n_success"] == 3).all())

    def test_figure_writes_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tolerance.svg"
            tolerance_figure(self._raw_results(), output)
            self.assertIn("<svg", output.read_text(encoding="utf-8")[:500])

    def test_every_figure_key_has_one_presentation_record(self) -> None:
        keys = {
            *CROSS_PACKAGE_BACKENDS,
            *MECHANISM_BACKENDS,
            *METHOD_ORDER,
            *CROSSOVER_FILES["OLS"][1],
            *CROSSOVER_FILES["PPML"][1],
        }
        for key in keys:
            self.assertIn(key, METHODS)
            self.assertEqual(len(style(key)), 2)
            self.assertTrue(linestyle(key))


if __name__ == "__main__":
    unittest.main()
