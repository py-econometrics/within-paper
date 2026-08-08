from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmarks.tolerance.measure import TOLERANCES, coefficient_error_se, residual_error
from scripts import make_figures
from scripts.benchmark_methods import METHODS
from scripts.make_figures import CROSS_PACKAGE_BACKENDS, MECHANISM_BACKENDS
from scripts.plot_tolerance import (
    METHOD_ORDER,
    _axis_limits,
    aggregate_results,
    tolerance_figure,
)


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

    def test_error_axes_put_greater_precision_on_the_right(self) -> None:
        points = aggregate_results(self._raw_results())
        for metric in ("coefficient_error_se", "residual_error"):
            limits, _floor = _axis_limits(points, metric)
            self.assertGreater(limits[0], limits[1])

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
            "GLFEM.jl",
        }
        for key in keys:
            self.assertIn(key, METHODS)
            self.assertEqual(len(METHODS[key]), 4)

    def test_make_figures_includes_the_tolerance_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            figures = root / "figures"
            results.mkdir()
            pd.DataFrame({"placeholder": [1]}).to_csv(
                results / "tolerance_frontier.csv", index=False
            )
            with (
                patch.object(make_figures, "ROOT", root),
                patch.object(make_figures, "RESULTS", results),
                patch.object(make_figures, "FIGURES", figures),
                patch.object(make_figures, "_load_points", return_value=[]),
                patch.object(make_figures, "headline_figure"),
                patch.object(make_figures, "tolerance_figure") as tolerance,
            ):
                make_figures.main()
            tolerance.assert_called_once()
            self.assertEqual(
                tolerance.call_args.args[1], figures / "tolerance_frontier.svg"
            )


if __name__ == "__main__":
    unittest.main()
