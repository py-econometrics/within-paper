from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from benchmarks.modular.benchmark_tolerance import (
    METHOD_BY_KEY,
    coefficient_error_se,
    residual_error,
)
from scripts.figure_style import METHOD_STYLE
from scripts.make_figures import CROSSOVER_STYLE, STYLE as PAPER_FIGURE_STYLE
from scripts.plot_tolerance import (
    STYLE as TOLERANCE_STYLE,
    aggregate_results,
    tolerance_figure,
)


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

    def test_method_styles_are_shared_across_paper_figures(self) -> None:
        self.assertEqual(PAPER_FIGURE_STYLE["rust-map"], METHOD_STYLE["map"])
        self.assertEqual(PAPER_FIGURE_STYLE["fixest"], METHOD_STYLE["fixest"])
        self.assertEqual(PAPER_FIGURE_STYLE["FEM.jl"], METHOD_STYLE["fem"])
        self.assertEqual(
            PAPER_FIGURE_STYLE["within"], METHOD_STYLE["lsmr_factor_pair"]
        )
        self.assertEqual(TOLERANCE_STYLE["pyfixest_map"], METHOD_STYLE["map"])
        self.assertEqual(TOLERANCE_STYLE["r_fixest"], METHOD_STYLE["fixest"])
        self.assertEqual(TOLERANCE_STYLE["julia_fem"], METHOD_STYLE["fem"])
        self.assertEqual(
            CROSSOVER_STYLE["Julia"][1:], METHOD_STYLE["fem"]
        )


if __name__ == "__main__":
    unittest.main()
