"""Checks for the standalone worker-specific year-slope benchmark."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from benchmarks.data import make_base_data
from benchmarks.varying_slopes import pyfixest as varying_pyfixest
from benchmarks.varying_slopes import run as varying_runner
from scripts import paper_results


class VaryingSlopeSpecificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("BENCH_THREADS", "1")
        os.environ.setdefault("RAYON_NUM_THREADS", "1")

    def test_selected_designs_and_formula_are_fixed(self) -> None:
        self.assertEqual(
            varying_runner.DESIGNS,
            ("akm_mobility_1", "akm_mobility_3", "akm_mobility_5"),
        )
        self.assertEqual(
            varying_pyfixest.FORMULA,
            "y ~ x1 | indiv_id[year] + firm_id + year",
        )

    def test_demeaners_use_only_within_diagonal_and_additive(self) -> None:
        lsmr = Mock()
        fake_pyfixest = SimpleNamespace(LsmrDemeaner=lsmr)
        with patch.dict(sys.modules, {"pyfixest": fake_pyfixest}):
            varying_pyfixest.demeaner("within-diagonal")
            varying_pyfixest.demeaner("within-additive")

        self.assertEqual(
            lsmr.call_args_list,
            [
                call(backend="within", preconditioner="diagonal"),
                call(backend="within", preconditioner="additive"),
            ],
        )

    def test_sample_writer_preserves_the_numeric_year_column(self) -> None:
        frame = make_base_data(1_000, "simple", 22)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            varying_runner._write_sample(lambda: frame, path)
            returned = pd.read_parquet(path)

        self.assertEqual(returned.columns.tolist(), frame.columns.tolist())
        pd.testing.assert_series_equal(returned["year"], frame["year"])

    def test_python_preconditioners_return_the_same_coefficient(self) -> None:
        frame = make_base_data(1_000, "simple", 22)
        estimates = [
            float(varying_pyfixest.fit_varying_slope(frame, backend).coef().loc["x1"])
            for backend in ("within-diagonal", "within-additive")
        ]
        self.assertLess(abs(estimates[0] - estimates[1]), 1e-10)


class VaryingSlopeRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["BENCH_THREADS"] = "1"

    @staticmethod
    def _successful_row(backend: str, repetitions: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "backend": backend,
                    "preconditioner": "diagonal",
                    "package_version": "test",
                    "repetition": repetition,
                    "n_planned": repetitions,
                    "runtime_s": 0.01,
                    "n_retained": 1_000,
                    "beta_x1": 0.5,
                    "converged": True,
                    "capped": False,
                    "error": "",
                }
                for repetition in range(repetitions)
            ]
        )

    def test_runner_uses_isolated_python_cells_and_common_sample(self) -> None:
        seen_backends = []

        def fake_process(target, *args, **_kwargs):
            if target is varying_runner._write_sample:
                target(*args)
                return None
            _, output, backend, repetitions = args
            seen_backends.append(backend)
            self._successful_row(backend, repetitions).to_csv(output, index=False)
            return None

        with (
            patch.object(varying_runner, "_run_process", side_effect=fake_process),
            patch.object(varying_runner, "_native_rows", side_effect=AssertionError),
        ):
            result = varying_runner.run_experiment(
                designs=[
                    ("akm_mobility_1", partial(make_base_data, 1_000, "simple", 22))
                ],
                output=None,
                backends=("within-diagonal", "within-additive"),
                repetitions=1,
            )

        self.assertEqual(seen_backends, ["within-diagonal", "within-additive"])
        self.assertEqual(result["varying_slope"].unique().tolist(), ["indiv_id[year]"])
        self.assertEqual(result["n_obs"].unique().tolist(), [1_000])

    def test_runner_records_a_crashed_python_cell_and_continues(self) -> None:
        def fake_process(target, *args, **_kwargs):
            if target is varying_runner._write_sample:
                target(*args)
                return None
            _, output, backend, repetitions = args
            if backend == "within-diagonal":
                return "python estimator worker exited with status 1"
            self._successful_row(backend, repetitions).to_csv(output, index=False)
            return None

        with patch.object(varying_runner, "_run_process", side_effect=fake_process):
            result = varying_runner.run_experiment(
                designs=[
                    ("akm_mobility_1", partial(make_base_data, 1_000, "simple", 23))
                ],
                output=None,
                backends=("within-diagonal", "within-additive"),
                repetitions=1,
            )

        self.assertEqual(result["converged"].tolist(), [False, True])
        self.assertIn("exited with status 1", result.loc[0, "error"])


class VaryingSlopePaperTests(unittest.TestCase):
    @staticmethod
    def _rows(backend: str) -> list[dict[str, object]]:
        return [
            {
                "design": "akm_mobility_1",
                "backend": backend,
                "repetition": repetition,
                "n_planned": 3,
                "runtime_s": repetition + 1,
                "converged": True,
                "capped": False,
                "n_obs": 1_000_000,
                "n_fe": 3,
                "view": "default",
                "varying_slope": "indiv_id[year]",
            }
            for repetition in range(3)
        ]

    def test_collector_fills_only_the_matching_varying_slope_cells(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        rows = [
            *self._rows("within-diagonal"),
            *self._rows("within-additive"),
            *self._rows("fixest"),
            *self._rows("FEM.jl"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory)
            pd.DataFrame(rows).to_csv(latest / "varying_slopes.csv", index=False)
            with patch.object(paper_results, "LATEST_RUN", latest):
                paper_results._synchronize_canonical_tables(document, write=False)

        table = document["tables"]["varying_slopes"]
        self.assertEqual(table["rows"][0][1:], ["2.00s"] * 4)
        self.assertEqual(table["rows"][1][1:], ["#miss"] * 4)

    def test_absent_raw_file_preserves_collected_timing(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        document["tables"]["varying_slopes"]["rows"][0][1] = "2.00s"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(paper_results, "LATEST_RUN", Path(directory)):
                paper_results._synchronize_canonical_tables(document, write=False)
        self.assertEqual(document["tables"]["varying_slopes"]["rows"][0][1], "2.00s")

    def test_render_uses_move_probabilities_and_public_method_labels(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        rendered = paper_results._table_fragment(
            "varying_slopes", document["tables"]["varying_slopes"]
        )

        self.assertIn("Move probability $delta$", rendered)
        self.assertIn("factor-pair #linebreak() Schwarz", rendered)
        self.assertNotIn("akm_mobility_1", rendered)
        for label in ("[1]", "[0.05]", "[0.005]"):
            self.assertIn(label, rendered)


if __name__ == "__main__":
    unittest.main()
