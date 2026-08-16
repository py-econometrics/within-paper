"""Checks for the worker- and firm-specific year-slope OLS specification."""

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

import pandas as pd

from benchmarks.akm import SCENARIOS
from benchmarks.data import BASE_DESIGNS, make_base_data
from benchmarks.ols import akm as akm_benchmark
from benchmarks.ols import main as main_benchmark
from benchmarks.ols import pyfixest as ols_pyfixest
from benchmarks.ols import run as ols_runner
from benchmarks.ols.specifications import (
    INTERCEPTS,
    OlsSpecification,
    VARYING_SLOPE_BACKENDS,
    WORKER_FIRM_YEAR_SLOPES,
    WORKER_FIRM_YEAR_SLOPES_SPECIFICATION,
)
from scripts import paper_results


class VaryingSlopeSpecificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("BENCH_THREADS", "1")
        os.environ.setdefault("RAYON_NUM_THREADS", "1")

    def test_formula_and_supported_backends_are_fixed(self) -> None:
        formula = ols_pyfixest.formula_for_specification(
            ("indiv_id", "firm_id", "year"), WORKER_FIRM_YEAR_SLOPES
        )
        self.assertEqual(
            formula, "y ~ x1 | indiv_id[year] + firm_id[year] + year"
        )
        self.assertEqual(
            VARYING_SLOPE_BACKENDS,
            ("within-diagonal", "within-additive", "fixest", "FEM.jl"),
        )
        self.assertNotIn("rust-map", VARYING_SLOPE_BACKENDS)
        self.assertNotIn("within-off", VARYING_SLOPE_BACKENDS)

    def test_demeaners_explicitly_use_within_diagonal_and_additive(self) -> None:
        lsmr = Mock()
        fake_pyfixest = SimpleNamespace(LsmrDemeaner=lsmr)
        with patch.dict(sys.modules, {"pyfixest": fake_pyfixest}):
            ols_pyfixest.demeaner("within-diagonal")
            ols_pyfixest.demeaner("within-additive")

        self.assertEqual(
            lsmr.call_args_list,
            [
                call(backend="within", preconditioner="diagonal"),
                call(backend="within", preconditioner="additive"),
            ],
        )

    def test_shared_sample_writer_preserves_numeric_year(self) -> None:
        frame = make_base_data(1_000, "simple", 22)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            ols_runner._write_sample(lambda: frame, path)
            returned = pd.read_parquet(path)

        self.assertEqual(returned.columns.tolist(), frame.columns.tolist())
        pd.testing.assert_series_equal(returned["year"], frame["year"])
        self.assertTrue(returned.groupby("indiv_id")["year"].nunique().eq(10).all())

    def test_python_preconditioners_return_the_same_coefficient(self) -> None:
        frame = make_base_data(1_000, "simple", 22)
        estimates = [
            float(
                ols_pyfixest.fit_ols(
                    frame,
                    backend,
                    ("indiv_id", "firm_id", "year"),
                    specification=WORKER_FIRM_YEAR_SLOPES,
                )
                .coef()
                .loc["x1"]
            )
            for backend in ("within-diagonal", "within-additive")
        ]
        self.assertLess(abs(estimates[0] - estimates[1]), 1e-10)


class VaryingSlopePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["BENCH_THREADS"] = "1"

    @staticmethod
    def _successful_row(backend: str, repetitions: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "backend": backend,
                    "preconditioner": backend.removeprefix("within-"),
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

    def test_main_and_akm_tasks_add_the_specification(self) -> None:
        with patch.object(main_benchmark, "run_experiment") as run_main:
            main_benchmark.main()
        main_call = run_main.call_args.kwargs
        self.assertEqual(
            main_call["additional_specifications"],
            (WORKER_FIRM_YEAR_SLOPES_SPECIFICATION,),
        )
        self.assertEqual(
            [name for name, _generate in main_call["designs"]],
            [name for name, _seed in BASE_DESIGNS],
        )

        with patch.object(akm_benchmark, "run_experiment") as run_akm:
            akm_benchmark.main()
        akm_call = run_akm.call_args.kwargs
        self.assertEqual(
            akm_call["additional_specifications"],
            (WORKER_FIRM_YEAR_SLOPES_SPECIFICATION,),
        )
        self.assertEqual(
            [name for name, _generate in akm_call["designs"]], list(SCENARIOS)
        )

    def test_runner_reuses_one_sample_for_both_specifications(self) -> None:
        generated = 0
        seen_specifications = []

        def generate() -> pd.DataFrame:
            nonlocal generated
            generated += 1
            return make_base_data(1_000, "simple", 22)

        def fake_process(target, *args, **_kwargs):
            if target is ols_runner._write_sample:
                target(*args)
                return None
            _, output, _fixed_effects, backend, repetitions, _, _, specification = args
            seen_specifications.append(specification)
            self._successful_row(backend, repetitions).to_csv(output, index=False)
            return None

        varying = OlsSpecification(
            name=WORKER_FIRM_YEAR_SLOPES,
            fixed_effects=("indiv_id", "firm_id", "year"),
            backends=("within-diagonal", "within-additive"),
            repetitions=1,
            varying_slope="indiv_id[year]+firm_id[year]",
        )
        with patch.object(ols_runner, "_run_process", side_effect=fake_process):
            result = ols_runner.run_experiment(
                experiment="test",
                designs=[("simple", generate)],
                output=None,
                backends=("within-diagonal",),
                repetitions=1,
                additional_specifications=(varying,),
            )

        self.assertEqual(generated, 1)
        self.assertEqual(
            seen_specifications,
            [INTERCEPTS, WORKER_FIRM_YEAR_SLOPES, WORKER_FIRM_YEAR_SLOPES],
        )
        self.assertEqual(
            result.groupby("specification").size().to_dict(),
            {INTERCEPTS: 1, WORKER_FIRM_YEAR_SLOPES: 2},
        )

    def test_runner_records_a_crashed_varying_slope_cell(self) -> None:
        def fake_process(target, *args, **_kwargs):
            if target is ols_runner._write_sample:
                target(*args)
                return None
            _, output, _fixed_effects, backend, repetitions, _, _, specification = args
            if (
                specification == WORKER_FIRM_YEAR_SLOPES
                and backend == "within-diagonal"
            ):
                return "python estimator worker exited with status 1"
            self._successful_row(backend, repetitions).to_csv(output, index=False)
            return None

        varying = OlsSpecification(
            name=WORKER_FIRM_YEAR_SLOPES,
            fixed_effects=("indiv_id", "firm_id", "year"),
            backends=("within-diagonal", "within-additive"),
            repetitions=1,
            varying_slope="indiv_id[year]+firm_id[year]",
        )
        with patch.object(ols_runner, "_run_process", side_effect=fake_process):
            result = ols_runner.run_experiment(
                experiment="test",
                designs=[
                    ("simple", partial(make_base_data, 1_000, "simple", 23))
                ],
                output=None,
                backends=(),
                repetitions=1,
                additional_specifications=(varying,),
            )

        self.assertEqual(result["converged"].tolist(), [False, True])
        self.assertIn("exited with status 1", result.loc[0, "error"])


class VaryingSlopePaperTests(unittest.TestCase):
    @staticmethod
    def _rows(design: str, backend: str, n_obs: int) -> list[dict[str, object]]:
        return [
            {
                "design": design,
                "backend": backend,
                "repetition": repetition,
                "n_planned": 3,
                "runtime_s": repetition + 1,
                "converged": True,
                "capped": False,
                "n_obs": n_obs,
                "n_fe": 3,
                "view": "default",
                "specification": WORKER_FIRM_YEAR_SLOPES,
                "varying_slope": "indiv_id[year]+firm_id[year]",
            }
            for repetition in range(3)
        ]

    def test_collector_reads_varying_slopes_from_existing_ols_files(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        backends = VARYING_SLOPE_BACKENDS
        ols_rows = [
            row
            for backend in backends
            for row in self._rows("simple", backend, 10_000_000)
        ]
        akm_rows = [
            row
            for design in ("akm_mobility_1", "akm_sorting_1")
            for backend in backends
            for row in self._rows(design, backend, 1_000_000)
        ]
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory)
            pd.DataFrame(ols_rows).to_csv(latest / "ols.csv", index=False)
            pd.DataFrame(akm_rows).to_csv(latest / "akm.csv", index=False)
            with patch.object(paper_results, "LATEST_RUN", latest):
                paper_results._synchronize_canonical_tables(document, write=False)

        self.assertEqual(
            document["tables"]["varying_slopes_base"]["rows"][0][1:],
            ["2.00s"] * 4,
        )
        self.assertEqual(
            document["tables"]["varying_slopes_mobility"]["rows"][0][1:],
            ["2.00s"] * 4,
        )
        self.assertEqual(
            document["tables"]["varying_slopes_sorting"]["rows"][0][1:],
            ["2.00s"] * 4,
        )

    def test_old_ols_files_preserve_collected_varying_slope_values(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        previous = document["tables"]["varying_slopes_mobility"]["rows"][0][1]
        old_row = {
            **self._rows("akm_mobility_1", "within-diagonal", 1_000_000)[0],
            "specification": INTERCEPTS,
            "varying_slope": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory)
            pd.DataFrame([old_row]).to_csv(latest / "akm.csv", index=False)
            with patch.object(paper_results, "LATEST_RUN", latest):
                paper_results._synchronize_canonical_tables(document, write=False)

        self.assertEqual(
            document["tables"]["varying_slopes_mobility"]["rows"][0][1],
            previous,
        )

    def test_render_uses_design_parameters_and_public_method_labels(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        mobility = paper_results._table_fragment(
            "varying_slopes_mobility",
            document["tables"]["varying_slopes_mobility"],
        )
        sorting = paper_results._table_fragment(
            "varying_slopes_sorting",
            document["tables"]["varying_slopes_sorting"],
        )

        self.assertIn("Move probability $delta$", mobility)
        self.assertIn("Sorting strength $rho$", sorting)
        self.assertIn("factor-pair #linebreak() Schwarz", mobility)
        self.assertNotIn("akm_mobility_1", mobility)
        self.assertNotIn("akm_sorting_1", sorting)
        for label in ("[1]", "[0.5]", "[0.05]", "[0.01]", "[0.005]", "[0.001]"):
            self.assertIn(label, mobility)
        for label in ("[0]", "[20]", "[500]", "[2000]", "[10000]", "[150000]"):
            self.assertIn(label, sorting)


if __name__ == "__main__":
    unittest.main()
