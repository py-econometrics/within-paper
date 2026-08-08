"""Numerical and paper-facing checks for the compact benchmark code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmarks import runtime
from benchmarks.accuracy import external_normal_residuals, projection_errors
from benchmarks.akm import AKMConfig, SCENARIOS, simulate_akm_panel
from benchmarks.data import drop_singletons, make_base_data, solver_data
from benchmarks.ols import agreement as ols_agreement, pyfixest as ols_pyfixest, run as ols_runner
from benchmarks.ols.pyfixest import fit_ols
from benchmarks.ols.run import PACKAGE_RUNTIME_BACKENDS, run_experiment
from benchmarks.ppml import pyfixest as ppml_pyfixest
from benchmarks.ppml.pyfixest import fit_ppml
from benchmarks.within import amortization, scaling, setup_cost
from benchmarks.within.map import map_demean_with_sweeps
from scripts import analyze_gap_runtime, compute_hardness, paper_results
from scripts.paper_results import _render_trial_result


class DataTests(unittest.TestCase):
    def test_base_data_is_deterministic_and_has_the_paper_columns(self) -> None:
        first = make_base_data(1_000, "simple", 42)
        second = make_base_data(1_000, "simple", 42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            list(first), ["indiv_id", "firm_id", "year", "y", "negbin_y", "x1"]
        )
        self.assertAlmostEqual(first.iloc[0]["y"], 3.3482659592032387)

    def test_akm_designs_and_small_draw_are_deterministic(self) -> None:
        self.assertEqual(len(SCENARIOS), 12)
        sorting = [
            SCENARIOS[f"akm_sorting_{index}"]
            for index in range(1, 7)
        ]
        self.assertEqual([item["delta"] for item in sorting], [1.0] * 6)
        self.assertEqual(
            [item["rho"] for item in sorting],
            [0.0, 20.0, 500.0, 2_000.0, 10_000.0, 150_000.0],
        )
        config = AKMConfig(
            n_workers=8, n_firms=4, n_time=3, n_industries=2, n_match_bins=4
        )
        first = simulate_akm_panel(config, seed=7)
        second = simulate_akm_panel(config, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.shape, (24, 5))
        self.assertAlmostEqual(first.iloc[0]["x1"], -1.9924197841744944)

    def test_extreme_sorting_has_valid_firm_assignments(self) -> None:
        config = AKMConfig(
            n_workers=20,
            n_firms=8,
            n_time=3,
            n_industries=2,
            n_match_bins=8,
            delta=1.0,
            rho=150_000.0,
        )
        frame = simulate_akm_panel(config, seed=7)
        self.assertTrue(frame["firm_id"].between(1, config.n_firms).all())

    def test_solver_data_factorizes_every_fixed_effect(self) -> None:
        categories, rhs = solver_data(make_base_data(1_000, "difficult", 43))
        self.assertEqual(categories.shape, (1_000, 3))
        self.assertEqual(rhs.shape, (1_000, 2))
        self.assertTrue(categories.flags.f_contiguous)
        self.assertTrue(rhs.flags.f_contiguous)

    def test_singleton_pruning_returns_the_retained_frame(self) -> None:
        frame = pd.DataFrame(
            {"a": [1, 1, 2], "b": [1, 1, 2], "value": [4, 5, 6]}
        )
        retained, dropped = drop_singletons(frame, ("a", "b"))
        self.assertEqual(dropped, 1)
        self.assertEqual(retained["value"].tolist(), [4, 5])


class AccuracyTests(unittest.TestCase):
    def test_exact_demeaning_has_small_external_residual(self) -> None:
        categories = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        rhs = np.array([1.0, 2.0, 3.0, 4.0])[:, None]
        demeaned = np.zeros_like(rhs)
        self.assertLess(external_normal_residuals(categories, rhs, demeaned)[0], 1e-14)

    def test_projection_error_uses_the_original_rhs_norm(self) -> None:
        rhs = np.array([3.0, 4.0])[:, None]
        reference = np.zeros_like(rhs)
        result = rhs.copy()
        self.assertAlmostEqual(projection_errors(result, reference, rhs)[0], 1.0)


class MapTests(unittest.TestCase):
    def test_map_reports_the_iteration_cap(self) -> None:
        categories = np.array([[0, 0], [0, 1], [1, 1], [1, 2]], dtype=np.uint32)
        result = map_demean_with_sweeps(
            np.arange(4.0)[:, None], categories, tol=0.0, maxiter=1
        )
        self.assertEqual(result.iterations, [1])
        self.assertEqual(result.converged, [False])

    def test_zero_iteration_cap_reports_zero_sweeps(self) -> None:
        categories = np.array([[0, 0], [1, 1]], dtype=np.uint32)
        result = map_demean_with_sweeps(
            np.arange(2.0)[:, None], categories, maxiter=0
        )
        self.assertEqual(result.iterations, [0])
        self.assertEqual(result.converged, [False])


class FailureLoggingTests(unittest.TestCase):
    def test_failure_fields_distinguish_caps_from_other_errors(self) -> None:
        capped = runtime.failure_fields(
            ValueError("Demeaning failed after 10000 iterations.")
        )
        failed = runtime.failure_fields(ValueError("invalid estimator input"))

        self.assertTrue(capped["capped"])
        self.assertFalse(failed["capped"])
        self.assertFalse(capped["converged"])
        self.assertFalse(failed["converged"])

    def test_native_driver_failure_becomes_a_complete_failed_cell(self) -> None:
        error = subprocess.CalledProcessError(
            1, ["Rscript"], stderr="estimator process failed"
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runtime.subprocess, "run", side_effect=error),
        ):
            rows = runtime.run_native(
                Path("driver.R"),
                [],
                Path(directory) / "missing.csv",
                backend="fixest",
                failure_repetitions=3,
            )

        self.assertEqual([row["repetition"] for row in rows], [0, 1, 2])
        self.assertTrue(all(not row["converged"] for row in rows))
        self.assertTrue(all("estimator process failed" in row["error"] for row in rows))

    def test_keyboard_interrupt_is_not_converted_to_estimator_failure(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runtime.subprocess, "run", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            runtime.run_native(
                Path("driver.R"),
                [],
                Path(directory) / "missing.csv",
                backend="fixest",
                failure_repetitions=3,
            )

    def test_standalone_scaling_exception_becomes_a_failed_row(self) -> None:
        categories = np.zeros((4, 2), dtype=np.uint32)
        rhs = np.ones((4, 1))
        with (
            patch.object(scaling, "solve_batch", return_value=None),
            patch.object(scaling, "Solver", side_effect=RuntimeError("solver failed")),
        ):
            row = scaling._measure(categories, rhs, "additive")

        self.assertFalse(row["converged"])
        self.assertFalse(row["capped"])
        self.assertEqual(row["error"], "solver failed")

class HardnessTests(unittest.TestCase):
    def test_complete_bipartite_graph_has_unit_gap(self) -> None:
        block = sp.csr_matrix(np.ones((3, 4)))
        self.assertAlmostEqual(compute_hardness._component_rho(block), 0.0)

    def test_sparse_calculation_falls_back_from_propack_to_arpack(self) -> None:
        calls = []

        def fake_svds(*args, solver, **kwargs):
            calls.append(solver)
            if solver == "propack":
                raise RuntimeError("not available")
            return np.array([0.5, 1.0])

        block = sp.eye(100, format="csr")
        with (
            patch.object(compute_hardness, "DENSE_MAX_ENTRIES", 0),
            patch.object(compute_hardness, "svds", side_effect=fake_svds),
        ):
            self.assertAlmostEqual(compute_hardness._component_rho(block), 0.25)
        self.assertEqual(calls, ["propack", "arpack"])


class GapAnalysisTests(unittest.TestCase):
    def test_final_runtime_rows_join_to_hardness_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            hardness = root / "hardness.csv"
            pd.DataFrame(
                [
                    {
                        "dataset_id": "simple",
                        "n_obs_raw": 1_000,
                        "fe_a": "indiv_id",
                        "fe_b": "firm_id",
                        "one_minus_rho": 0.25,
                        "worst_component_obs_share": 1.0,
                        "rho_qr": 0.75,
                        "kind": "base",
                    }
                ]
            ).to_csv(hardness, index=False)
            pd.DataFrame(
                [
                    {
                        "design": "simple",
                        "backend": "rust-map",
                        "view": "default",
                        "runtime_s": runtime,
                        "converged": True,
                        "n_obs": 1_000,
                        "n_fe": 3,
                    }
                    for runtime in (1.0, 2.0)
                ]
            ).to_csv(results / "ols.csv", index=False)

            report = analyze_gap_runtime.analyze(hardness, results)

        self.assertEqual(len(report["points"]), 1)
        self.assertEqual(report["points"][0]["view"], "default")
        self.assertEqual(report["points"][0]["median_time"], 1.5)


class PythonFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("BENCH_THREADS", "1")
        os.environ.setdefault("RAYON_NUM_THREADS", "1")

    def test_direct_ols_fit_has_named_x1_coefficient(self) -> None:
        frame = make_base_data(2_000, "simple", 12)
        fit = fit_ols(frame, "rust-map", ("indiv_id", "firm_id", "year"))
        self.assertTrue(np.isfinite(float(fit.coef().loc["x1"])))
        self.assertFalse(hasattr(fit, "_Y"))

    def test_package_runtime_ols_methods_include_all_lsmr_preconditioners(self) -> None:
        self.assertEqual(
            PACKAGE_RUNTIME_BACKENDS,
            (
                "rust-map",
                "within-off",
                "within-diagonal",
                "within",
                "fixest",
                "FEM.jl",
            ),
        )

    def test_agreement_keeps_its_explicit_four_backend_scope(self) -> None:
        self.assertEqual(
            ols_agreement.AGREEMENT_BACKENDS,
            ("rust-map", "within", "fixest", "FEM.jl"),
        )

    def test_lsmr_ablations_leave_tolerance_and_iteration_cap_at_defaults(self) -> None:
        lsmr = Mock()
        fake_pyfixest = SimpleNamespace(LsmrDemeaner=lsmr)
        with patch.dict(sys.modules, {"pyfixest": fake_pyfixest}):
            ols_pyfixest.demeaner("within-off")
            ols_pyfixest.demeaner("within-diagonal")

        self.assertEqual(
            lsmr.call_args_list,
            [call(preconditioner="off"), call(preconditioner="diagonal")],
        )

    def test_direct_ppml_fit_uses_three_fixed_effects(self) -> None:
        frame = make_base_data(2_000, "simple", 13)
        for backend in ("rust-map", "within"):
            fit = fit_ppml(frame, backend)
            self.assertTrue(np.isfinite(float(fit.coef().loc["x1"])))
            self.assertFalse(hasattr(fit, "_Y"))

    def test_ppml_measure_records_a_failed_warmup(self) -> None:
        error = ValueError("Demeaning failed after 10000 iterations.")
        with patch.object(ppml_pyfixest, "fit_ppml", side_effect=error):
            rows = ppml_pyfixest.measure(pd.DataFrame(), "rust-map", repetitions=3)

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not row["converged"] for row in rows))
        self.assertTrue(all(row["capped"] for row in rows))
        self.assertTrue(all(row["error"] == str(error) for row in rows))

    def test_ols_measure_records_a_failed_warmup(self) -> None:
        error = ValueError("Demeaning failed after 10000 iterations.")
        with patch.object(ols_pyfixest, "fit_ols", side_effect=error):
            rows = ols_pyfixest.measure(
                pd.DataFrame(), "rust-map", ("id1", "id2"), repetitions=3
            )

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not row["converged"] for row in rows))
        self.assertTrue(all(row["capped"] for row in rows))
        self.assertTrue(all(row["error"] == str(error) for row in rows))

    def test_measured_estimator_errors_are_failed_not_capped(self) -> None:
        error = ValueError("estimator rejected the model")
        with patch.object(ols_pyfixest, "fit_ols", side_effect=error):
            ols_rows = ols_pyfixest.measure(
                pd.DataFrame(), "rust-map", ("id1", "id2"), repetitions=2
            )
        with patch.object(ppml_pyfixest, "fit_ppml", side_effect=error):
            ppml_rows = ppml_pyfixest.measure(
                pd.DataFrame(), "rust-map", repetitions=2
            )

        for rows in (ols_rows, ppml_rows):
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(not row["converged"] for row in rows))
            self.assertTrue(all(not row["capped"] for row in rows))
            self.assertTrue(all(row["error"] == str(error) for row in rows))

    def test_ppml_outer_nonconvergence_is_capped(self) -> None:
        from pyfixest.errors import NonConvergenceError

        error = NonConvergenceError("The IRLS algorithm did not converge.")
        with patch.object(ppml_pyfixest, "fit_ppml", side_effect=error):
            rows = ppml_pyfixest.measure(pd.DataFrame(), "rust-map", repetitions=1)

        self.assertFalse(rows[0]["converged"])
        self.assertTrue(rows[0]["capped"])

    def test_ols_runner_uses_an_isolated_python_cell(self) -> None:
        result = run_experiment(
            experiment="test",
            designs=[("simple", partial(make_base_data, 2_000, "simple", 15))],
            output=None,
            backends=("rust-map",),
            repetitions=1,
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result.loc[0, "converged"])

    def test_ols_runner_records_package_default_retained_counts(self) -> None:
        def fake_process(target, *args, **_kwargs) -> None:
            if target is ols_runner._write_sample:
                target(*args)
                return
            _, output, _, backend, repetitions = args
            retained = 100 if backend == "rust-map" else 99
            pd.DataFrame(
                [{
                    "backend": backend,
                    "repetition": 0,
                    "n_planned": repetitions,
                    "runtime_s": 0.01,
                    "n_retained": retained,
                    "beta_x1": 1.0,
                    "max_eta": None,
                    "converged": True,
                    "error": "",
                }]
            ).to_csv(output, index=False)

        with patch.object(ols_runner, "_run_process", side_effect=fake_process):
            result = run_experiment(
                experiment="test",
                designs=[("simple", partial(make_base_data, 2_000, "simple", 16))],
                output=None,
                backends=("rust-map", "within"),
                repetitions=1,
            )

        self.assertEqual(result["n_retained"].tolist(), [100, 99])

    def test_ols_runner_routes_lsmr_ablations_to_isolated_python_workers(self) -> None:
        worker_backends = []

        def fake_process(target, *args, **_kwargs) -> None:
            if target is ols_runner._write_sample:
                target(*args)
                return None
            self.assertIs(target, ols_runner._python_rows)
            _, output, _, backend, repetitions = args
            worker_backends.append(backend)
            pd.DataFrame(
                [
                    {
                        "backend": backend,
                        "repetition": 0,
                        "n_planned": repetitions,
                        "runtime_s": 0.01,
                        "n_retained": 100,
                        "beta_x1": 1.0,
                        "max_eta": None,
                        "converged": True,
                        "capped": False,
                        "error": "",
                    }
                ]
            ).to_csv(output, index=False)
            return None

        with (
            patch.object(ols_runner, "_run_process", side_effect=fake_process),
            patch.object(ols_runner, "_native_rows", side_effect=AssertionError),
        ):
            result = run_experiment(
                experiment="test",
                designs=[("simple", partial(make_base_data, 2_000, "simple", 18))],
                output=None,
                backends=("within-off", "within-diagonal"),
                repetitions=1,
            )

        self.assertEqual(worker_backends, ["within-off", "within-diagonal"])
        self.assertEqual(result["backend"].tolist(), worker_backends)

    def test_ols_runner_continues_after_an_isolated_backend_crash(self) -> None:
        def fake_process(target, *args, **_kwargs):
            if target is ols_runner._write_sample:
                target(*args)
                return None
            _, output, _, backend, repetitions = args
            if backend == "rust-map":
                return "python estimator worker exited with status 1"
            pd.DataFrame(
                [
                    {
                        "backend": backend,
                        "repetition": 0,
                        "n_planned": repetitions,
                        "runtime_s": 0.01,
                        "n_retained": 99,
                        "beta_x1": 1.0,
                        "max_eta": None,
                        "converged": True,
                        "capped": False,
                        "error": "",
                    }
                ]
            ).to_csv(output, index=False)
            return None

        with patch.object(ols_runner, "_run_process", side_effect=fake_process):
            result = run_experiment(
                experiment="test",
                designs=[("simple", partial(make_base_data, 2_000, "simple", 17))],
                output=None,
                backends=("rust-map", "within"),
                repetitions=1,
            )

        self.assertEqual(result["converged"].tolist(), [False, True])
        self.assertIn("exited with status 1", result.loc[0, "error"])

class SetupCostTests(unittest.TestCase):
    def test_akm_mobility_setup_contract(self) -> None:
        self.assertEqual(
            setup_cost.MOBILITY_DESIGNS,
            tuple(name for name in SCENARIOS if name.startswith("akm_mobility_")),
        )
        self.assertEqual(setup_cost.AKM_REPETITIONS, 20)
        self.assertEqual(setup_cost.OPTIONS.tol, 1e-12)
        self.assertEqual(setup_cost.OPTIONS.maxiter, 10_000)

        categories, rhs = solver_data(make_base_data(1_000, "difficult", 43))
        rows = setup_cost._measure(
            "small",
            categories,
            rhs,
            experiment="test",
            repetitions=1,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["preconditioner"], "additive")
        self.assertEqual(row["n_planned"], 1)
        self.assertTrue(row["converged"])
        self.assertLess(row["max_eta"], 1e-8)
        self.assertLess(row["max_delta"], 1e-7)

    def test_setup_cost_rejects_a_capped_reference(self) -> None:
        categories = np.zeros((4, 2), dtype=np.uint32)
        rhs = np.ones((4, 1))
        warmup = SimpleNamespace(converged=[True], demeaned=rhs)
        reference = SimpleNamespace(converged=[False], demeaned=rhs)

        with patch.object(setup_cost, "solve_batch", side_effect=[warmup, reference]):
            with self.assertRaisesRegex(RuntimeError, "tight reference did not converge"):
                setup_cost._measure(
                    "capped-reference",
                    categories,
                    rhs,
                    experiment="test",
                    repetitions=1,
                )


class PaperResultTests(unittest.TestCase):
    @staticmethod
    def _rows(design: str, backend: str, n_obs: int, n_fe: int, view: str):
        return [
            {
                "design": design,
                "backend": backend,
                "repetition": repetition,
                "n_planned": 3,
                "runtime_s": repetition + 1,
                "converged": True,
                "n_obs": n_obs,
                "n_fe": n_fe,
                "view": view,
            }
            for repetition in range(3)
        ]

    def test_one_sample_with_three_repetitions_is_complete(self) -> None:
        rows = [
            {
                "repetition": str(index), "n_planned": "3",
                "runtime_s": str(index + 1), "converged": "true",
            }
            for index in range(3)
        ]
        self.assertEqual(_render_trial_result(rows), "2.00s")

    def test_failed_trials_remain_in_the_denominator(self) -> None:
        rows = [
            {"repetition": "0", "n_planned": "3", "runtime_s": "1", "converged": "true"},
            {"repetition": "1", "n_planned": "3", "runtime_s": "2", "converged": "false"},
            {"repetition": "2", "n_planned": "3", "runtime_s": "3", "converged": "true"},
        ]
        self.assertEqual(_render_trial_result(rows), "2.00s (2/3)")

    def test_capped_trials_are_labelled_in_the_paper_table(self) -> None:
        rows = [
            {
                "repetition": str(index), "n_planned": "3",
                "runtime_s": str(index + 1), "converged": "false", "capped": "true",
            }
            for index in range(3)
        ]
        self.assertEqual(_render_trial_result(rows), "capped (0/3)")

    def test_all_four_final_runtime_files_feed_the_paper_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory)
            fixtures = {
                "ols.csv": self._rows("simple", "rust-map", 10_000_000, 3, "default"),
                "ppml.csv": self._rows("simple", "rust-map", 1_000_000, 3, "default"),
                "akm.csv": [
                    *self._rows("akm_mobility_1", "rust-map", 1_000_000, 3, "default"),
                    *self._rows("akm_mobility_1", "rust-map", 1_000_000, 3, "matched"),
                ],
                "correia.csv": self._rows(
                    "synthetic-complete", "rust-map", 1_000, 2, "default"
                ),
            }
            for filename, rows in fixtures.items():
                pd.DataFrame(rows).to_csv(latest / filename, index=False)
            document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
            with patch.object(paper_results, "LATEST_RUN", latest):
                paper_results._synchronize_canonical_tables(document, write=False)
            self.assertEqual(document["tables"]["ols"]["rows"][0][2], "2.00s")
            self.assertEqual(document["tables"]["ppml"]["rows"][0][2], "2.00s")
            self.assertEqual(document["tables"]["akm_mobility"]["rows"][0][2], "2.00s")
            self.assertEqual(document["tables"]["mechanism_mobility"]["rows"][0][2], "2.00s")
            self.assertEqual(document["tables"]["correia_synthetic"]["rows"][0][2], "2.00s")

    def test_five_ols_runtime_tables_reserve_default_lsmr_ablation_columns(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        for table_name in (
            "akm_mobility",
            "akm_sorting",
            "ols",
            "correia_synthetic",
            "correia_real",
        ):
            table = document["tables"][table_name]
            self.assertEqual(
                [header.strip("`") for header in table["header"][2:]],
                [
                    "rust-map",
                    "within-off",
                    "within-diagonal",
                    "within",
                    "fixest",
                    "FEM.jl",
                ],
            )
            self.assertTrue(
                all(len(row) == len(table["header"]) for row in table["rows"])
            )

    def test_runtime_collector_fills_lsmr_ablation_columns_in_all_five_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory)
            fixtures = {
                "ols.csv": [
                    *self._rows("simple", "within-off", 10_000_000, 3, "default"),
                    *self._rows("simple", "within-diagonal", 10_000_000, 3, "default"),
                ],
                "akm.csv": [
                    *self._rows("akm_mobility_1", "within-off", 1_000_000, 3, "default"),
                    *self._rows("akm_mobility_1", "within-diagonal", 1_000_000, 3, "default"),
                    *self._rows("akm_sorting_1", "within-off", 1_000_000, 3, "default"),
                    *self._rows("akm_sorting_1", "within-diagonal", 1_000_000, 3, "default"),
                ],
                "correia.csv": [
                    *self._rows("synthetic-complete", "within-off", 1_000, 2, "default"),
                    *self._rows("synthetic-complete", "within-diagonal", 1_000, 2, "default"),
                    *self._rows("credit", "within-off", 1_000, 2, "default"),
                    *self._rows("credit", "within-diagonal", 1_000, 2, "default"),
                ],
            }
            for filename, rows in fixtures.items():
                pd.DataFrame(rows).to_csv(latest / filename, index=False)
            document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
            with patch.object(paper_results, "LATEST_RUN", latest):
                paper_results._synchronize_canonical_tables(document, write=False)

        for table_name in (
            "akm_mobility",
            "akm_sorting",
            "ols",
            "correia_synthetic",
            "correia_real",
        ):
            table = document["tables"][table_name]
            self.assertEqual(table["rows"][0][3:5], ["2.00s", "2.00s"])

    def test_missing_runtime_file_preserves_collected_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
            document["tables"]["ols"]["rows"][0][2] = "2.00s"
            with patch.object(paper_results, "LATEST_RUN", Path(directory)):
                paper_results._synchronize_canonical_tables(document, write=False)
            self.assertEqual(document["tables"]["ols"]["rows"][0][2], "2.00s")

    def test_partial_hardness_file_preserves_other_collected_gaps(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        mobility_gap = document["tables"]["akm_mobility"]["rows"][0][1]
        rows = [
            {
                "dataset_id": "akm_sorting_1",
                "fe_a": "indiv_id",
                "fe_b": "firm_id",
                "one_minus_rho": "0.25",
                "worst_component_obs_share": "1.0",
            }
        ]
        with patch.object(paper_results, "_latest_rows", return_value=rows):
            paper_results._synchronize_hardness(document)
        self.assertEqual(document["tables"]["akm_mobility"]["rows"][0][1], mobility_gap)
        self.assertEqual(document["tables"]["akm_sorting"]["rows"][0][1], "0.25 (1.00)")

    def test_render_does_not_collect_raw_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(paper_results, "GENERATED_DIR", Path(directory) / "tables"),
                patch.object(paper_results, "_synchronize_canonical_tables") as synchronize,
            ):
                paper_results.render(None)
        synchronize.assert_not_called()

    def test_accuracy_frontier_uses_one_median_row_per_setting(self) -> None:
        document = {"tables": {"accuracy_frontier": {"rows": []}}}
        rows = [
            {
                "design": "simple", "backend": "rust-map", "tolerance": "1e-8",
                "runtime_s": runtime, "max_eta": eta, "converged": "true",
            }
            for runtime, eta in (("0.02", "2e-8"), ("0.04", "4e-8"), ("0.06", "6e-8"))
        ]
        with patch.object(paper_results, "_latest_rows", return_value=rows):
            paper_results._synchronize_accuracy_frontier(document)
        rendered = document["tables"]["accuracy_frontier"]["rows"]
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0][3], "0.040s")
        self.assertEqual(rendered[0][4], "$4.0 times 10^(-8)$")

    def test_accuracy_frontier_keeps_a_capped_setting(self) -> None:
        document = {"tables": {"accuracy_frontier": {"rows": []}}}
        rows = [
            {
                "design": "simple",
                "backend": "rust-map",
                "tolerance": "1e-10",
                "runtime_s": "1",
                "max_eta": "",
                "repetition": str(repetition),
                "n_planned": "3",
                "converged": "false",
                "capped": "true",
            }
            for repetition in range(3)
        ]
        with patch.object(paper_results, "_latest_rows", return_value=rows):
            paper_results._synchronize_accuracy_frontier(document)

        rendered = document["tables"]["accuracy_frontier"]["rows"]
        self.assertEqual(rendered[0][3:], ["capped (0/3)", "--"])

    def test_factor_scaling_does_not_average_failed_trials(self) -> None:
        document = {"tables": {"factor_scaling": {"rows": []}}}
        rows = [
            {
                "n_factors": "2",
                "setup_s": "1",
                "solve_s": "2",
                "setup_share": "0.333",
                "iterations_max": "5",
                "converged": "true",
                "capped": "false",
            },
            {
                "n_factors": "2",
                "setup_s": "100",
                "solve_s": "100",
                "setup_share": "0.5",
                "iterations_max": "10000",
                "converged": "false",
                "capped": "true",
            },
        ]
        with patch.object(paper_results, "_latest_rows", return_value=rows):
            paper_results._synchronize_factor_scaling(document)

        rendered = document["tables"]["factor_scaling"]["rows"][0]
        self.assertEqual(rendered[2], "1.000 (1/2)")
        self.assertEqual(rendered[3], "2.000 (1/2)")

    def test_akm_setup_table_separates_two_and_three_factors(self) -> None:
        document = {
            "tables": {
                "akm_mobility": {
                    "header": ["Scenario", "Gap (share)"],
                    "rows": [["`akm_mobility_1`", "0.41 (1.00)"]]
                },
                "akm_setup_cost": {"rows": []},
            }
        }
        rows = []
        for n_factors, setup, solve in ((2, 0.1, 0.2), (3, 0.3, 0.4)):
            rows.extend(
                {
                    "design": "akm_mobility_1",
                    "n_factors": str(n_factors),
                    "setup_s": str(setup),
                    "solve_s": str(solve),
                    "converged": "true",
                    "repetition": str(repetition),
                    "n_planned": "5",
                }
                for repetition in range(5)
            )
        with patch.object(paper_results, "_latest_rows", return_value=rows):
            paper_results._synchronize_akm_setup_cost(document)
        self.assertEqual(
            document["tables"]["akm_setup_cost"]["rows"][0],
            [
                "`akm_mobility_1`",
                "0.41 (1.00)",
                "0.100s",
                "0.200s",
                "0.300s",
                "0.400s",
            ],
        )

    def test_reuse_table_reports_speedup_against_diagonal(self) -> None:
        document = {"tables": {"regression_reuse": {"rows": []}}}
        rows = []
        for design in ("simple", "difficult"):
            for policy, setup, solve in (
                ("diagonal", 1.0, 9.0),
                ("additive_rebuilt", 2.0, 3.0),
                ("additive_cached", 0.5, 2.0),
            ):
                rows.extend(
                    {
                        "design": design,
                        "policy": policy,
                        "setup_s": str(setup),
                        "solve_s": str(solve),
                        "total_s": str(setup + solve),
                        "converged": "true",
                        "repetition": str(repetition),
                        "n_planned": "3",
                    }
                    for repetition in range(3)
                )
        with patch.object(paper_results, "_latest_rows", return_value=rows):
            paper_results._synchronize_regression_reuse(document)
        rendered = document["tables"]["regression_reuse"]["rows"]
        self.assertEqual(rendered[0][-1], "1.0x")
        self.assertEqual(rendered[1][-1], "2.0x")
        self.assertEqual(rendered[2][-1], "4.0x")
        self.assertEqual(rendered[3][-1], "1.0x")
        self.assertEqual(rendered[4][-1], "2.0x")
        self.assertEqual(rendered[5][-1], "4.0x")
        self.assertEqual(rendered[0][0], "simple")
        self.assertEqual(rendered[3][0], "difficult")

    def test_reuse_benchmark_runs_both_ten_regression_designs(self) -> None:
        self.assertEqual(amortization.N_REGRESSIONS, 10)
        for design in ("simple", "difficult"):
            with patch.object(amortization, "N_OBS", 1_000):
                categories, regressions = amortization._regression_right_hand_sides(
                    design
                )
            self.assertEqual(len(regressions), 10)
            self.assertEqual(categories.shape[1], 3)
            self.assertTrue(all(rhs.shape == (1_000, 2) for rhs in regressions))

if __name__ == "__main__":
    unittest.main()
