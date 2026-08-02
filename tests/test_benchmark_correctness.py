"""Numerical and paper-facing checks for the compact benchmark code."""

from __future__ import annotations

import os
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmarks.accuracy import external_normal_residuals, projection_errors
from benchmarks.akm import AKMConfig, SCENARIOS, simulate_akm_panel
from benchmarks.data import drop_singletons, make_base_data, solver_data
from benchmarks.ols.pyfixest import fit_ols
from benchmarks.ols.run import run_experiment
from benchmarks.ppml.pyfixest import fit_ppml
from benchmarks.within import ppml_inner_outer
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
        self.assertEqual(len(SCENARIOS), 11)
        config = AKMConfig(
            n_workers=8, n_firms=4, n_time=3, n_industries=2, n_match_bins=4
        )
        first = simulate_akm_panel(config, seed=7)
        second = simulate_akm_panel(config, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.shape, (24, 5))
        self.assertAlmostEqual(first.iloc[0]["x1"], -1.9924197841744944)

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

    def test_direct_ppml_fit_uses_three_fixed_effects(self) -> None:
        frame = make_base_data(2_000, "simple", 13)
        fit = fit_ppml(frame, "rust-map")
        self.assertTrue(np.isfinite(float(fit.coef().loc["x1"])))
        self.assertFalse(hasattr(fit, "_Y"))

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

    def test_pyfixest_ppml_cache_policies_agree(self) -> None:
        frame = make_base_data(2_000, "simple", 14)
        reused = ppml_inner_outer.measure_policy(frame, False, 1e-8, 1_000)
        rebuilt = ppml_inner_outer.measure_policy(frame, True, 1e-8, 1_000)
        self.assertTrue(reused["outer_converged"])
        self.assertTrue(rebuilt["outer_converged"])
        self.assertEqual(reused["n_retained"], rebuilt["n_retained"])
        self.assertAlmostEqual(reused["beta_x1"], rebuilt["beta_x1"], places=8)
        self.assertAlmostEqual(reused["deviance"], rebuilt["deviance"], places=7)


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

    def test_missing_runtime_file_preserves_collected_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
            document["tables"]["ols"]["rows"][0][2] = "2.00s"
            with patch.object(paper_results, "LATEST_RUN", Path(directory)):
                paper_results._synchronize_canonical_tables(document, write=False)
            self.assertEqual(document["tables"]["ols"]["rows"][0][2], "2.00s")

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

    def test_ppml_inner_outer_table_distinguishes_inner_regimes(self) -> None:
        self.assertEqual(ppml_inner_outer.N_OBS, 100_000)
        self.assertEqual(ppml_inner_outer.REBUILD_OPTIONS, (False, True))
        self.assertEqual(ppml_inner_outer.REPETITIONS, 7)
        document = {
            "tables": {
                "ppml_inner_outer": {
                    "rows": [],
                }
            }
        }
        rows = []
        for rebuild, tolerance, cap in (
            (False, "1e-8", "1000"),
            (True, "1e-12", "10000"),
        ):
            rows.extend(
                {
                    "design": "simple",
                    "engine": "pyfixest",
                    "n_obs": "100000",
                    "rebuild_each_step": str(rebuild).lower(),
                    "inner_tol": tolerance,
                    "inner_maxiter": cap,
                    "outer_converged": "true",
                    "outer_iterations": "8",
                    "runtime_s": str(repetition + 1),
                    "repetition": str(repetition),
                    "n_planned": "7",
                    "error": "",
                }
                for repetition in range(7)
            )
        with patch.object(paper_results, "_latest_rows", return_value=rows):
            paper_results._synchronize_ppml_inner_outer(document)

        rendered = document["tables"]["ppml_inner_outer"]["rows"]
        self.assertEqual(len(rendered[0]), 6)
        self.assertEqual(rendered[0][1], "reuse")
        self.assertEqual(rendered[0][2:4], ["$1.0 times 10^(-8)$", "1#h(0.18em)000"])
        self.assertEqual(rendered[0][5], "4.00s [2.50--5.50s]")
        self.assertEqual(rendered[1][1], "rebuild")
        self.assertEqual(rendered[1][2:4], ["$1.0 times 10^(-12)$", "10#h(0.18em)000"])


if __name__ == "__main__":
    unittest.main()
