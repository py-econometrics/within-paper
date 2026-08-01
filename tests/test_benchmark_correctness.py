from __future__ import annotations

import ast
import csv
import re
import hashlib
import json
import sys
import subprocess
import warnings
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]

from benchmarks.bench_within_setup_cost import _setup_share
from benchmarks.modular.benchmark_correia import summarize_results
from benchmarks.modular.benchmark_fepois import (
    SIZES as FEPOIS_SIZES,
    SPECS as FEPOIS_SPECS,
)
from benchmarks.modular.benchmark_main import SIZES as OLS_SIZES
from benchmarks.dgp.akm import AKMConfig, simulate_akm_panel
from benchmarks.dgp.base import paper_base_dgp
from benchmarks.dgp.scenarios import BaseDGP, _seed_for, get_akm_sweep_scenario_names
from benchmarks.core.accuracy import (
    GATE_A_ETA,
    accuracy_record,
    external_normal_residuals,
    pair_edge_stats,
    projection_errors,
)
from benchmarks.modular.benchmarker_sets import (
    MATCHED_ACCURACY,
    PACKAGE_DEFAULTS,
    build_feols_benchmarkers,
    require_multiple_absorbed_factors,
)
from benchmarks.modular.feols_benchmarkers import (
    _as_bool,
    _external_eta,
    _fit_converged,
)
from benchmarks.solvers.settings import (
    DEFAULT_WITHIN_PRECONDITIONER,
    MECHANISM_LSMR_TOL,
    MECHANISM_MAXITER,
    WITHIN_PRECONDITIONERS,
    demeaner_for,
)
from benchmarks.core.records import RunRecord
from benchmarks.dgp.samples import FE_COLS, SampleSpec, clear_sample_cache, load_sample
from benchmarks.solvers.specs import matched_solver_specs
from benchmarks.core.interfaces import BenchmarkDataset, FeolsSpec
from benchmarks.modular.subprocess_backend import _parse_subprocess_output
from benchmarks.modular.map_diagnostics import map_demean_with_sweeps
from benchmarks.core.results import write_rows
from benchmarks.core.timing import (
    randomized_order,
    repetitions_for_runtime,
    summarize_times,
    timed,
)
from scripts.paper_results import (
    _backend_name,
    _iteration_rows,
    _read_json,
    _component_share,
    _largest_metric,
    _numeric_cell,
    _render_trial_result,
    _table_fragment,
    _validate_ppml_results,
)
from benchmarks.modular.analyze_gap_runtime import _sized_key
from benchmarks.modular.compute_hardness import _component_rho


def _frame_hash(frame: pd.DataFrame) -> str:
    metadata = "|".join(frame.columns) + "\n" + "|".join(map(str, frame.dtypes))
    values = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(metadata.encode() + values).hexdigest()


class BenchmarkCorrectnessTests(unittest.TestCase):
    def test_ols_skips_unreported_one_million_fits(self) -> None:
        self.assertEqual(OLS_SIZES, [10_000_000])
        self.assertEqual(FEPOIS_SIZES, [1_000_000])

    def test_akm_scenarios_match_the_paper_sweeps(self) -> None:
        self.assertEqual(
            get_akm_sweep_scenario_names(),
            (
                "akm_sorting_1",
                "akm_sorting_2",
                "akm_sorting_3",
                "akm_sorting_4",
                "akm_sorting_5",
                "akm_mobility_1",
                "akm_mobility_2",
                "akm_mobility_3",
                "akm_mobility_4",
                "akm_mobility_5",
                "akm_mobility_6",
            ),
        )

    def test_paper_base_dgp_preserves_values_and_six_column_schema(self) -> None:
        expected = {
            "simple": "b95147d29c724cf3079a3cc46079369d4da778bf0bccb5af5948f99888e900bb",
            "difficult": "9ab47efb6b6cc909903555afb8d76e2a976b091f2fe7ba40f00394a2d5c5c64a",
        }
        for dgp_type, digest in expected.items():
            frame = paper_base_dgp(n=2_300, type_=dgp_type, seed=123)
            self.assertEqual(
                list(frame.columns),
                ["indiv_id", "firm_id", "year", "y", "negbin_y", "x1"],
            )
            self.assertEqual(_frame_hash(frame), digest)

    def test_base_cache_rejects_an_extra_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dgp = BaseDGP(Path(tmpdir), "simple")
            with patch("builtins.print"):
                dataset = dgp.generate(n=2_300, n_iters=1, burn_in=0)[0]

                with patch(
                    "benchmarks.dgp.scenarios.paper_base_dgp", side_effect=AssertionError("cache miss")
                ):
                    dgp.generate(n=2_300, n_iters=1, burn_in=0)

                frame = pd.read_parquet(dataset.data_path)
                frame["unused"] = 0
                frame.to_parquet(dataset.data_path, index=False)
                with patch("benchmarks.dgp.scenarios.paper_base_dgp", wraps=paper_base_dgp) as generator:
                    dgp.generate(n=2_300, n_iters=1, burn_in=0)
            generator.assert_called_once()

    def test_akm_paper_path_is_deterministic(self) -> None:
        common = {
            "n_workers": 100,
            "n_firms": 20,
            "n_time": 4,
            "n_industries": 2,
            "n_match_bins": 8,
            "lambda_": 0.8,
        }
        cases = (
            (
                AKMConfig(**common, rho=20.0, delta=0.1),
                "c026e83821a4e48924a8a13e06e4abf1f4f8918f4816daabc47ad04afd505b89",
            ),
            (
                AKMConfig(**common, rho=1.0, delta=0.01),
                "a0ccdf33ab1c7a6c982609bba6ed88eea989aaecae9fd82bfad2cc63f38f561b",
            ),
        )
        for config, digest in cases:
            frame = simulate_akm_panel(config, seed=123)
            self.assertEqual(
                list(frame.columns), ["indiv_id", "firm_id", "year", "x1", "y"]
            )
            self.assertEqual(_frame_hash(frame), digest)

    def test_ppml_uses_only_worker_firm_year_fixed_effects(self) -> None:
        # The order was reconciled with the AKM sweep on 2026-07-26; see
        # test_absorbed_factor_order_is_the_same_everywhere for why it matters.
        self.assertEqual(len(FEPOIS_SPECS), 1)
        self.assertEqual(FEPOIS_SPECS[0].fe_cols, ["indiv_id", "firm_id", "year"])

    def test_canonical_ppml_table_contains_only_three_fe_rows(self) -> None:
        document = json.loads(
            (ROOT / "results" / "paper" / "benchmark_tables.json").read_text()
        )
        rows = document["tables"]["ppml"]["rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual({row[1] for row in rows}, {"3"})

    def test_ppml_sync_rejects_old_two_fe_results(self) -> None:
        rows = [
            {
                "_source_file": "benchmarks/results/fepois_bench__example.csv",
                "n_fe": "2",
            }
        ]
        with self.assertRaisesRegex(ValueError, "only n_fe=3"):
            _validate_ppml_results(rows)

    def test_named_dgp_seed_is_stable(self) -> None:
        self.assertEqual(_seed_for("akm_mobility_1", 1_000_000, 1), 100_000_085)

    def test_setup_share_matches_displayed_decomposition(self) -> None:
        self.assertAlmostEqual(_setup_share(6.4, 1.52), 6.4 / 7.92)

    def test_string_false_is_not_truthy(self) -> None:
        self.assertFalse(_as_bool("false", default=True))
        self.assertTrue(_as_bool("true", default=False))

    def test_correia_summary_retains_trial_counts(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "source_dataset_id": "example",
                    "backend": "fixest",
                    "n_obs": 100,
                    "n_fe": 2,
                    "time": 1.0,
                    "success": True,
                    "error": None,
                },
                {
                    "source_dataset_id": "example",
                    "backend": "fixest",
                    "n_obs": 100,
                    "n_fe": 2,
                    "time": 3.0,
                    "success": True,
                    "error": None,
                },
                {
                    "source_dataset_id": "example",
                    "backend": "fixest",
                    "n_obs": 100,
                    "n_fe": 2,
                    "time": None,
                    "success": False,
                    "error": "did not converge",
                },
            ]
        )
        row = summarize_results(frame).iloc[0]
        self.assertEqual(row["n_runs"], 3)
        self.assertEqual(row["n_success"], 2)
        self.assertFalse(row["success"])
        self.assertEqual(row["time"], 2.0)

    def test_complete_trial_rendering_preserves_nonconvergence(self) -> None:
        partial = [
            {"iter_num": "1", "success": "True", "time": "1.0"},
            {"iter_num": "2", "success": "True", "time": "3.0"},
            {"iter_num": "3", "success": "False", "time": ""},
        ]
        failed = [
            {"iter_num": str(i), "success": "False", "time": ""}
            for i in range(1, 4)
        ]
        self.assertEqual(_render_trial_result(partial), "2.00s (2/3)")
        self.assertEqual(_render_trial_result(failed), "failed (0/3)")
        self.assertEqual(_render_trial_result(partial[:2]), "incomplete")

    def test_pre_environment_modules_stay_standard_library_only(self) -> None:
        # scripts/paper_results.py imports these and must keep running before
        # the Pixi environment exists, so a third-party import in either would
        # break the pre-environment runtime checks.
        for relative in ("benchmarks/core/timing.py", "benchmarks/core/paths.py"):
            with self.subTest(module=relative):
                tree = ast.parse((ROOT / relative).read_text())
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(a.name.split(".")[0] for a in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        imported.add(node.module.split(".")[0])
                self.assertLessEqual(imported, set(sys.stdlib_module_names))

    def test_shared_root_is_the_repository_root(self) -> None:
        """paths.ROOT must agree with a root derived independently of it.

        Every module now takes its paths from core.paths rather than counting
        parent directories itself, so nothing else would notice if that one
        definition drifted. This test derives the root the old way, from the
        test file's own location, and is the only place that still does.
        """
        from benchmarks.core import paths

        self.assertEqual(paths.ROOT, ROOT)
        self.assertTrue((paths.ROOT / "pixi.toml").is_file())
        self.assertEqual(paths.DATA_DIR, ROOT / "benchmarks" / "data")
        self.assertEqual(paths.LATEST_RUN, ROOT / "results" / "runs" / "latest")

    def test_join_key_is_shared_between_harness_and_analysis(self) -> None:
        # Pairing a 1M runtime with a 10M gap moved a fitted slope once; the
        # two sides now build the key from one definition.
        self.assertEqual(
            _sized_key("difficult", "1000000"),
            SampleSpec(design="difficult", n_obs=1_000_000).key,
        )
        self.assertEqual(_sized_key("difficult", "not-a-size"), "difficult")

    def test_iteration_units_are_never_mixed(self) -> None:
        # A MAP sweep is a full pass over the factors; an LSMR iteration is one
        # operator application. Recording the unit per row is what stops a
        # later renderer from pooling them into one column.
        document = _read_json(ROOT / "results" / "paper" / "benchmark_tables.json")
        header = [cell.replace("`", "") for cell in document["tables"]["iterations"]["header"]]
        self.assertEqual(header[1], "map (sweeps)")

        with (ROOT / "results" / "runs" / "latest" / "within_preconditioners.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        units = {row["solver_label"]: row.get("iteration_unit") for row in rows}
        self.assertEqual(units.get("map"), "map-sweep")
        for label in ("within-off", "within-diagonal", "within-additive"):
            self.assertEqual(units.get(label), "lsmr-iteration", label)

    def test_iteration_table_marks_capped_and_absent_cells(self) -> None:
        rows = [
            {"design": "d", "solver_label": "map", "iterations_max": "9000",
             "censoring": "capped"},
            {"design": "d", "solver_label": "map", "iterations_max": "10000",
             "censoring": "capped"},
            {"design": "d", "solver_label": "within-off", "iterations_max": "40",
             "censoring": "none"},
            {"design": "d", "solver_label": "within-diagonal", "iterations_max": "30",
             "censoring": "none"},
        ]
        # map: median 9500 and capped; additive: no rows at all, so absent.
        self.assertEqual(
            _iteration_rows(rows),
            [["d", "9500 (capped)", "40", "30", "#miss"]],
        )

    def test_complete_bipartite_graph_has_unit_gap(self) -> None:
        # A complete bipartite graph has rank one, so its second singular value is
        # zero and the gap is one. This catches spurious nonzero values from PROPACK.
        complete = sp.csr_matrix(np.ones((1000, 500)))
        self.assertAlmostEqual(_component_rho(complete), 0.0, places=6)

    def test_very_large_block_uses_arpack_only(self) -> None:
        # Above PROPACK_MAX_MIN_DIM, use only ARPACK; SciPy's PROPACK
        # implementation can terminate on very large irregular blocks.
        matrix = sp.eye(20_001, format="csr")
        calls: list[str] = []

        def fake_svds(*_args, **kwargs):
            calls.append(kwargs["solver"])
            return np.array([0.5, 1.0])

        with patch("benchmarks.modular.compute_hardness.svds", side_effect=fake_svds):
            self.assertEqual(_component_rho(matrix), 0.25)
        self.assertEqual(calls, ["arpack"])

    def test_sparse_block_uses_propack_first(self) -> None:
        # This block is too large for dense SVD but small enough for PROPACK.
        # A successful PROPACK call should not fall back to ARPACK.
        matrix = sp.eye(5_000, format="csr")
        calls: list[str] = []

        def fake_svds(*_args, **kwargs):
            calls.append(kwargs["solver"])
            return np.array([0.5, 1.0])

        with patch("benchmarks.modular.compute_hardness.svds", side_effect=fake_svds):
            self.assertEqual(_component_rho(matrix), 0.25)
        self.assertEqual(calls, ["propack"])

    def test_sparse_block_falls_back_to_arpack(self) -> None:
        matrix = sp.eye(5_000, format="csr")
        calls: list[str] = []

        def fake_svds(*_args, **kwargs):
            calls.append(kwargs["solver"])
            if kwargs["solver"] == "propack":
                raise RuntimeError("PROPACK did not converge")
            return np.array([0.5, 1.0])

        with patch("benchmarks.modular.compute_hardness.svds", side_effect=fake_svds):
            self.assertEqual(_component_rho(matrix), 0.25)
        self.assertEqual(calls, ["propack", "arpack"])

    def test_prose_values_select_the_named_backend_and_largest_metric(self) -> None:
        rows = [
            ["#agreement-simple", "`fixest`", "", "$1.1 times 10^(-14)$"],
            ["#agreement-difficult", "`fixest`", "", "$1.9 times 10^(-7)$"],
        ]
        self.assertEqual(_numeric_cell(rows[0][3]), 1.1e-14)
        self.assertEqual(
            _largest_metric(rows, 3, backend="fixest"), "$1.9 times 10^(-7)$"
        )

    def test_component_share_ignores_scientific_exponent(self) -> None:
        self.assertEqual(
            _component_share("$5.12 times 10^(-4)$ (0.30)"), 0.30
        )

    def test_agreement_rowspan_is_rendered_as_typst_code(self) -> None:
        table = {
            "columns": "(1fr, 1fr)",
            "align": "(left, left)",
            "header": ["Design", "Backend"],
            "rows": [
                ["#agreement-simple", "`rust-map`"],
                ["", "`within`"],
            ],
        }
        fragment = _table_fragment("agreement", table)
        self.assertIn(
            "table.cell(rowspan: 4)[simple], [PyFixest #linebreak() MAP #linebreak() none]",
            fragment,
        )
        self.assertNotIn("[table.cell(rowspan: 4)[simple]]", fragment)
        self.assertIn(
            "\n  [PyFixest #linebreak() LSMR #linebreak() factor-pair],",
            fragment,
        )
        self.assertNotIn(
            "\n  [], [PyFixest #linebreak() LSMR #linebreak() factor-pair],",
            fragment,
        )

    def test_demeaner_backend_labels_map_to_preconditioners(self) -> None:
        rust = demeaner_for("rust")
        self.assertEqual(rust.kind, "map")
        self.assertEqual(rust.backend, "rust")

        alias = demeaner_for("within")
        self.assertEqual(alias.kind, "lsmr")
        self.assertEqual(alias.preconditioner, DEFAULT_WITHIN_PRECONDITIONER)
        self.assertEqual(alias.fixef_atol, alias.fixef_btol)

        for name in WITHIN_PRECONDITIONERS:
            demeaner = demeaner_for(f"within-{name}")
            self.assertEqual(demeaner.preconditioner, name)
            self.assertEqual(demeaner.fixef_atol, 1e-8)
            self.assertEqual(demeaner.fixef_btol, 1e-8)

        with self.assertRaisesRegex(ValueError, "Unknown within preconditioner"):
            demeaner_for("within-bogus")

    def test_both_views_are_measured_in_one_pass(self) -> None:
        """One sweep produces the cross-package rows and the mechanism rows.

        The two views differ only in stopping rule, so they are separate labels
        on one run rather than separate drivers over the same designs. Distinct
        labels are what lets the curation step tell them apart.
        """
        self.assertEqual(
            [spec.label for spec in MATCHED_ACCURACY],
            [
                "pyfixest (rust-map, matched)",
                "pyfixest (within-off)",
                "pyfixest (within-diagonal)",
                "pyfixest (within-additive)",
            ],
        )
        default_labels = {spec.label for spec in PACKAGE_DEFAULTS}
        matched_labels = {spec.label for spec in MATCHED_ACCURACY}
        self.assertEqual(default_labels & matched_labels, set())

        both = build_feols_benchmarkers(matched_accuracy=True, external=False)
        self.assertEqual(
            [b.name for b in both.benchmarkers],
            [spec.label for spec in (*PACKAGE_DEFAULTS, *MATCHED_ACCURACY)],
        )
        defaults_only = build_feols_benchmarkers(external=False)
        self.assertEqual(len(defaults_only.benchmarkers), len(PACKAGE_DEFAULTS))

    def test_preconditioner_comparison_needs_two_factors(self) -> None:
        require_multiple_absorbed_factors(FeolsSpec("y", ["x1"], ["a", "b"], "iid"))
        with self.assertRaisesRegex(ValueError, "at least two absorbed factors"):
            require_multiple_absorbed_factors(FeolsSpec("y", ["x1"], ["a"], "iid"))

    def test_matched_arms_share_one_iteration_budget(self) -> None:
        """MAP must not be handed ten times the budget it is compared against.

        The package defaults give MAP 10,000 iterations and LSMR 1,000. At 1M
        that asymmetry censored within-off in 30 of 33 trials, every one at its
        own lower cap, which would make "unpreconditioned LSMR does not remove
        the slow directions" a statement about the budget rather than about the
        preconditioner.
        """
        budgets = set()
        matched = build_feols_benchmarkers(
            package_defaults=False, matched_accuracy=True, external=False
        )
        for benchmarker in matched.benchmarkers:
            demeaner = demeaner_for(
                benchmarker._demeaner_backend,
                tol=benchmarker._tol,
                maxiter=benchmarker._maxiter,
            )
            budgets.add(demeaner.fixef_maxiter)
        self.assertEqual(budgets, {MECHANISM_MAXITER})

        # The cross-package view keeps each package's documented default.
        default_lsmr = demeaner_for("within")
        self.assertEqual(default_lsmr.fixef_maxiter, 1_000)

    def test_external_residual_is_small_for_exact_demeaning(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        categories = np.column_stack(
            [
                rng.integers(0, 25, n),
                rng.integers(0, 18, n),
                rng.integers(0, 4, n),
            ]
        )
        # Construct rhs = D alpha + noise orthogonalized by a tight solve.
        from within import LsmrOptions, PreconditionerConfig, solve_batch

        rhs = rng.standard_normal((n, 2))
        result = solve_batch(
            np.asfortranarray(categories.astype(np.uint32)),
            np.asfortranarray(rhs),
            LsmrOptions(tol=1e-12, maxiter=2_000),
            preconditioner=PreconditionerConfig.Additive,
        )
        eta = external_normal_residuals(categories, rhs, result.demeaned)
        self.assertTrue(np.all(eta <= GATE_A_ETA))
        delta = projection_errors(result.demeaned, result.demeaned, rhs)
        self.assertTrue(np.all(delta == 0.0))
        record = accuracy_record(
            categories=categories,
            rhs=rhs,
            demeaned=result.demeaned,
            reference_demeaned=result.demeaned,
            beta=np.array([1.0]),
            beta_star=np.array([1.0]),
            se_star=np.array([0.5]),
        )
        self.assertTrue(record.gate_a_components_measured)
        self.assertTrue(record.clears_gate_a)

    def test_map_censoring_marks_capped_runs(self) -> None:
        # Path-like worker-firm incidence: MAP needs many sweeps at tight tol.
        n_nodes = 80
        worker = np.arange(n_nodes)
        firm = np.arange(n_nodes)
        # Two observations per edge of a path, plus the reverse orientation.
        categories = np.column_stack(
            [
                np.concatenate([worker[:-1], firm[1:]]),
                np.concatenate([firm[1:], worker[:-1]]),
            ]
        )
        rhs = np.linspace(-1.0, 1.0, categories.shape[0]).reshape(-1, 1)
        capped = map_demean_with_sweeps(rhs, categories, tol=1e-12, maxiter=3)
        self.assertTrue(capped.any_capped)
        self.assertEqual(capped.iterations, [3])
        self.assertEqual(capped.censoring, ["capped"])
        self.assertFalse(capped.converged[0])

        easy_categories = np.column_stack(
            [np.repeat(np.arange(20), 10), np.tile(np.arange(10), 20)]
        )
        easy = map_demean_with_sweeps(
            np.linspace(-1.0, 1.0, easy_categories.shape[0]).reshape(-1, 1),
            easy_categories,
            tol=1e-8,
            maxiter=100,
        )
        self.assertEqual(easy.n_converged, 1)
        self.assertFalse(easy.any_capped)
        self.assertEqual(easy.censoring, ["none"])

    def test_pair_edge_stats_count_unique_edges(self) -> None:
        categories = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [0, 0],  # duplicate edge
            ]
        )
        stats = pair_edge_stats(categories)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["n_edges"], 3)
        self.assertEqual(stats[0]["max_edges"], 4)

    def test_repetition_counts_follow_the_runtime_bands(self) -> None:
        # PROTOCOL.md R1/R2/R3. The flatness claim needs many trials on the
        # subsecond cells, where the differences it rests on are milliseconds.
        self.assertEqual(repetitions_for_runtime(0.3), 20)
        self.assertEqual(repetitions_for_runtime(0.999), 20)
        self.assertEqual(repetitions_for_runtime(1.0), 7)
        self.assertEqual(repetitions_for_runtime(9.999), 7)
        self.assertEqual(repetitions_for_runtime(10.0), 3)
        self.assertEqual(repetitions_for_runtime(350.0), 3)
        with self.assertRaises(ValueError):
            repetitions_for_runtime(-1.0)

    def test_failed_trials_stay_in_the_denominator(self) -> None:
        # A failure must not be dropped before the median: reporting the median
        # of the survivors without the count is a selected estimator.
        summary = summarize_times([1.0, 2.0, None, 4.0, 3.0])
        self.assertEqual(summary.n_attempted, 5)
        self.assertEqual(summary.n_converged, 4)
        self.assertFalse(summary.is_complete)
        self.assertAlmostEqual(summary.median_s, 2.5)
        self.assertAlmostEqual(summary.iqr_s, 1.5)

        all_failed = summarize_times([None, None, None])
        self.assertIsNone(all_failed.median_s)
        self.assertEqual(all_failed.n_attempted, 3)
        self.assertEqual(all_failed.n_converged, 0)

    def test_backend_order_is_shuffled_but_reproducible(self) -> None:
        backends = ["a", "b", "c", "d", "e", "f"]
        first = randomized_order(backends, 20260726)
        self.assertEqual(first, randomized_order(backends, 20260726))
        self.assertCountEqual(first, backends)
        self.assertNotEqual(
            first, randomized_order(backends, 20260727), "seeds must differ"
        )

    def test_absorbed_factor_order_is_the_same_everywhere(self) -> None:
        # MAP cycles through factors in the given order, so a table that
        # absorbs in a different order is not comparing the same specification.
        from benchmarks.modular.benchmark_akm_sweep import SPECS as AKM_SPECS
        from benchmarks.modular.benchmark_fepois import SPECS as PPML_SPECS
        from benchmarks.modular.benchmark_main import SPECS as MAIN_SPECS

        expected = ["indiv_id", "firm_id", "year"]
        for label, specs in (
            ("main", MAIN_SPECS),
            ("fepois", PPML_SPECS),
            ("akm", AKM_SPECS),
        ):
            for spec in specs:
                self.assertEqual(spec.fe_cols, expected, label)

    def test_matched_arms_do_not_collapse_onto_default_cells(self) -> None:
        """Both views share a sweep, so the renderer must keep them apart.

        _backend_name folds any label containing "within" onto "within" and any
        containing "rust-map" onto "rust-map". With both views in one run that
        put four within variants and two MAP variants into the same cell, whose
        duplicate trial ids then render as "incomplete".
        """
        self.assertEqual(_backend_name("pyfixest (within)"), "within")
        self.assertEqual(_backend_name("pyfixest (rust-map)"), "rust-map")
        for preconditioner in ("off", "diagonal", "additive"):
            self.assertEqual(
                _backend_name(f"pyfixest (within-{preconditioner})"),
                f"within-{preconditioner}",
            )
        self.assertEqual(
            _backend_name("pyfixest (rust-map, matched)"), "rust-map-matched"
        )

        names = {_backend_name(spec.label) for spec in PACKAGE_DEFAULTS}
        matched = {_backend_name(spec.label) for spec in MATCHED_ACCURACY}
        self.assertEqual(names & matched, set())

    def test_unmeasured_gate_a_components_do_not_pass(self) -> None:
        """An absent metric is not a passing metric."""
        rng = np.random.default_rng(0)
        n = 300
        categories = np.column_stack(
            [rng.integers(0, 20, n), rng.integers(0, 12, n), rng.integers(0, 3, n)]
        )
        from within import LsmrOptions, PreconditionerConfig, solve_batch

        rhs = rng.standard_normal((n, 2))
        result = solve_batch(
            np.asfortranarray(categories.astype(np.uint32)),
            np.asfortranarray(rhs),
            LsmrOptions(tol=1e-12, maxiter=2_000),
            preconditioner=PreconditionerConfig.Additive,
        )
        # eta only: delta and slope were never computed.
        eta_only = accuracy_record(
            categories=categories, rhs=rhs, demeaned=result.demeaned
        )
        self.assertTrue(eta_only.gate_a_eta)
        self.assertFalse(eta_only.clears_gate_a)
        self.assertFalse(eta_only.gate_a_components_measured)

    def test_sized_design_key_separates_sample_sizes(self) -> None:
        """simple/difficult run at three sizes with very different gaps.

        The difficult worker-firm gap is 1.7e-3 at 100K, 1.7e-5 at 1M and
        1.7e-7 at 10M, so a join on the family name alone pairs a 1M runtime
        with a 10M gap.
        """
        from benchmarks.modular.analyze_gap_runtime import _design_key_from_hardness, _sized_key

        self.assertEqual(_design_key_from_hardness("difficult_10000000_k1_iter_1"), "difficult")
        self.assertNotEqual(
            _sized_key("difficult", 10_000_000), _sized_key("difficult", 1_000_000)
        )
        self.assertEqual(_sized_key("difficult", 1_000_000), "difficult@1000000")
        # An unparseable size degrades to the bare family rather than raising.
        self.assertEqual(_sized_key("enron", None), "enron")

    def test_runtime_rising_with_gap_is_the_counterexample(self) -> None:
        """Sorted by increasing gap, better connectivity should mean less work.

        A falling series is the expected pattern; flagging it inverted the
        counterexample list.
        """
        import pandas as pd

        from benchmarks.modular.analyze_gap_runtime import _counter_examples

        def frame(times):
            return pd.DataFrame(
                {
                    "family": ["akm_sorting_1", "akm_sorting_2", "akm_sorting_3"],
                    "design": ["akm_sorting_1@1", "akm_sorting_2@1", "akm_sorting_3@1"],
                    "backend": ["b"] * 3,
                    "gap": [1e-4, 1e-3, 1e-2],
                    "median_time": times,
                    "component_share": [1.0] * 3,
                }
            )

        expected = [name["name"] for name in _counter_examples(frame([3.0, 2.0, 1.0]))]
        self.assertNotIn("sorting_non_monotonic", expected)
        flagged = [name["name"] for name in _counter_examples(frame([1.0, 2.0, 3.0]))]
        self.assertIn("sorting_non_monotonic", flagged)

    def test_sample_seed_ignores_the_repetition_index(self) -> None:
        """Repeated timings must run on one fixed sample.

        Two drivers previously derived the seed from the repetition counter, so
        each "repetition" measured a different draw and solver variance was
        confounded with DGP variance (PROTOCOL.md section 2).
        """
        spec = SampleSpec(design="difficult", n_obs=5_000)
        self.assertEqual(spec.seed, SampleSpec(design="difficult", n_obs=5_000).seed)
        self.assertNotEqual(spec.seed, SampleSpec(design="simple", n_obs=5_000).seed)
        self.assertNotEqual(spec.seed, SampleSpec(design="difficult", n_obs=6_000).seed)

    def test_repeated_loads_return_the_identical_sample(self) -> None:
        clear_sample_cache()
        spec = SampleSpec(design="simple", n_obs=3_000)
        first = load_sample(spec)
        second = load_sample(spec)
        self.assertEqual(first.sample_hash, second.sample_hash)
        # Identity, not just equality: the cache hands back the same arrays.
        self.assertIs(first.categories, second.categories)
        self.assertEqual(len(FE_COLS), first.categories.shape[1])
        clear_sample_cache()

    def test_sized_sample_key_distinguishes_scales(self) -> None:
        self.assertEqual(SampleSpec("difficult", 1_000_000).key, "difficult@1000000")
        self.assertNotEqual(
            SampleSpec("difficult", 1_000_000).key,
            SampleSpec("difficult", 10_000_000).key,
        )

    def test_standalone_settings_match_the_end_to_end_ablation(self) -> None:
        """The two halves of the mechanism evidence must share a stopping rule.

        The standalone diagnostics ran at the package defaults while the
        ablation was frozen at the matched settings, so their iteration counts
        described different problems.
        """
        specs = matched_solver_specs()
        self.assertEqual(
            [spec.preconditioner for spec in specs], list(WITHIN_PRECONDITIONERS)
        )
        for spec in specs:
            self.assertEqual(spec.maxiter, MECHANISM_MAXITER)
            self.assertEqual(spec.tol, MECHANISM_LSMR_TOL)

    def test_run_record_rejects_incomplete_measurements(self) -> None:
        """A record missing a protocol-required field must not write silently."""

        def record(**overrides):
            base = dict(
                design="difficult",
                n_obs=1_000,
                sample_hash="abc",
                config_id="additive/tol=1e-12/maxiter=10000",
                solver_label="within-additive",
                view="matched-accuracy",
                repetition=0,
                setup_s=0.1,
                solve_s=0.2,
                total_s=0.3,
                converged=True,
                censoring="none",
                max_eta=1e-12,
            )
            base.update(overrides)
            return RunRecord(**base)

        self.assertEqual(record().validate(), [])
        self.assertIn("max_eta missing: no external accuracy was recorded",
                      record(max_eta=None).validate())
        self.assertIn("converged missing", record(converged=None).validate())
        self.assertTrue(
            any("capped or failed" in problem
                for problem in record(converged=False).validate())
        )
        self.assertTrue(
            any("sample_hash" in problem for problem in record(sample_hash="").validate())
        )
        self.assertTrue(
            any("Gate A components" in problem
                for problem in record(clears_gate_a=True).validate())
        )
        # setup + solve may not exceed the reported total.
        self.assertTrue(
            any("exceeds total" in problem
                for problem in record(setup_s=1.0, solve_s=1.0, total_s=0.5).validate())
        )

    def test_headline_fits_carry_an_accuracy_record(self) -> None:
        """Every timed fit must be able to report the accuracy it achieved.

        A runtime on its own cannot answer whether a speedup is a tolerance
        artifact, because each package stops on its own quantity. The recomputed
        eta has to track the requested tolerance, and it has to be measured on
        the rows the model actually kept.
        """
        import warnings

        import pyfixest as pf

        frame = pd.read_parquet(ROOT / "benchmarks" / "data" / "difficult_100k.parquet")
        formula = "y ~ x1 | indiv_id + firm_id + year"

        def eta_at(backend: str, tol: float | None) -> float:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = pf.feols(
                    formula,
                    data=frame,
                    vcov="iid",
                    demeaner=demeaner_for(backend, tol=tol),
                )
            value = _external_eta(fit, frame, "y", ["x1"])
            self.assertIsNotNone(value)
            return value

        tight = eta_at("within-additive", 1e-12)
        loose = eta_at("within-additive", None)
        self.assertLess(tight, GATE_A_ETA)
        # The package default does not reach the gate; that is the finding the
        # calibration pilot froze, and it must remain visible.
        self.assertGreater(loose, GATE_A_ETA)
        self.assertLess(tight, loose)

    def test_accuracy_record_is_absent_rather_than_wrong(self) -> None:
        """A model this cannot measure yields None, never a plausible number."""

        class NotAModel:
            pass

        self.assertIsNone(_external_eta(NotAModel(), pd.DataFrame({"y": [1.0]}), "y", []))


class TypedDemeanerTests(unittest.TestCase):
    """Every fit goes through a typed demeaner, not the deprecated strings."""

    def test_no_source_file_passes_the_deprecated_arguments(self) -> None:
        # PyFixest 0.60 deprecated demeaner_backend/fixef_tol/fixef_maxiter on
        # feols and fepois. They remain valid as typed-constructor keywords, so
        # only the call-site form is banned.
        offenders = []
        for path in sorted((ROOT / "benchmarks").rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                # Prose that names the deprecated argument is fine; a call
                # that passes it is not.
                if "`demeaner_backend=`" in line or line.lstrip().startswith("#"):
                    continue
                if "demeaner_backend=" in line:
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(offenders, [], "pass a typed demeaner= instead")

    def test_rust_cg_resolves_to_the_within_lsmr_backend(self) -> None:
        # Pre-0.60 alias. It was not conjugate gradient by then: PyFixest
        # mapped it onto the within backend with preconditioner "auto", and the
        # agreement check depends on that being what it compares against.
        demeaner = demeaner_for("rust-cg")
        self.assertEqual(demeaner.backend, "within")
        self.assertEqual(demeaner.preconditioner, "auto")
        self.assertEqual(demeaner.precision, "float64")

    def test_named_configurations_build_the_expected_demeaner(self) -> None:
        self.assertEqual(demeaner_for("rust").backend, "rust")
        self.assertEqual(demeaner_for("within").preconditioner, "additive")
        for name in ("off", "diagonal", "additive"):
            self.assertEqual(demeaner_for(f"within-{name}").preconditioner, name)
        self.assertEqual(demeaner_for("torch_mps").precision, "float32")


class SubprocessOutputTests(unittest.TestCase):
    """Parsing the records a driver emits, including the unhappy paths."""

    @staticmethod
    def _datasets():
        return [
            BenchmarkDataset("d1", Path("/tmp/x.parquet"), "simple", 1, 100, "trial", 1),
            BenchmarkDataset("d2", Path("/tmp/x.parquet"), "simple", 1, 100, "trial", 2),
        ]

    def _parse(self, stdout: str, returncode: int = 0):
        spec = FeolsSpec(depvar="y", covariates=["x1"], fe_cols=["a", "b"], vcov="iid")
        completed = subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=""
        )
        return _parse_subprocess_output(
            datasets=self._datasets(), spec=spec, backend="r.fixest",
            completed_process=completed,
        )

    def test_a_trial_the_driver_never_reported_is_a_failure(self) -> None:
        # Not a reused timing from the trial that did run.
        line = json.dumps(
            {"dataset_id": "d1", "iter_num": 1, "time": 1.5, "success": "true"}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = self._parse(line + "\n")
        self.assertTrue(results[0].success)
        self.assertAlmostEqual(results[0].time, 1.5)
        self.assertFalse(results[1].success)
        self.assertIsNone(results[1].time)
        self.assertTrue(any("1/2" in str(w.message) for w in caught))

    def test_string_success_flags_are_coerced(self) -> None:
        rows = "\n".join(
            json.dumps({"dataset_id": d, "iter_num": i, "time": 1.0, "success": flag})
            for d, i, flag in (("d1", 1, "true"), ("d2", 2, "false"))
        )
        results = self._parse(rows + "\n")
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)


class SharedPrimitiveTests(unittest.TestCase):
    """The timing and result-IO primitives every driver now goes through."""

    def test_timed_block_reports_its_own_duration(self) -> None:
        with timed(collect=False) as elapsed:
            self.assertIsNone(elapsed.seconds)
            time.sleep(0.01)
        self.assertIsNotNone(elapsed.seconds)
        self.assertGreaterEqual(elapsed.seconds, 0.01)

    def test_timed_block_records_time_even_when_the_fit_raises(self) -> None:
        # A backend that throws still consumed wall time, and a driver that
        # wants to record the failure needs the duration rather than nothing.
        with self.assertRaises(RuntimeError):
            with timed(collect=False) as elapsed:
                time.sleep(0.01)
                raise RuntimeError("backend failed")
        self.assertIsNotNone(elapsed.seconds)
        self.assertGreaterEqual(elapsed.seconds, 0.01)

    def test_writer_unions_keys_so_optional_fields_stay_aligned(self) -> None:
        # An optional diagnostic recorded for only some rows must widen the
        # table, not truncate it to the first row's columns.
        rows = [{"a": 1}, {"a": 2, "b": 3}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "rows.csv"
            write_rows(out, rows)
            written = list(csv.DictReader(out.open()))
        self.assertEqual([row["a"] for row in written], ["1", "2"])
        self.assertEqual([row["b"] for row in written], ["", "3"])

    def test_writer_honours_a_pinned_column_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.csv"
            write_rows(out, [{"b": 1, "a": 2}], fieldnames=["a", "b"])
            self.assertEqual(out.read_text().splitlines()[0], "a,b")

    def test_writer_refuses_an_empty_result_set(self) -> None:
        # A header-only file reads downstream as a completed run that measured
        # nothing, which is exactly the failure it should surface instead.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_rows(Path(tmp) / "rows.csv", [])


class DriverEntryPointTests(unittest.TestCase):
    """Importing a driver must never run it.

    bench_memory_py once had no __main__ guard, so importing the module ran
    all eight benchmarks and overwrote results/runs/latest/memory.csv with
    whatever the import happened to produce. Anything that imports broadly
    (a test collector, a dead-code sweep, an IDE) could destroy a recorded
    result that way, and results/runs is not tracked, so there is no undo.
    """

    DRIVERS = sorted(
        path
        for path in (ROOT / "benchmarks").rglob("*.py")
        if path.name != "__init__.py"
    ) + sorted((ROOT / "scripts").glob("*.py"))

    # matplotlib.use() and the rcParams assignment have to run before pyplot is
    # imported, so they are legitimately module level. Nothing else here is.
    IMPORT_TIME_SETUP = ("matplotlib.use", "matplotlib.rcParams")

    def test_no_driver_executes_work_at_module_level(self) -> None:
        offenders = []
        for path in self.DRIVERS:
            source = path.read_text(encoding="utf-8")
            for node in ast.parse(source).body:
                # Loops and bare calls at module level run on import.
                # Assignments, imports, defs and classes are declarations.
                if not isinstance(node, (ast.Expr, ast.For, ast.While, ast.With)):
                    continue
                # A module docstring is an Expr but does nothing.
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    continue
                text = ast.get_source_segment(source, node) or ""
                if text.startswith(self.IMPORT_TIME_SETUP):
                    continue
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], f"module-level work on import: {offenders}")

    def test_every_driver_guards_its_entry_point(self) -> None:
        """A driver with argparse must reach it only through __main__."""
        offenders = []
        for path in self.DRIVERS:
            source = path.read_text(encoding="utf-8")
            if "argparse.ArgumentParser(" not in source:
                continue
            if 'if __name__ == "__main__":' not in source:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [], f"argparse without a guard: {offenders}")


class ConvergenceCheckTests(unittest.TestCase):
    """Convergence must be read through the helper, not off a raw attribute.

    PyFixest defines `.convergence` on Feglm (PPML) but not on Feols, so
    `fit.convergence` on a linear fit raises AttributeError rather than
    reporting non-convergence. Two standalone drivers did exactly that, which
    made every one of their runs fail at the check.
    """

    def test_no_driver_reads_convergence_off_a_fit_directly(self) -> None:
        offenders = []
        for path in DriverEntryPointTests.DRIVERS:
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if re.search(r"\bfit\w*\.convergence\b", line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(offenders, [], f"use _fit_converged instead: {offenders}")

    def test_helper_tolerates_a_model_with_no_flag(self) -> None:
        class LinearFit:  # Feols exposes no convergence attribute at all
            pass

        class PoissonFit:  # Feglm does
            convergence = False

        self.assertTrue(_fit_converged(LinearFit()))
        self.assertFalse(_fit_converged(PoissonFit()))


if __name__ == "__main__":
    unittest.main()
