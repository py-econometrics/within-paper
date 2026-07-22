from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
MODULAR = ROOT / "benchmarks" / "modular"
BENCHMARKS = ROOT / "benchmarks"
SCRIPTS = ROOT / "scripts"
for path in (MODULAR, BENCHMARKS, SCRIPTS):
    sys.path.insert(0, str(path))

from bench_within_setup_cost import _setup_share  # noqa: E402
from benchmark_correia import summarize_results  # noqa: E402
from benchmark_fepois import (  # noqa: E402
    SIZES as FEPOIS_SIZES,
    SPECS as FEPOIS_SPECS,
)
from benchmark_main import SIZES as OLS_SIZES  # noqa: E402
from akm_dgp import AKMConfig, simulate_akm_panel  # noqa: E402
from dgp_functions import paper_base_dgp  # noqa: E402
from dgps import BaseDGP, _seed_for, get_akm_sweep_scenario_names  # noqa: E402
from feols_benchmarkers import _as_bool  # noqa: E402
from paper_results import (  # noqa: E402
    _component_share,
    _largest_backend_metric,
    _numeric_cell,
    _render_trial_result,
    _synchronize_external_results,
    _table_fragment,
    _validate_external_results,
    _validate_ppml_results,
)
from benchmarks.modular.compute_hardness import _component_rho  # noqa: E402


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
                    "dgps.paper_base_dgp", side_effect=AssertionError("cache miss")
                ):
                    dgp.generate(n=2_300, n_iters=1, burn_in=0)

                frame = pd.read_parquet(dataset.data_path)
                frame["unused"] = 0
                frame.to_parquet(dataset.data_path, index=False)
                with patch("dgps.paper_base_dgp", wraps=paper_base_dgp) as generator:
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
        self.assertEqual(len(FEPOIS_SPECS), 1)
        self.assertEqual(FEPOIS_SPECS[0].fe_cols, ["indiv_id", "year", "firm_id"])

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

    def test_external_cuda_measurements_are_explicit(self) -> None:
        document = {
            "tables": {
                "ols": {
                    "header": ["Design", "Gap", "`torch-cuda`"],
                    "rows": [["simple (dense graph)", "", "old"], ["difficult (sparse graph)", "", "old"]],
                }
            }
        }
        changed = _synchronize_external_results(document)
        self.assertEqual(changed, 2)
        self.assertEqual(document["tables"]["ols"]["rows"][0][2], "4.73s")
        self.assertEqual(document["tables"]["ols"]["rows"][1][2], "8.73s")

    def test_external_cuda_policy_disallows_cross_machine_comparisons(self) -> None:
        external = json.loads(
            (ROOT / "results" / "external" / "cuda.json").read_text()
        )
        measurements = _validate_external_results(external)
        self.assertEqual(external["source"], "legacy PyFixest benchmark suite")
        self.assertEqual(external["status"], "indicative_only")
        self.assertFalse(external["exact_run_provenance_available"])
        self.assertFalse(external["cross_machine_comparison_allowed"])
        self.assertEqual([row["time_s"] for row in measurements], [4.73, 8.73])

        invalid = {**external, "cross_machine_comparison_allowed": True}
        with self.assertRaisesRegex(ValueError, "cross_machine_comparison_allowed"):
            _validate_external_results(invalid)

    def test_generated_values_contain_no_cuda_cpu_ratio(self) -> None:
        values = (ROOT / "generated" / "paper_values.typ").read_text()
        manuscript = (ROOT / "graph_preconditioner_hdfe.typ").read_text()
        self.assertNotIn("result_ols_gpu_vs_fem", values)
        self.assertNotIn("result_ols_gpu_vs_fem", manuscript)

    def test_complete_bipartite_graph_has_unit_gap(self) -> None:
        # A complete bipartite graph is rank one, so its second singular value is
        # exactly zero and the reported gap 1 - rho is exactly 1. This guards the
        # synthetic-complete cell against the PROPACK spurious-value regression.
        complete = sp.csr_matrix(np.ones((1000, 500)))
        self.assertAlmostEqual(_component_rho(complete), 0.0, places=6)

    def test_very_large_block_uses_arpack_only(self) -> None:
        # min-dim above PROPACK_MAX_MIN_DIM: ARPACK only, no PROPACK (which can
        # terminate the interpreter on very large irregular blocks).
        matrix = sp.eye(20_001, format="csr")
        calls: list[str] = []

        def fake_svds(*_args, **kwargs):
            calls.append(kwargs["solver"])
            return np.array([0.5, 1.0])

        with patch("benchmarks.modular.compute_hardness.svds", side_effect=fake_svds):
            self.assertEqual(_component_rho(matrix), 0.25)
        self.assertEqual(calls, ["arpack"])

    def test_sparse_block_uses_propack_first(self) -> None:
        # A block too large to densify but within the PROPACK dimension budget
        # takes the fast PROPACK path without invoking ARPACK.
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
            ["#agreement-simple", "`fixest`", "", "", "$1.1 times 10^(-14)$"],
            ["#agreement-difficult", "`fixest`", "", "", "$1.9 times 10^(-7)$"],
        ]
        self.assertEqual(_numeric_cell(rows[0][4]), 1.1e-14)
        self.assertEqual(
            _largest_backend_metric(rows, "fixest", 4), "$1.9 times 10^(-7)$"
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
        self.assertIn("table.cell(rowspan: 4)[simple], [`rust-map`]", fragment)
        self.assertNotIn("[table.cell(rowspan: 4)[simple]]", fragment)
        self.assertIn("\n  [`within`],", fragment)
        self.assertNotIn("\n  [], [`within`],", fragment)


if __name__ == "__main__":
    unittest.main()
