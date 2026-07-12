from __future__ import annotations

import sys
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
from dgps import _seed_for  # noqa: E402
from feols_benchmarkers import _as_bool  # noqa: E402
from paper_results import _render_trial_result, _synchronize_external_results  # noqa: E402
from benchmarks.modular.compute_hardness import _component_rho  # noqa: E402


class BenchmarkCorrectnessTests(unittest.TestCase):
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

    def test_large_hardness_blocks_use_arpack(self) -> None:
        matrix = sp.eye(20_001, format="csr")
        with patch(
            "benchmarks.modular.compute_hardness.svds",
            return_value=np.array([0.5, 1.0]),
        ) as mocked_svds:
            self.assertEqual(_component_rho(matrix), 0.25)
        self.assertEqual(mocked_svds.call_args.kwargs["solver"], "arpack")

    def test_smaller_hardness_blocks_use_propack(self) -> None:
        matrix = sp.eye(65, format="csr")
        with patch(
            "benchmarks.modular.compute_hardness.svds",
            return_value=np.array([0.5, 1.0]),
        ) as mocked_svds:
            self.assertEqual(_component_rho(matrix), 0.25)
        self.assertEqual(mocked_svds.call_args.kwargs["solver"], "propack")


if __name__ == "__main__":
    unittest.main()
