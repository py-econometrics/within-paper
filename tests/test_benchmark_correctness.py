from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULAR = ROOT / "benchmarks" / "modular"
BENCHMARKS = ROOT / "benchmarks"
for path in (MODULAR, BENCHMARKS):
    sys.path.insert(0, str(path))

from bench_within_setup_cost import _setup_share  # noqa: E402
from benchmark_correia import summarize_results  # noqa: E402
from dgps import _seed_for  # noqa: E402
from feols_benchmarkers import _as_bool  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
