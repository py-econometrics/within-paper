from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "modular"))

from timing import randomized_order, repetitions_for_runtime, summarize_times  # noqa: E402


class TimingTests(unittest.TestCase):
    def test_repetition_counts_follow_runtime_bands(self) -> None:
        self.assertEqual(repetitions_for_runtime(0.5), 20)
        self.assertEqual(repetitions_for_runtime(5.0), 7)
        self.assertEqual(repetitions_for_runtime(10.0), 3)

    def test_failures_remain_in_the_timing_denominator(self) -> None:
        summary = summarize_times([1.0, None, 3.0])
        self.assertEqual(summary.median_s, 2.0)
        self.assertEqual(summary.n_attempted, 3)
        self.assertEqual(summary.n_converged, 2)

    def test_backend_order_is_seeded(self) -> None:
        self.assertEqual(randomized_order(["a", "b", "c"], 12), randomized_order(["a", "b", "c"], 12))


if __name__ == "__main__":
    unittest.main()
