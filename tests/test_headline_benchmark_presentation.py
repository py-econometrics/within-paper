"""Checks for the paper-facing mobility and sorting benchmark presentation."""

from __future__ import annotations

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import make_figures, paper_results


def _raw_trial(
    backend: str,
    repetition: int,
    *,
    converged: bool,
    capped: bool = False,
    runtime_s: float = 1.0,
    view: str = "default",
) -> dict[str, object]:
    return {
        "backend": backend,
        "repetition": repetition,
        "runtime_s": runtime_s,
        "converged": converged,
        "capped": capped,
        "n_planned": 3,
        "design": "akm_mobility_1",
        "n_obs": 1_000_000,
        "n_fe": 3,
        "view": view,
    }


class HeadlineFigureCollectionTests(unittest.TestCase):
    def test_records_preserve_convergence_and_cap_status(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        rows = [
            *[
                _raw_trial("rust-map", repetition, converged=True, runtime_s=runtime)
                for repetition, runtime in enumerate((1.0, 2.0, 3.0))
            ],
            _raw_trial("within", 0, converged=True, runtime_s=1.0),
            _raw_trial("within", 1, converged=False, runtime_s=2.0),
            _raw_trial("within", 2, converged=True, runtime_s=3.0),
            *[
                _raw_trial(
                    "within-off",
                    repetition,
                    converged=False,
                    capped=True,
                    runtime_s=runtime,
                    view="matched",
                )
                for repetition, runtime in enumerate((4.0, 5.0, 6.0))
            ],
            *[
                _raw_trial(
                    "within-diagonal",
                    repetition,
                    converged=False,
                    runtime_s=runtime,
                    view="matched",
                )
                for repetition, runtime in enumerate((4.0, 5.0, 6.0))
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory)
            with (latest / "akm.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with patch.object(paper_results, "LATEST_RUN", latest):
                raw = paper_results._rows_from_csvs()
                paper_results._synchronize_headline_figure(document, raw)

        records = {
            (row["view"], row["backend"]): row
            for row in document["headline_figure"]["points"]
            if row["family"] == "mobility" and row["design"] == "akm_mobility_1"
        }
        self.assertEqual(records[("default", "rust-map")]["status"], "complete")
        self.assertEqual(records[("default", "rust-map")]["median_time"], 2.0)
        self.assertEqual(records[("default", "within")]["status"], "partial")
        self.assertEqual(records[("default", "within")]["n_success"], 2)
        self.assertEqual(records[("matched", "within-off")]["status"], "capped")
        self.assertEqual(records[("matched", "within-off")]["median_time"], 5.0)
        self.assertEqual(records[("matched", "within-diagonal")]["status"], "failed")
        self.assertIsNone(records[("matched", "within-diagonal")]["median_time"])
        self.assertEqual(records[("default", "fixest")]["status"], "missing")

    def test_absent_akm_file_preserves_collected_figure_records(self) -> None:
        document = {"headline_figure": {"schema_version": 1, "points": [{"status": "complete"}]}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(paper_results, "LATEST_RUN", Path(directory)):
                changed = paper_results._synchronize_headline_figure(document, [])
        self.assertEqual(changed, 0)
        self.assertEqual(document["headline_figure"]["points"], [{"status": "complete"}])

    def test_incomplete_successful_cell_remains_plottable_as_hollow(self) -> None:
        point = paper_results._headline_point(
            [_raw_trial("rust-map", 0, converged=True, runtime_s=2.0)],
            design="akm_mobility_1",
            family="mobility",
            view="default",
            backend="rust-map",
            gap=0.4,
        )
        self.assertEqual(point["status"], "partial")
        self.assertTrue(make_figures._visible_point(point))

    def test_appendix_panels_use_default_lsmr_cells_and_generator_parameters(self) -> None:
        document = json.loads(paper_results.TABLES_PATH.read_text(encoding="utf-8"))
        table = document["tables"]["akm_mobility"]
        defaults = paper_results._akm_appendix_panel(table, panel="defaults")
        lsmr = paper_results._akm_appendix_panel(table, panel="lsmr")

        self.assertEqual(defaults["header"][:2], ["Move probability", "Gap (share)"])
        self.assertEqual(lsmr["header"][2:], ["within-off", "within-diagonal", "within"])
        self.assertEqual(defaults["rows"][0][0], "1")
        self.assertEqual(defaults["rows"][-1][0], "0.001")
        self.assertEqual(lsmr["rows"][0][-1], table["rows"][0][5])
        self.assertEqual(lsmr["rows"][0][2:4], table["rows"][0][3:5])

    def test_render_writes_split_akm_panels_without_scenario_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "tables"
            with patch.object(paper_results, "GENERATED_DIR", generated):
                paper_results.render(None)
            default_panel = (generated / "akm_mobility_defaults.typ").read_text(encoding="utf-8")
            lsmr_panel = (generated / "akm_sorting_lsmr.typ").read_text(encoding="utf-8")

        self.assertIn("Move probability", default_panel)
        self.assertIn("Sorting strength", lsmr_panel)
        self.assertNotIn("akm_mobility_1", default_panel)
        self.assertIn("0.543s", default_panel)


class HeadlineFigurePlotTests(unittest.TestCase):
    @staticmethod
    def _points() -> list[dict[str, object]]:
        points = []
        gaps = {"mobility": (0.4, 0.01), "sorting": (0.02, 0.002)}
        for family, (high, low) in gaps.items():
            for view, backends in (
                ("default", make_figures.CROSS_PACKAGE_BACKENDS),
                ("matched", make_figures.MECHANISM_BACKENDS),
            ):
                for backend in backends:
                    for index, gap in enumerate((high, low)):
                        points.append(
                            {
                                "design": f"akm_{family}_{index + 1}",
                                "family": family,
                                "view": view,
                                "backend": backend,
                                "gap": gap,
                                "median_time": index + 1.0,
                                "n_trials": 3,
                                "n_success": 3,
                                "n_capped": 0,
                                "status": "complete",
                            }
                        )
        return points

    def test_layout_has_four_family_view_panels_and_reversed_row_scales(self) -> None:
        points = self._points()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "headline.svg"
            with patch.object(make_figures, "_runtime_panel", wraps=make_figures._runtime_panel) as panel:
                make_figures.headline_figure(points, output)
            self.assertTrue(output.exists())

        self.assertEqual(panel.call_count, 4)
        observed = [(call.kwargs["family"], call.kwargs["view"]) for call in panel.call_args_list]
        self.assertEqual(
            observed,
            [
                ("mobility", "default"),
                ("mobility", "matched"),
                ("sorting", "default"),
                ("sorting", "matched"),
            ],
        )
        mobility_limits = [call.kwargs["x_limits"] for call in panel.call_args_list[:2]]
        sorting_limits = [call.kwargs["x_limits"] for call in panel.call_args_list[2:]]
        legend_panels = [call.kwargs["show_legend"] for call in panel.call_args_list]
        self.assertEqual(mobility_limits[0], mobility_limits[1])
        self.assertEqual(sorting_limits[0], sorting_limits[1])
        self.assertGreater(mobility_limits[0][0], mobility_limits[0][1])
        self.assertGreater(sorting_limits[0][0], sorting_limits[0][1])
        self.assertEqual(legend_panels, [True, True, False, False])

    def test_runtime_panel_connects_returned_points_without_fitting(self) -> None:
        source = inspect.getsource(make_figures._runtime_panel)
        self.assertNotIn("polyfit", source)
        self.assertIn("ax.plot", source)
        self.assertIn("if show_legend", source)
        self.assertIn('loc="upper left"', source)
        self.assertIn(
            "Lines join returned medians",
            inspect.getsource(make_figures.headline_figure),
        )

    def test_runtime_lines_exclude_capped_lower_bounds(self) -> None:
        points = [
            {
                "family": "mobility",
                "view": "default",
                "backend": "rust-map",
                "gap": 0.4,
                "median_time": 1.0,
                "status": "complete",
            },
            {
                "family": "mobility",
                "view": "default",
                "backend": "rust-map",
                "gap": 0.1,
                "median_time": 2.0,
                "status": "partial",
            },
            {
                "family": "mobility",
                "view": "default",
                "backend": "rust-map",
                "gap": 0.01,
                "median_time": 5.0,
                "status": "capped",
            },
        ]
        figure, ax = make_figures.plt.subplots()
        try:
            make_figures._runtime_panel(
                ax,
                points,
                ("rust-map",),
                family="mobility",
                view="default",
                title="test",
                x_limits=(0.5, 0.005),
                show_legend=False,
            )
            self.assertEqual(len(ax.lines), 1)
            self.assertEqual(ax.lines[0].get_xdata().tolist(), [0.4, 0.1])
            self.assertEqual(ax.lines[0].get_ydata().tolist(), [1.0, 2.0])
        finally:
            make_figures.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
