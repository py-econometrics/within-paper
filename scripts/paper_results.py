#!/usr/bin/env python3
"""Manage reproducible benchmark results for the paper.

Runtime preflight has no third-party imports, so it can run before the Pixi environment
is available. Rendering loads the AKM configuration only when it needs parameter labels.
Paper table data is stored in ``results/paper/benchmark_tables.json``; ``render`` writes
the Typst includes, and ``collect`` folds the raw benchmark output into them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
import urllib.request
import zipfile
from statistics import median
from pathlib import Path


from scripts.benchmark_methods import METHODS

ROOT = Path(__file__).absolute().parents[1]
LATEST_RUN = ROOT / "results" / "runs" / "latest"
CORREIA_DIR = ROOT / "benchmarks" / "data" / "correia_data"
TABLES_PATH = ROOT / "results" / "paper" / "benchmark_tables.json"
GENERATED_DIR = ROOT / "generated" / "tables"
RUNTIME_CONFIG = ROOT / "config" / "external_runtimes.json"
EXPECTED_TRIALS = 3

# The headline figure is a presentation of the two controlled AKM benchmark
# families.  The tables remain the canonical source of the gap calculation;
# this registry only fixes which package/runtime cells belong in each panel.
HEADLINE_FIGURE_BACKENDS = {
    "default": ("rust-map", "within", "fixest", "FEM.jl"),
    "matched": ("rust-map", "within-off", "within-diagonal", "within-additive"),
}
RENDERED_TABLES = (
    "agreement",
    "akm_setup_cost",
    "correia_real",
    "correia_synthetic",
    "iterations",
    "memory",
    "ols",
    "ppml",
    "regression_reuse",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latest(filename: str) -> Path:
    """One result file from the current run directory."""
    return LATEST_RUN / filename


def _latest_rows(filename: str) -> list[dict[str, str]] | None:
    """Rows of one result CSV, or None when the file is not there.

    Absent and empty are different and the synchronizers rely on the
    difference: a table whose benchmark has not been run is left as it stands,
    while a table whose benchmark ran and produced no usable row has its cells
    marked missing. Returning None for the first case keeps that distinction at
    the call site instead of making every caller re-test the path.
    """
    path = _latest(filename)
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(
        command, text=True, stderr=subprocess.STDOUT, env=env
    ).strip()


def _git_dirty() -> bool:
    try:
        return bool(_run(["git", "status", "--porcelain", "--untracked-files=all"]))
    except (OSError, subprocess.CalledProcessError):
        return True


def _positive_thread_setting(name: str) -> tuple[int | None, str | None]:
    value = os.environ.get(name, "")
    try:
        threads = int(value)
    except ValueError:
        return None, f"{name} must be set to a positive integer before running benchmarks"
    if threads < 1:
        return None, f"{name} must be set to a positive integer before running benchmarks"
    return threads, None


def check_external_runtimes(_: argparse.Namespace) -> None:
    config = _read_json(RUNTIME_CONFIG)
    failures: list[str] = []
    bench_threads, bench_threads_error = _positive_thread_setting("BENCH_THREADS")
    if bench_threads_error:
        failures.append(bench_threads_error)
    if shutil.which("Rscript") is None:
        failures.append("Rscript is not on PATH")
    else:
        expected = config["r"]["packages"]
        packages = ", ".join(repr(x) for x in expected)
        probe = (
            f"required <- c({packages}); missing <- required[!vapply(required, "
            "requireNamespace, logical(1), quietly=TRUE)]; "
            "if (length(missing)) stop(paste(missing, collapse=', ')); "
            "cat(R.version.string, '\\n'); "
            "for (p in required) cat(p, as.character(packageVersion(p)), '\\n')"
        )
        try:
            output = _run(["Rscript", "-e", probe])
            observed = {
                line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1].strip()
                for line in output.splitlines()
                if len(line.split(maxsplit=1)) == 2 and line.split(maxsplit=1)[0] in expected
            }
            mismatched = [
                f"{name}: expected {version}, found {observed.get(name, 'missing')}"
                for name, version in expected.items()
                if observed.get(name) != version
            ]
            if mismatched:
                failures.extend("native R package " + item for item in mismatched)
            if bench_threads is not None:
                configured_threads_output = _run(
                    [
                        "Rscript",
                        "-e",
                        "library(fixest); setFixest_nthreads(as.integer(Sys.getenv('BENCH_THREADS'))); "
                        "cat(getFixest_nthreads())",
                    ]
                )
                matches = re.findall(r"(?m)^\s*(\d+)\s*$", configured_threads_output)
                configured_threads = matches[-1] if matches else configured_threads_output.strip()
                if configured_threads.strip() != str(bench_threads):
                    failures.append(
                        f"fixest thread check: expected {bench_threads}, found {configured_threads.strip()}"
                    )
                else:
                    print(f"fixest threads: {configured_threads.strip()}")
            print(output)
        except subprocess.CalledProcessError as exc:
            failures.append(f"native R package check failed: {exc.output.strip()}")
    if shutil.which("julia") is None:
        failures.append("julia is not on PATH")
    else:
        project = ROOT / config["julia"]["project"]
        if not (project / "Project.toml").exists() or not (project / "Manifest.toml").exists():
            failures.append(f"tracked Julia Project/Manifest missing in {project}")
        else:
            with (project / "Manifest.toml").open("rb") as handle:
                expected_julia = tomllib.load(handle).get("julia_version")
            try:
                julia_env = dict(os.environ)
                if bench_threads is not None:
                    julia_env["JULIA_NUM_THREADS"] = str(bench_threads)
                julia_version = _run(["julia", "--version"], env=julia_env)
                if expected_julia and expected_julia not in julia_version:
                    failures.append(
                        f"Julia runtime: expected {expected_julia} from Manifest.toml, found {julia_version}"
                    )
                if bench_threads is not None:
                    configured_threads = _run(
                        ["julia", "-e", "print(Threads.nthreads())"], env=julia_env
                    )
                    if configured_threads.strip() != str(bench_threads):
                        failures.append(
                            f"Julia is using {configured_threads.strip()} thread(s), "
                            f"but BENCH_THREADS={bench_threads}"
                        )
                    else:
                        print(f"Julia threads: {configured_threads.strip()}")
                print(
                    _run(
                        ["julia", f"--project={project}", "-e", "using Pkg; Pkg.status()"],
                        env=julia_env,
                    )
                )
            except subprocess.CalledProcessError as exc:
                failures.append(f"Julia project check failed: {exc.output.strip()}")
    if failures:
        raise SystemExit("External runtime preflight failed:\n- " + "\n- ".join(failures))


def setup_julia_env(_: argparse.Namespace) -> None:
    project = ROOT / _read_json(RUNTIME_CONFIG)["julia"]["project"]
    if shutil.which("julia") is None:
        raise SystemExit("julia is not on PATH")
    subprocess.run(["julia", f"--project={project}", "-e", "using Pkg; Pkg.instantiate()"], check=True)


def fetch_correia(args: argparse.Namespace) -> None:
    """Download each archive listed in the metadata and verify its checksum."""
    metadata_dir = CORREIA_DIR / "metadata"
    if not metadata_dir.exists():
        raise SystemExit(f"Metadata directory not found: {metadata_dir}")
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = _read_json(metadata_path)
        slug = metadata["slug"]
        destination = CORREIA_DIR / f"{slug}.csv"
        if destination.exists() and _sha256(destination) == metadata["checksum_csv"]:
            print(f"[ok] {slug}: existing CSV checksum verified")
            continue
        if args.offline:
            raise SystemExit(f"Missing or invalid {destination}; --offline forbids download")
        archive = CORREIA_DIR / ".downloads" / f"{slug}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {slug}: {metadata['package_url']}")
        urllib.request.urlretrieve(metadata["package_url"], archive)
        if archive.stat().st_size != metadata["package_bytes"] or _sha256(archive) != metadata["package_sha256"]:
            archive.unlink(missing_ok=True)
            raise SystemExit(f"Checksum or size mismatch for {slug} package")
        with zipfile.ZipFile(archive) as bundle:
            csv_members = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise SystemExit(f"Expected one CSV in {archive}, found {csv_members}")
            with bundle.open(csv_members[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        if destination.stat().st_size != metadata["csv_bytes"] or _sha256(destination) != metadata["checksum_csv"]:
            destination.unlink(missing_ok=True)
            raise SystemExit(f"CSV checksum or size mismatch after extracting {slug}")
        print(f"[ok] {slug}: verified")


def _display_header(cell: str) -> str:
    """Replace internal backend keys with the paper's public method names."""
    if cell == "Backend":
        return "Configuration"
    key = cell.strip("`")
    return _method_header(key) if key in METHODS else cell


def _display_method_cell(cell: str) -> str:
    """Format an exact internal method key when it occurs in a table body."""
    key = cell.strip("`")
    if cell.startswith("`") and cell.endswith("`") and key in METHODS:
        return _method_header(key)
    return cell


def _method_header(key: str) -> str:
    lsmr_preconditioners = {
        "within-off": "none",
        "within-diagonal": "diagonal",
        "within": "factor-pair",
        "within-additive": "factor-pair",
    }
    if key in lsmr_preconditioners:
        return (
            "PyFixest #linebreak() LSMR #linebreak() "
            f"{lsmr_preconditioners[key]}"
        )
    label = METHODS[key][0]
    return label.replace(" ", " #linebreak() ", 1) if " " in label else label


AKM_PARAMETER_TABLES = {
    "akm_setup_cost": ("Move probability $delta$", "akm_mobility_"),
}


def _akm_parameterized_table(name: str, table: dict) -> dict:
    """Replace internal AKM design keys with their varied parameter at render time."""
    presentation = AKM_PARAMETER_TABLES.get(name)
    if presentation is None:
        return table
    header, prefix = presentation
    rows = []
    for source in table["rows"]:
        design = _row_label(table, source)
        if not design.startswith(prefix):
            raise ValueError(f"Unexpected design {design!r} in {name}")
        rows.append([_akm_parameter_label(design), *source[1:]])
    return {**table, "header": [header, *table["header"][1:]], "rows": rows}


def _table_fragment(name: str, table: dict) -> str:
    table = _akm_parameterized_table(name, table)
    lines = [
        "// Generated by scripts/paper_results.py; do not edit by hand.",
        "#let table-rule = rgb(\"#7b8494\")",
        "#let table-light-rule = rgb(\"#d8dee8\")",
        "#let table-head-fill = rgb(\"#eef2f7\")",
        "#let th(body) = table.cell(fill: table-head-fill)[#strong(body)]",
        "#let miss = text(fill: rgb(\"#777777\"))[--]",
        "#table(",
        f"  columns: {table['columns']},",
        "  stroke: 0.35pt + table-light-rule,",
        "  inset: (x: 5pt, y: 3.6pt),",
        f"  align: {table['align']},",
        "  table.hline(stroke: 0.8pt + table-rule),",
        "  table.header("
        + ", ".join(f"th[{_display_header(cell)}]" for cell in table["header"])
        + "),",
        "  table.hline(stroke: 0.45pt + table-rule),",
    ]
    for row in table["rows"]:
        marker = _row_label(table, row)
        if marker == "#memory-100k":
            lines.append("  table.cell(colspan: 4, fill: table-head-fill)[#emph[100K observations]],")
            continue
        if marker == "#memory-1m":
            lines.extend([
                "  table.hline(stroke: 0.35pt + table-light-rule),",
                "  table.cell(colspan: 4, fill: table-head-fill)[#emph[1M observations]],",
            ])
            continue
        if marker == "#agreement-simple":
            row = ["table.cell(rowspan: 4)[simple]", *row[1:]]
        elif marker == "#agreement-difficult":
            lines.append("  table.hline(stroke: 0.35pt + table-light-rule),")
            row = ["table.cell(rowspan: 4)[difficult]", *row[1:]]
        elif name == "agreement":
            # The first grid slot is already occupied by the row-spanning
            # design cell, so omit the empty marker from subsequent rows.
            row = row[1:]
        cells = [_display_method_cell(cell) if cell else "" for cell in row]
        rendered_cells = [
            cell if index == 0 and cell.startswith("table.cell(") else f"[{cell}]"
            for index, cell in enumerate(cells)
        ]
        lines.append("  " + ", ".join(rendered_cells) + ",")
    lines.extend(["  table.hline(stroke: 0.8pt + table-rule),", ")", ""])
    return "\n".join(lines)


def _akm_parameter_label(design: str) -> str:
    """The controlled variable printed in the appendix AKM tables.

    Import the generator configuration only when rendering.  Keeping the
    standard-library collector importable without the benchmark environment is
    useful for preflight checks and keeps the canonical table as the source of
    the measured values.
    """
    from benchmarks.akm import SCENARIOS

    config = SCENARIOS[design]
    value = config["delta"] if design.startswith("akm_mobility_") else config["rho"]
    return f"{value:g}"


def _akm_appendix_panel(table: dict, *, panel: str) -> dict:
    """Project one wide AKM table into a compact appendix panel.

    The source table deliberately remains unchanged: it is the one
    machine-readable location for the recorded package-default cells.  The
    two rendered panels merely group those cells by the comparison they answer.
    """
    source_name = _row_label(table, table["rows"][0])
    mobility = source_name.startswith("akm_mobility_")
    parameter = "Move probability $delta$" if mobility else "Sorting strength $rho$"
    if panel == "defaults":
        backends = ("rust-map", "fixest", "FEM.jl", "within")
        columns = "(1.15fr, 0.95fr, 0.82fr, 0.70fr, 0.70fr, 0.95fr)"
    elif panel == "lsmr":
        backends = ("within-off", "within-diagonal", "within")
        columns = "(1.20fr, 1.00fr, 0.90fr, 0.96fr, 1.00fr)"
    else:
        raise ValueError(f"Unknown AKM appendix panel {panel!r}")

    rows = []
    for source in table["rows"]:
        design = _row_label(table, source)
        rows.append(
            [
                _akm_parameter_label(design),
                _table_cell(table, source, "Gap (share)"),
                *(_table_cell(table, source, backend) for backend in backends),
            ]
        )
    return {
        "columns": columns,
        "align": "(right, right, right, right, right, right)"
        if panel == "defaults"
        else "(right, right, right, right, right)",
        "header": [parameter, "Gap (share)", *backends],
        "rows": rows,
    }


def render(_: argparse.Namespace) -> None:
    document = _read_json(TABLES_PATH)
    tables = document["tables"]
    destination = GENERATED_DIR
    destination.mkdir(parents=True, exist_ok=True)
    targets = {destination / f"{name}.typ" for name in RENDERED_TABLES}
    for family in ("mobility", "sorting"):
        for panel in ("defaults", "lsmr"):
            targets.add(destination / f"akm_{family}_{panel}.typ")
    for path in destination.glob("*.typ"):
        if path not in targets:
            path.unlink()
    for name in RENDERED_TABLES:
        table = tables[name]
        (destination / f"{name}.typ").write_text(_table_fragment(name, table), encoding="utf-8")
    for family in ("mobility", "sorting"):
        table = tables[f"akm_{family}"]
        for panel in ("defaults", "lsmr"):
            target = destination / f"akm_{family}_{panel}.typ"
            target.write_text(
                _table_fragment(target.stem, _akm_appendix_panel(table, panel=panel)),
                encoding="utf-8",
            )
    values = ["// Generated result values; do not edit by hand."]
    agreement_table = tables["agreement"]
    memory_table = tables["memory"]

    def memory_overheads(rows: list[list[str]]) -> list[float]:
        values = []
        for row in rows:
            map_memory = _numeric_cell(_table_cell(memory_table, row, "rust-map"))
            within_memory = _numeric_cell(_table_cell(memory_table, row, "within"))
            if map_memory is not None and within_memory is not None:
                values.append(within_memory - map_memory)
        return values

    memory_100k_rows = _rows_after_marker(memory_table, "#memory-100k")
    memory_1m_rows = _rows_after_marker(memory_table, "#memory-1m")
    memory_100k = memory_overheads(memory_100k_rows)
    memory_1m = memory_overheads(memory_1m_rows)
    agreement_simple_rows = _rows_after_marker(agreement_table, "#agreement-simple")
    agreement_difficult_rows = _rows_after_marker(agreement_table, "#agreement-difficult")
    prose_values = {
        "result_agreement_simple_max": _largest_metric(
            agreement_table, agreement_simple_rows, "Absolute difference"
        ),
        "result_agreement_difficult_max": _largest_metric(
            agreement_table, agreement_difficult_rows, "Absolute difference"
        ),
        "result_memory_100k_overhead": f"{min(memory_100k):.0f}--{max(memory_100k):.0f} MiB" if memory_100k else "--",
        "result_memory_1m_overhead": f"{min(memory_1m):.0f}--{max(memory_1m):.0f} MiB" if memory_1m else "--",
    }
    values.extend(f"#let {name} = [{_prose_cell(str(value))}]" for name, value in prose_values.items())
    (destination.parent / "paper_values.typ").write_text("\n".join(values) + "\n", encoding="utf-8")
    print(f"[render] wrote {len(targets)} table fragments to {destination}")


def collect(_: argparse.Namespace) -> None:
    """Fold the raw benchmark output into the canonical paper tables.

    The two refusals below are preconditions on the paper's numbers, not
    bookkeeping: a run against a development build of `within`, or against
    uncommitted benchmark code, produces timings nobody can trace back to a
    revision.
    """
    if os.environ.get("WITHIN_REPO"):
        raise SystemExit(
            "Refusing to collect paper results with WITHIN_REPO set; "
            "the paper run must use the locked within-py package"
        )
    if _git_dirty():
        raise SystemExit(
            "Refusing to collect paper results from a dirty tracked worktree; "
            "commit benchmark code and documentation first"
        )
    updated = _synchronize_canonical_tables()
    print(f"[collect] updated {updated} paper table cells")


def _rows_from_csvs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename in (
        "ols.csv",
        "ppml.csv",
        "akm.csv",
        "correia.csv",
    ):
        path = _latest(filename)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(
                {**row, "_source_file": filename} for row in csv.DictReader(handle)
            )
    return rows



def _format_seconds(value: float) -> str:
    if value < 1:
        return f"{value:.3f}s"
    if value < 10:
        return f"{value:.2f}s"
    return f"{value:.1f}s"


def _row_success(row: dict[str, str]) -> bool:
    return str(row.get("converged", "")).strip().lower() in {"true", "1"}


def _row_capped(row: dict[str, str]) -> bool:
    return str(row.get("capped", "")).strip().lower() in {"true", "1"}


def _row_time(row: dict[str, str]) -> str:
    return row.get("runtime_s", "")


def _cell_is_complete(candidates: list[dict[str, str]]) -> bool:
    planned = {_integer_field(row, "n_planned") for row in candidates}
    planned.discard(None)
    expected = next(iter(planned)) if len(planned) == 1 else EXPECTED_TRIALS
    repetitions = {_integer_field(row, "repetition") for row in candidates}
    return len(planned) <= 1 and repetitions == set(range(expected))


def _trial_key(row: dict[str, str]) -> int | None:
    return _integer_field(row, "repetition")


def _render_trial_result(candidates: list[dict[str, str]]) -> str:
    """Format the benchmark trials for one cell, including failed trials."""
    if not candidates:
        return "#miss"

    trial_ids = [_trial_key(row) for row in candidates]
    if None in trial_ids or len(set(trial_ids)) != len(trial_ids):
        return "incomplete"
    if not _cell_is_complete(candidates):
        return "incomplete"
    total = len(candidates)
    successful_rows = [row for row in candidates if _row_success(row)]
    successful = len(successful_rows)
    try:
        values = [float(_row_time(row)) for row in successful_rows]
    except (TypeError, ValueError):
        return "incomplete"
    if successful == 0:
        status = "capped" if all(_row_capped(row) for row in candidates) else "failed"
        return f"{status} (0/{total})"
    rendered = _format_seconds(median(values))
    if successful < total:
        return f"{rendered} ({successful}/{total})"
    return rendered


def _headline_point(
    candidates: list[dict[str, str]],
    *,
    design: str,
    family: str,
    view: str,
    backend: str,
    gap: float | None,
) -> dict[str, object]:
    """One structured record for the 2-by-2 headline figure.

    The plot must distinguish a returned median from an iteration cap.  A
    capped run has no successful fit, but its elapsed wall time is informative:
    it is a lower bound on the time required to finish the requested solve.
    Non-cap failures have no comparable timing and are left out of the plot.
    """
    if not candidates:
        return {
            "design": design,
            "family": family,
            "view": view,
            "backend": backend,
            "gap": gap,
            "median_time": None,
            "n_trials": 0,
            "n_success": 0,
            "n_capped": 0,
            "status": "missing",
        }

    successful = [row for row in candidates if _row_success(row)]
    capped = [row for row in candidates if _row_capped(row)]

    def median_time(rows: list[dict[str, str]]) -> float | None:
        values = []
        for row in rows:
            try:
                value = float(_row_time(row))
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        return float(median(values)) if values else None

    complete = _cell_is_complete(candidates)
    if successful:
        status = "complete" if complete and len(successful) == len(candidates) else "partial"
        elapsed = median_time(successful)
    elif complete and len(capped) == len(candidates):
        status = "capped"
        elapsed = median_time(capped)
    elif complete:
        status = "failed"
        elapsed = None
    else:
        status = "incomplete"
        elapsed = median_time(successful)

    return {
        "design": design,
        "family": family,
        "view": view,
        "backend": backend,
        "gap": gap,
        "median_time": elapsed,
        "n_trials": len(candidates),
        "n_success": len(successful),
        "n_capped": len(capped),
        "status": status,
    }


def _synchronize_headline_figure(document: dict, raw: list[dict[str, str]]) -> int:
    """Record the AKM headline-figure projection in the tracked result file.

    An absent raw AKM file means that the current benchmark command did not run
    this experiment.  In that case keep the last collected figure records;
    treating absence as a new set of failures would erase a valid paper figure.
    """
    if not _latest("akm.csv").exists():
        return 0

    points: list[dict[str, object]] = []
    for family in ("mobility", "sorting"):
        table = document["tables"][f"akm_{family}"]
        for view in ("default", "matched"):
            for source in table["rows"]:
                design = _row_label(table, source)
                gap = _numeric_cell(_table_cell(table, source, "Gap (share)"))
                for backend in HEADLINE_FIGURE_BACKENDS[view]:
                    candidates = [
                        row
                        for row in raw
                        if _matches_runtime_target(
                            row,
                            design,
                            backend,
                            {"n_obs": 1_000_000, "n_fe": 3},
                            f"akm.csv:{view}",
                        )
                    ]
                    points.append(
                        _headline_point(
                            candidates,
                            design=design,
                            family=family,
                            view=view,
                            backend=backend,
                            gap=gap,
                        )
                    )

    figure = {"schema_version": 1, "points": points}
    if document.get("headline_figure") == figure:
        return 0
    document["headline_figure"] = figure
    return len(points)


def _integer_field(row: dict[str, str], name: str) -> int | None:
    try:
        return int(float(row.get(name, "")))
    except (TypeError, ValueError):
        return None


def _validate_ppml_results(rows: list[dict[str, str]]) -> None:
    unexpected = sorted(
        {
            str(row.get("n_fe") or "missing")
            for row in rows
            if row.get("_source_file") == "ppml.csv"
            and _integer_field(row, "n_fe") != 3
        }
    )
    if unexpected:
        raise ValueError(
            "PPML result files must contain only n_fe=3 rows; found "
            f"{', '.join(unexpected)}."
        )


def _paper_runtime_target(
    table_name: str, table: dict, row: list[str]
) -> tuple[str, dict[str, int], str]:
    """Return the exact run specification represented by a paper timing cell."""
    dataset = _row_label(table, row).split(" ")[0]
    if table_name in {"ols", "ppml"}:
        dataset = dataset.split("(")[0]
    if table_name in {"akm_mobility", "akm_sorting"}:
        return dataset, {"n_obs": 1_000_000, "n_fe": 3}, "akm.csv:default"
    if table_name == "ols":
        return dataset, {"n_obs": 10_000_000, "n_fe": 3}, "ols.csv:default"
    if table_name == "ppml":
        return (
            dataset,
            {"n_obs": 1_000_000, "n_fe": int(_table_cell(table, row, "FE"))},
            "ppml.csv:default",
        )
    return dataset, {"n_fe": 2}, "correia.csv:default"


def _matches_runtime_target(
    row: dict[str, str],
    dataset: str,
    backend: str,
    requirements: dict[str, int],
    source_marker: str,
) -> bool:
    filename, view = source_marker.split(":", 1)
    if row.get("_source_file") != filename or row.get("view") != view:
        return False
    if row.get("design") != dataset:
        return False
    raw_backend = row.get("backend", "")
    if raw_backend != backend:
        return False
    return all(_integer_field(row, field) == value for field, value in requirements.items())


def _numeric_cell(value: str) -> float | None:
    if "failed" in value.lower() or "capped" in value.lower():
        # A failed or capped cell such as "capped (0/3)" carries no runtime; do not
        # let the "0" in the trial count read back as a 0.0-second measurement.
        return None
    scientific = re.search(
        r"(-?\d[\d,]*\.?\d*)\s+times\s+10\^\((-?\d+)\)", value
    )
    if scientific is not None:
        mantissa = float(scientific.group(1).replace(",", ""))
        return mantissa * 10 ** int(scientific.group(2))
    match = re.search(r"(?:\d[\d,]*\.?\d*|\.\d+)", value)
    if match is None:
        return None
    return float(match.group().replace(",", ""))


def _largest_metric(
    table: dict,
    rows: list[list[str]],
    column: str,
    *,
    backend: str | None = None,
) -> str:
    """The largest numeric cell in one column, optionally within one backend.

    Cells that do not parse as a number are skipped rather than read as zero,
    so a "--" or a "#miss" marker cannot win the comparison and be reported as
    a measured worst case.
    """
    candidates = [
        _table_cell(table, row, column)
        for row in rows
        if (backend is None or _clean_cell(_table_cell(table, row, "Backend")) == backend)
        and _numeric_cell(_table_cell(table, row, column)) is not None
    ]
    if not candidates:
        return "--"
    return max(candidates, key=lambda value: _numeric_cell(value) or 0.0)


def _format_typst_scientific(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(f"{value:.0e}".split("e")[1])
    mantissa = value / (10**exponent)
    return f"${mantissa:.1f} times 10^({exponent})$"


def _clean_cell(value: str) -> str:
    return value.replace("`", "")


def _header_index(table: dict, header: str) -> int:
    """Return a table column by its stable semantic header, not its position."""
    headers = [_clean_cell(value) for value in table["header"]]
    try:
        return headers.index(header)
    except ValueError as error:
        raise ValueError(f"Table does not have a {header!r} column") from error


def _table_cell(table: dict, row: list[str], header: str) -> str:
    return row[_header_index(table, header)]


def _set_table_cell(table: dict, row: list[str], header: str, value: str) -> None:
    row[_header_index(table, header)] = value


def _row_label(table: dict, row: list[str]) -> str:
    for header in ("Scenario", "Design", "Dataset"):
        try:
            return _clean_cell(_table_cell(table, row, header))
        except ValueError:
            continue
    # Structural table markers always occupy the first cell. Tables such as the
    # tolerance frontier use a different label column but still need rendering.
    return _clean_cell(row[0])


def _rows_after_marker(table: dict, marker: str) -> list[list[str]]:
    """Return rows in the section immediately following a structural marker."""
    rows = table["rows"]
    for index, row in enumerate(rows):
        if _row_label(table, row) != marker:
            continue
        section = []
        for candidate in rows[index + 1 :]:
            if _row_label(table, candidate).startswith("#"):
                break
            section.append(candidate)
        return section
    raise ValueError(f"Table marker {marker!r} was not found")


def _backend_columns(table: dict) -> tuple[tuple[int, str], ...]:
    return tuple(
        (index, _clean_cell(header))
        for index, header in enumerate(table["header"])
        if _clean_cell(header) in METHODS
    )


def _ensure_akm_runtime_rows(document: dict) -> int:
    """Match the canonical AKM rows to the registered designs."""
    from benchmarks.akm import SCENARIOS

    changed = 0
    for family in ("mobility", "sorting"):
        prefix = f"akm_{family}_"
        expected = [name for name in SCENARIOS if name.startswith(prefix)]
        table = document["tables"][f"akm_{family}"]
        rows_by_design = {_row_label(table, row): row for row in table["rows"]}
        synchronized = [
            rows_by_design.get(
                design,
                [f"`{design}`", *("#miss" for _ in table["header"][1:])],
            )
            for design in expected
        ]
        if synchronized != table["rows"]:
            old_designs = set(rows_by_design)
            changed += max(1, len(old_designs ^ set(expected)))
            table["rows"] = synchronized
    return changed


def _prose_cell(value: str) -> str:
    """Replace failure markers before inserting a value into Typst text."""
    if value == "#miss" or value == "--" or value.startswith(("failed", "capped")):
        return "--"
    return value


def _format_hardness(gap: float, share: float) -> str:
    """Format a gap and component share for Typst."""
    if gap and abs(gap) < 1e-2:
        exponent = int(f"{gap:.0e}".split("e")[1])
        mantissa = gap / (10**exponent)
        gap_text = f"${mantissa:.2f} times 10^({exponent})$"
    elif gap >= 1.0:
        gap_text = f"{gap:.2f}"
    else:
        gap_text = f"{gap:.3g}"
    return f"{gap_text} ({share:.2f})"


def _synchronize_hardness(document: dict) -> int:
    rows = _latest_rows("hardness.csv")
    # A partial collection must not erase an earlier gap.
    if rows is None:
        return 0
    diagnostics = {
        row["dataset_id"]: row
        for row in rows
        if {row["fe_a"], row["fe_b"]} in ({"indiv_id", "firm_id"}, {"id1", "id2"})
    }

    def update(source_id: str, target_row: list[str]) -> int:
        diagnostic = diagnostics.get(source_id)
        if diagnostic is None:
            return 0
        rendered = _format_hardness(
            float(diagnostic["one_minus_rho"]),
            float(diagnostic["worst_component_obs_share"]),
        )
        if _table_cell(table, target_row, "Gap (share)") == rendered:
            return 0
        _set_table_cell(table, target_row, "Gap (share)", rendered)
        return 1

    changed = 0
    for table_name in ("akm_mobility", "akm_sorting"):
        table = document["tables"][table_name]
        for row in table["rows"]:
            scenario = _row_label(table, row)
            changed += update(scenario, row)
    table = document["tables"]["ols"]
    for row in table["rows"]:
        family = _row_label(table, row).split()[0]
        changed += update(family, row)
    for table_name in ("correia_synthetic", "correia_real"):
        table = document["tables"][table_name]
        for row in table["rows"]:
            changed += update(_row_label(table, row), row)
    table = document["tables"]["memory"]
    for index, row in enumerate(table["rows"]):
        if _row_label(table, row).startswith("#"):
            continue
        family = _row_label(table, row).split()[0]
        size = "100k" if index < 4 else "1m"
        changed += update(f"memory_{family}_{size}", row)
    return changed


def _synchronize_agreement(document: dict) -> int:
    observations = _latest_rows("agreement.csv")
    if observations is None:
        return 0
    by_key = {
        (row["design"], row["backend"]): row
        for row in observations
    }
    changed = 0
    dgp = ""
    table = document["tables"]["agreement"]
    for row in table["rows"]:
        marker = _row_label(table, row)
        if marker == "#agreement-simple":
            dgp = "simple"
        elif marker == "#agreement-difficult":
            dgp = "difficult"
        backend = _clean_cell(_table_cell(table, row, "Backend"))
        source = by_key.get((dgp, backend))
        if source is None:
            replacement = ["#miss", "#miss"]
        elif not _row_success(source):
            status = "capped" if _row_capped(source) else "failed"
            replacement = [status, status]
        else:
            replacement = [
                f"{float(source['x1']):.8f}",
                "--" if backend == "rust-map" else _format_typst_scientific(float(source["max_abs_diff"])),
            ]
        for header, value in zip(("$hat(beta)_1$", "Absolute difference"), replacement):
            if _table_cell(table, row, header) != value:
                _set_table_cell(table, row, header, value)
                changed += 1
    return changed


def _synchronize_akm_setup_cost(document: dict) -> int:
    """Fill setup and solve times for two- and three-factor AKM designs."""
    rows = _latest_rows("akm_setup_cost.csv")
    table = document["tables"].get("akm_setup_cost")
    if table is None or rows is None:
        return 0

    mobility_table = document["tables"]["akm_mobility"]
    gap_by_design = {
        _row_label(mobility_table, row): _table_cell(mobility_table, row, "Gap (share)")
        for row in mobility_table["rows"]
    }

    def cells(design: str, n_factors: int) -> list[str]:
        group = [
            row
            for row in rows
            if row.get("design") == design
            and _integer_field(row, "n_factors") == n_factors
        ]
        successful = [row for row in group if _row_success(row)]
        if not group:
            return ["#miss", "#miss"]
        if not _cell_is_complete(group):
            return ["incomplete", "incomplete"]
        if not successful:
            status = "capped" if all(_row_capped(row) for row in group) else "failed"
            return [status, status]
        return [
            _format_seconds(median(float(row["setup_s"]) for row in successful)),
            _format_seconds(median(float(row["solve_s"]) for row in successful)),
        ]

    rendered = []
    for source in mobility_table["rows"]:
        design = _row_label(mobility_table, source)
        rendered.append(
            [
                f"`{design}`",
                gap_by_design[design],
                *cells(design, 2),
                *cells(design, 3),
            ]
        )
    if rendered == table["rows"]:
        return 0
    table["rows"] = rendered
    return len(rendered)


def _synchronize_regression_reuse(document: dict) -> int:
    """Fill the ten-regression comparison of preconditioner cache policies."""
    rows = _latest_rows("regression_reuse.csv")
    table = document["tables"].get("regression_reuse")
    if table is None or rows is None:
        return 0

    policies = (
        ("diagonal", "Diagonal"),
        ("additive_rebuilt", "Additive, rebuilt"),
        ("additive_cached", "Additive, cached"),
    )
    designs = ("simple", "difficult")
    summaries: dict[tuple[str, str], tuple[float, float, float] | str] = {}
    for design in designs:
        for policy, _label in policies:
            group = [
                row
                for row in rows
                if row.get("design") == design and row.get("policy") == policy
            ]
            successful = [row for row in group if _row_success(row)]
            key = (design, policy)
            if not group:
                summaries[key] = "#miss"
            elif not _cell_is_complete(group):
                summaries[key] = "incomplete"
            elif not successful:
                summaries[key] = (
                    "capped" if all(_row_capped(row) for row in group) else "failed"
                )
            else:
                summaries[key] = tuple(
                    median(float(row[field]) for row in successful)
                    for field in ("setup_s", "solve_s", "total_s")
                )

    rendered = []
    for design in designs:
        baseline = summaries[(design, "diagonal")]
        baseline_total = baseline[2] if isinstance(baseline, tuple) else None
        for index, (policy, label) in enumerate(policies):
            summary = summaries[(design, policy)]
            design_cell = design if index == 0 else ""
            if not isinstance(summary, tuple):
                rendered.append(
                    [design_cell, label, summary, summary, summary, "--"]
                )
                continue
            setup, solve, total = summary
            speedup = f"{baseline_total / total:.1f}x" if baseline_total else "--"
            rendered.append(
                [
                    design_cell,
                    label,
                    _format_seconds(setup),
                    _format_seconds(solve),
                    _format_seconds(total),
                    speedup,
                ]
            )
    if rendered == table["rows"]:
        return 0
    table["rows"] = rendered
    return len(rendered)


ITERATION_COLUMNS = (
    ("rust-map", "map-sweep"),
    ("within-off", "lsmr-iteration"),
    ("within-diagonal", "lsmr-iteration"),
    ("within-additive", "lsmr-iteration"),
)


def _synchronize_iterations(document: dict) -> int:
    """Fill the iteration-count table, in each solver's own unit.

    A MAP sweep is a full pass over the absorbed factors; an LSMR iteration is
    one application of the operator and its transpose. The two are never added
    or plotted on one axis, so the table records which unit each column is in
    and the median is taken within a column only.
    """
    rows = _latest_rows("within_preconditioners.csv")
    table = document["tables"].get("iterations")
    if table is None or rows is None:
        return 0
    rendered = _iteration_rows(rows)
    if rendered == table["rows"]:
        return 0
    table["rows"] = rendered
    return len(rendered)


def _iteration_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    """One row per design, one column per solver, each in its own unit."""
    designs: list[str] = []
    for row in rows:
        if row["design"] not in designs:
            designs.append(row["design"])

    rendered: list[list[str]] = []
    for design in designs:
        cells = [design]
        for label, _unit in ITERATION_COLUMNS:
            trials = [r for r in rows if r["design"] == design and r["backend"] == label]
            counts = [
                float(r["iterations_max"]) for r in trials if r.get("iterations_max", "")
            ]
            if not counts:
                # A solver with no rows is absent, not zero: the run that
                # produced this file may predate the column.
                cells.append("#miss")
                continue
            # A capped cell keeps its count and says so. Dropping it would make
            # a solver that never converged look like the fastest one.
            capped = any(str(r.get("capped", "")).lower() in {"true", "1"} for r in trials)
            value = f"{median(counts):.0f}"
            cells.append(f"{value} (capped)" if capped else value)
        rendered.append(cells)
    return rendered


def _synchronize_canonical_tables(
    document: dict | None = None, *, write: bool = True
) -> int:
    """Update runtime cells from current raw CSV files.

    Keep the separately computed gap and component-share values. Replace a runtime only
    when the new output records all expected trials.
    """
    raw = _rows_from_csvs()
    _validate_ppml_results(raw)
    if document is None:
        document = _read_json(TABLES_PATH)
    changed = _ensure_akm_runtime_rows(document)
    runtime_tables = {
        "ols", "ppml", "akm_mobility", "akm_sorting",
        "correia_synthetic", "correia_real",
    }
    for name, table in document["tables"].items():
        if name not in runtime_tables:
            continue
        for row in table["rows"]:
            dataset, requirements, source_marker = _paper_runtime_target(name, table, row)
            filename, _view = source_marker.split(":", 1)
            # An absent raw result means that this experiment was not part of
            # the current run. It is different from a present file with no
            # matching row, which should continue to show as missing.
            if not _latest(filename).exists():
                continue
            for column, backend in _backend_columns(table):
                backend_source = source_marker
                candidates = [
                    source
                    for source in raw
                    if _matches_runtime_target(
                        source, dataset, backend, requirements, backend_source
                    )
                ]
                rendered = _render_trial_result(candidates)
                if row[column] != rendered:
                    row[column] = rendered
                    changed += 1
    measurements = _latest_rows("memory.csv")
    if measurements is not None:
        table = document["tables"]["memory"]
        size = ""
        for row in table["rows"]:
            label = _row_label(table, row)
            if label == "#memory-100k":
                size = "100k"
                continue
            if label == "#memory-1m":
                size = "1m"
                continue
            dgp = label.split()[0]
            for column, backend in _backend_columns(table):
                candidates = [
                    item
                    for item in measurements
                    if item["design"] == dgp
                    and item["size"] == size
                    and item["backend"] == backend
                ]
                match = next(
                    (
                        item
                        for item in candidates
                        if _row_success(item)
                    ),
                    None,
                )
                if match and match["rss_mb"]:
                    rendered = f"{int(float(match['rss_mb'])):,} MiB"
                elif candidates:
                    rendered = (
                        "capped"
                        if all(_row_capped(item) for item in candidates)
                        else "failed"
                    )
                else:
                    rendered = "#miss"
                if row[column] != rendered:
                    row[column] = rendered
                    changed += 1
    changed += _synchronize_hardness(document)
    changed += _synchronize_headline_figure(document, raw)
    changed += _synchronize_agreement(document)
    changed += _synchronize_iterations(document)
    changed += _synchronize_akm_setup_cost(document)
    changed += _synchronize_regression_reuse(document)
    if write:
        _write_json(TABLES_PATH, document)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-external-runtimes").set_defaults(func=check_external_runtimes)
    sub.add_parser("setup-julia-env").set_defaults(func=setup_julia_env)
    fetch = sub.add_parser("fetch-correia")
    fetch.add_argument(
        "--offline",
        action="store_true",
        help="Validate local CSVs without network access",
    )
    fetch.set_defaults(func=fetch_correia)
    sub.add_parser("collect").set_defaults(func=collect)
    sub.add_parser("render").set_defaults(func=render)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
