#!/usr/bin/env python3
"""Manage reproducible benchmark results for the paper.

This script uses only the Python standard library, so runtime checks can run before the
Pixi environment is available. Paper table data is stored in
``results/paper/benchmark_tables.json``; ``render`` writes the Typst includes, and
``collect`` folds the raw benchmark output into them.
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
import time
import urllib.request
import zipfile
from statistics import median
from pathlib import Path


# The timing rules live with the benchmark harness that produces the numbers,
# not beside the renderer that formats them, so that "median over converged
# trials, failures kept in the denominator" has one implementation rather than
# one per consumer. The module is standard-library only, which is what lets
# this script keep running before the Pixi environment exists; a test pins
# that property.
from benchmarks.core.timing import summarize_times
from benchmarks.core.methods import METHOD_TABLE_HEADER
from benchmarks.core.paths import CORREIA_DIR, LATEST_RUN, ROOT
TABLES_PATH = ROOT / "results" / "paper" / "benchmark_tables.json"
GENERATED_DIR = ROOT / "generated" / "tables"
RUNTIME_CONFIG = ROOT / "config" / "external_runtimes.json"
EXPECTED_TRIALS = 3


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


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


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
    julia_threads, julia_threads_error = _positive_thread_setting("JULIA_NUM_THREADS")
    failures.extend(error for error in (bench_threads_error, julia_threads_error) if error)
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
                julia_version = _run(["julia", "--version"])
                if expected_julia and expected_julia not in julia_version:
                    failures.append(
                        f"Julia runtime: expected {expected_julia} from Manifest.toml, found {julia_version}"
                    )
                if julia_threads is not None:
                    configured_threads = _run(["julia", "-e", "print(Threads.nthreads())"])
                    if configured_threads.strip() != str(julia_threads):
                        failures.append(
                            f"Julia is using {configured_threads.strip()} thread(s), "
                            f"but JULIA_NUM_THREADS={julia_threads}"
                        )
                    else:
                        print(f"Julia threads: {configured_threads.strip()}")
                print(_run(["julia", f"--project={project}", "-e", "using Pkg; Pkg.status()"] ))
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
    selected = set(args.datasets or [])
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = _read_json(metadata_path)
        slug = metadata["slug"]
        if selected and slug not in selected:
            continue
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
    return METHOD_TABLE_HEADER.get(key, cell)


def _display_method_cell(cell: str) -> str:
    """Format an exact internal method key when it occurs in a table body."""
    key = cell.strip("`")
    if cell.startswith("`") and cell.endswith("`"):
        return METHOD_TABLE_HEADER.get(key, cell)
    return cell


def _table_fragment(name: str, table: dict) -> str:
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
        marker = row[0]
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


def render(args: argparse.Namespace) -> None:
    # Synchronize before rendering so that a table fragment can never be built
    # from a result file that is older than the raw benchmark output beside it.
    # Rendering used to read the stored document as-is, which made every table
    # correct only until the next benchmark run.
    _synchronize_canonical_tables()
    document = _read_json(TABLES_PATH)
    tables = document["tables"]
    prose = document.get("prose", {})
    destination = Path(args.output_dir) if args.output_dir else GENERATED_DIR
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (destination / f"{name}.typ").write_text(_table_fragment(name, table), encoding="utf-8")
    values = ["// Generated result values; do not edit by hand."]
    ppml_rows = {
        _clean_cell(row[0]): row
        for row in tables["ppml"]["rows"]
    }
    ppml_simple = ppml_rows["simple (well-connected)"]
    ppml_difficult = ppml_rows["difficult (near-nested)"]
    ols_difficult = tables["ols"]["rows"][1]
    correia_real_rows = {
        _clean_cell(row[0]): row
        for row in tables["correia_real"]["rows"]
    }
    enron = correia_real_rows["enron"]
    agreement_rows = tables["agreement"]["rows"]
    memory_rows = tables["memory"]["rows"]

    def memory_overheads(rows: list[list[str]]) -> list[float]:
        values = []
        for row in rows:
            map_memory, within_memory = _numeric_cell(row[2]), _numeric_cell(row[3])
            if map_memory is not None and within_memory is not None:
                values.append(within_memory - map_memory)
        return values

    def gap_without_share(value: str) -> str:
        return re.sub(r"\s+\([^)]*\)\s*$", "", value)

    def seconds_range(row: list[str], columns: range) -> str:
        candidates = [
            value
            for column in columns
            if (value := _numeric_cell(row[column])) is not None
        ]
        if not candidates:
            return "--"
        lower = _format_seconds(min(candidates)).removesuffix("s")
        upper = _format_seconds(max(candidates))
        return f"{lower}--{upper}"

    memory_100k = memory_overheads(memory_rows[1:3])
    memory_1m = memory_overheads(memory_rows[4:6])
    directors_share = _component_share(tables["correia_real"]["rows"][-1][1])
    prose_values = {
        "result_ols_difficult_gap": gap_without_share(ols_difficult[1]),
        "result_ols_difficult_rust_map": ols_difficult[2],
        "result_ols_difficult_fixest": ols_difficult[3],
        "result_ols_difficult_fem": ols_difficult[4],
        "result_ols_difficult_within": ols_difficult[5],
        "result_correia_enron_fem": enron[4],
        "result_correia_enron_within": enron[5],
        "result_ppml_simple_range": seconds_range(ppml_simple, range(2, 6)),
        "result_ppml_difficult_three_fixest": ppml_difficult[3],
        "result_ppml_difficult_three_glfem": ppml_difficult[4],
        "result_ppml_difficult_three_within": ppml_difficult[5],
        "result_agreement_simple_gap": gap_without_share(memory_rows[1][1]),
        "result_agreement_difficult_gap": gap_without_share(memory_rows[2][1]),
        "result_agreement_simple_max": _largest_metric(agreement_rows[:4], 3),
        "result_agreement_difficult_max": _largest_metric(agreement_rows[4:], 3),
        "result_setup_simple_setup": _format_seconds(float(prose["setup_simple_setup_s"])),
        "result_setup_simple_solve": _format_seconds(float(prose["setup_simple_solve_s"])),
        "result_setup_simple_share": f"{float(prose['setup_simple_share']):.0%}",
        "result_setup_difficult_setup": _format_seconds(float(prose["setup_difficult_setup_s"])),
        "result_setup_difficult_solve": _format_seconds(float(prose["setup_difficult_solve_s"])),
        "result_setup_difficult_share": f"{float(prose['setup_difficult_share']):.0%}",
        # The abstract quotes a magnitude, so it has to move with the run
        # rather than be typed in once and go stale.
        "result_ols_difficult_within_vs_rust_map": _format_ratio(
            _numeric_cell(ols_difficult[2]), _numeric_cell(ols_difficult[5])
        ),
        "result_ols_difficult_within_vs_fixest": _format_ratio(
            _numeric_cell(ols_difficult[3]), _numeric_cell(ols_difficult[5])
        ),
        "result_ppml_within_vs_fixest": _format_ratio(_numeric_cell(ppml_difficult[3]), _numeric_cell(ppml_difficult[5])),
        "result_memory_100k_overhead": f"{min(memory_100k):.0f}--{max(memory_100k):.0f} MiB" if memory_100k else "--",
        "result_memory_1m_overhead": f"{min(memory_1m):.0f}--{max(memory_1m):.0f} MiB" if memory_1m else "--",
        "result_directors_component_share": (
            f"{directors_share:.0%}" if directors_share is not None else "--"
        ),
        "result_zigzag_within": _format_seconds(float(prose["zigzag_within_s"])),
        "result_zigzag_fem": _format_seconds(float(prose["zigzag_fem_s"])),
    }
    # Values synchronized straight into the result file, such as the legacy CUDA
    # timings of Appendix C, are published without further formatting.
    prose_values.update(
        {name: value for name, value in prose.items() if name.startswith("result_")}
    )
    values.extend(f"#let {name} = [{_prose_cell(str(value))}]" for name, value in prose_values.items())
    (destination.parent / "paper_values.typ").write_text("\n".join(values) + "\n", encoding="utf-8")
    print(f"[render] wrote {len(tables)} table fragments to {destination}")


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


def _is_git_tracked(path: Path) -> bool:
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False
    return True


def archive_legacy_results(_: argparse.Namespace) -> None:
    """Archive untracked generated files without touching input caches."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_root = ROOT / "results" / "legacy" / stamp
    sources = (
        ROOT / "benchmarks" / "results",
        ROOT / "figures" / "benchmarks",
        ROOT / "results" / "runs",
    )
    moved = 0
    skipped: list[Path] = []
    for source in sources:
        if not source.exists():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if _is_git_tracked(path):
                skipped.append(path)
                continue
            destination = archive_root / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            moved += 1
    if moved:
        print(f"[archive] moved {moved} generated files to {archive_root}")
    else:
        print("[archive] no untracked generated files found")
    if skipped:
        print("[archive] left tracked files in place:")
        for path in skipped:
            print(f"  {path.relative_to(ROOT)}")


def _rows_from_csvs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pattern in ("benchmarks/results/*.csv",):
        for path in ROOT.glob(pattern):
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    rows.extend(
                        {
                            **row,
                            "_source_file": str(path.relative_to(ROOT)),
                        }
                        for row in csv.DictReader(handle)
                    )
            except (OSError, UnicodeDecodeError):
                continue
    return rows


def _backend_name(value: str) -> str | None:
    text = value.lower()
    # Matched-accuracy arms are their own series and must never be folded into
    # the package-default cell they share a solver with. Both views are
    # measured in one sweep (PROTOCOL.md section 6), so without this the four
    # within variants would collapse onto "within", the two MAP variants onto
    # "rust-map", and every affected cell would render as "incomplete" from
    # duplicate trial ids.
    for preconditioner in ("off", "diagonal", "additive"):
        if f"within-{preconditioner}" in text:
            return f"within-{preconditioner}"
    if "matched" in text:
        return "rust-map-matched"
    if "within" in text or "rust-cg" in text or "rust_cg" in text:
        return "within"
    if "rust-map" in text or "rust_map" in text or text in {"rust", "pyfixest-map", "pyfixest_map"}:
        return "rust-map"
    if "torch-cuda" in text:
        return "torch-cuda"
    if "glfixed" in text or "glfem" in text:
        return "GLFEM.jl"
    if "fixedeffectmodels" in text or "fem.jl" in text:
        return "FEM.jl"
    if "fixest" in text:
        return "fixest"
    return None


def _format_seconds(value: float) -> str:
    if value < 1:
        return f"{value:.3f}s"
    if value < 10:
        return f"{value:.2f}s"
    return f"{value:.1f}s"


def _row_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "True")).strip().lower() in {"true", "1"}


def _cell_is_complete(candidates: list[dict[str, str]]) -> bool:
    """Whether a cell holds every trial it was supposed to.

    A cell used to be three trials, full stop. With the R1/R2/R3 rule it is one
    group of repetitions per DGP replicate, and the repetition count is chosen
    per replicate from its own runtime, so the plan is a property of the
    replicate rather than of the cell. Checking the recorded plan against the
    pooled total would call a perfectly good cell incomplete: three replicates
    of seven repetitions is twenty-one rows against a plan of seven.

    Rows carrying no plan are pre-rule results, where one row per replicate and
    three replicates is the whole cell.
    """
    by_replicate: dict[int | None, list[dict[str, str]]] = {}
    for row in candidates:
        by_replicate.setdefault(_integer_field(row, "iter_num"), []).append(row)

    if not any(_integer_field(row, "n_planned") for row in candidates):
        return len(candidates) == EXPECTED_TRIALS

    if len(by_replicate) != EXPECTED_TRIALS:
        return False
    for rows in by_replicate.values():
        planned = {_integer_field(row, "n_planned") for row in rows}
        planned.discard(None)
        # One replicate runs under one plan, and must deliver exactly it.
        if len(planned) != 1 or len(rows) != planned.pop():
            return False
    return True


def _trial_key(row: dict[str, str]) -> tuple[int | None, int]:
    """Identify a trial by DGP replicate and timing repetition.

    Repetitions on one fixed sample and replicates of the DGP are different
    things (PROTOCOL.md section 2), so they are separate coordinates. Rows
    without a repetition are pre-rule results and count as repetition zero.
    """
    repetition = _integer_field(row, "repetition")
    return (_integer_field(row, "iter_num"), 0 if repetition is None else repetition)


def _render_trial_result(candidates: list[dict[str, str]]) -> str:
    """Format the benchmark trials for one cell, including failed trials."""
    if not candidates:
        return "#miss"

    summarized = [row for row in candidates if row.get("n_runs", "")]
    if summarized:
        if len(candidates) != 1:
            return "incomplete"
        row = summarized[0]
        total = _integer_field(row, "n_runs")
        successful = _integer_field(row, "n_success")
        values = []
        try:
            if row.get("time", ""):
                values.append(float(row["time"]))
        except (TypeError, ValueError):
            return "incomplete"
    else:
        trial_ids = [_trial_key(row) for row in candidates]
        if any(replicate is None for replicate, _ in trial_ids):
            return "incomplete"
        if len(set(trial_ids)) != len(trial_ids):
            return "incomplete"
        total = len(candidates)
        successful_rows = [row for row in candidates if _row_success(row)]
        successful = len(successful_rows)
        values = []
        for row in successful_rows:
            try:
                values.append(float(row["time"]))
            except (KeyError, TypeError, ValueError):
                return "incomplete"

    if successful is None or not 0 <= successful <= total:
        return "incomplete"
    if not summarized and not _cell_is_complete(candidates):
        return "incomplete"
    summary = summarize_times(values, n_attempted=total)
    if successful == 0:
        return f"failed (0/{total})"
    if summary.median_s is None:
        return "incomplete"
    rendered = _format_seconds(summary.median_s)
    if successful < total:
        return f"{rendered} ({successful}/{total})"
    return rendered


def _runtime_dataset(row: dict[str, str]) -> str | None:
    return row.get("dgp") or row.get("source_dataset_id") or row.get("dataset")


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
            if "fepois_bench__" in row.get("_source_file", "")
            and _integer_field(row, "n_fe") != 3
        }
    )
    if unexpected:
        raise ValueError(
            "PPML result files must contain only n_fe=3 rows; found "
            f"{', '.join(unexpected)}. Archive old results and rerun without "
            "--reuse-existing."
        )


def _paper_runtime_target(
    table_name: str, row: list[str]
) -> tuple[str, dict[str, int], str]:
    """Return the exact run specification represented by a paper timing cell."""
    dataset = _clean_cell(row[0]).split(" ")[0]
    if table_name in {"ols", "ppml"}:
        dataset = dataset.split("(")[0]
    if table_name in {
        "akm_mobility",
        "akm_sorting",
        "mechanism_mobility",
        "mechanism_sorting",
    }:
        return dataset, {"n_obs": 1_000_000, "model_k": 1, "n_fe": 3}, "feols_akm_sweep__"
    if table_name == "ols":
        return dataset, {"n_obs": 10_000_000, "model_k": 1, "n_fe": 3}, "feols_bench__"
    if table_name == "ppml":
        return dataset, {"n_obs": 1_000_000, "model_k": 1, "n_fe": int(row[1])}, "fepois_bench__"
    return dataset, {"model_k": 1, "n_fe": 2}, "correia-benchmarks.csv"


def _matches_runtime_target(
    row: dict[str, str],
    dataset: str,
    backend: str,
    requirements: dict[str, int],
    source_marker: str,
) -> bool:
    if source_marker not in row.get("_source_file", ""):
        return False
    if _runtime_dataset(row) != dataset:
        return False
    if _backend_name(row.get("backend") or row.get("algo") or "") != backend:
        return False
    return all(_integer_field(row, field) == value for field, value in requirements.items())


def _numeric_cell(value: str) -> float | None:
    if "failed" in value.lower():
        # A failed cell such as "failed (0/3)" carries no runtime; do not let the
        # "0" in the trial count read back as a 0.0-second measurement.
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
    rows: list[list[str]], column: int, *, backend: str | None = None
) -> str:
    """The largest numeric cell in one column, optionally within one backend.

    Cells that do not parse as a number are skipped rather than read as zero,
    so a "--" or a "#miss" marker cannot win the comparison and be reported as
    a measured worst case.
    """
    candidates = [
        row[column]
        for row in rows
        if (backend is None or _clean_cell(row[1]) == backend)
        and _numeric_cell(row[column]) is not None
    ]
    if not candidates:
        return "--"
    return max(candidates, key=lambda value: _numeric_cell(value) or 0.0)


def _component_share(value: str) -> float | None:
    match = re.search(r"\((0(?:\.\d+)?|1(?:\.0+)?)\)\s*$", value)
    return float(match.group(1)) if match else None


def _format_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return "--"
    ratio = numerator / denominator
    return f"{ratio:.0f} times" if ratio >= 10 else f"{ratio:.1f} times"


def _format_typst_scientific(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(f"{value:.0e}".split("e")[1])
    mantissa = value / (10**exponent)
    return f"${mantissa:.1f} times 10^({exponent})$"


def _clean_cell(value: str) -> str:
    return value.replace("`", "")


def _prose_cell(value: str) -> str:
    """Replace failure markers before inserting a value into Typst text."""
    return "--" if value in {"#miss", "failed", "--"} else value


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
    rows = _latest_rows("hardness.csv") or []
    diagnostics = {
        row["dataset_id"]: row
        for row in rows
        if {row["fe_a"], row["fe_b"]} in ({"indiv_id", "firm_id"}, {"id1", "id2"})
    }

    def update(table_name: str, source_id: str, target_row: list[str]) -> int:
        diagnostic = diagnostics.get(source_id)
        if diagnostic is None:
            if target_row[1] != "#miss":
                target_row[1] = "#miss"
                return 1
            return 0
        rendered = _format_hardness(
            float(diagnostic["one_minus_rho"]),
            float(diagnostic["worst_component_obs_share"]),
        )
        if target_row[1] == rendered:
            return 0
        target_row[1] = rendered
        return 1

    changed = 0
    for table_name in (
        "akm_mobility",
        "akm_sorting",
        "mechanism_mobility",
        "mechanism_sorting",
    ):
        for row in document["tables"][table_name]["rows"]:
            scenario = _clean_cell(row[0])
            changed += update(table_name, f"{scenario}_1000000_k1_iter_1", row)
    for row in document["tables"]["ols"]["rows"]:
        family = row[0].split()[0]
        changed += update("ols", f"{family}_10000000_k1_iter_1", row)
    for table_name in ("correia_synthetic", "correia_real"):
        for row in document["tables"][table_name]["rows"]:
            changed += update(table_name, _clean_cell(row[0]), row)
    for index, row in enumerate(document["tables"]["memory"]["rows"]):
        if row[0].startswith("#"):
            continue
        family = row[0].split()[0]
        size = "100000" if index < 4 else "1000000"
        changed += update("memory", f"memory_{family}_{size}", row)
    return changed


def _synchronize_agreement(document: dict) -> int:
    observations = _latest_rows("agreement.csv")
    if observations is None:
        return 0
    by_key = {
        (row["dgp"], _backend_name(row["backend"])): row
        for row in observations
        if _backend_name(row["backend"]) and _integer_field(row, "model_k") == 1
    }
    changed = 0
    dgp = ""
    for row in document["tables"]["agreement"]["rows"]:
        if row[0] == "#agreement-simple":
            dgp = "simple"
        elif row[0] == "#agreement-difficult":
            dgp = "difficult"
        backend = _clean_cell(row[1])
        source = by_key.get((dgp, backend))
        if source is None:
            replacement = ["#miss", "#miss"]
        elif source.get("success", "").lower() != "true":
            replacement = ["failed", "failed"]
        else:
            replacement = [
                f"{float(source['x1']):.8f}",
                "--" if backend == "rust-map" else _format_typst_scientific(float(source["max_abs_diff"])),
            ]
        for index, value in enumerate(replacement, start=2):
            if row[index] != value:
                row[index] = value
                changed += 1
    return changed


def _synchronize_setup_cost(document: dict) -> int:
    rows = _latest_rows("within_setup_cost_summary.csv")
    if rows is None:
        return 0
    summary = {row["dgp"]: row for row in rows}
    prose = document.setdefault("prose", {})
    changed = 0
    for dgp in ("simple", "difficult"):
        row = summary.get(dgp)
        if row is None or _integer_field(row, "k") != 1:
            raise ValueError(f"Missing one-covariate setup summary for {dgp}")
        if _integer_field(row, "n_runs") != EXPECTED_TRIALS:
            raise ValueError(f"Setup summary for {dgp} does not contain {EXPECTED_TRIALS} runs")
        if row.get("all_converged_reused", "").lower() != "true" or row.get(
            "all_converged_oneshot", ""
        ).lower() != "true":
            raise ValueError(f"Setup benchmark did not converge for {dgp}")
        fields = {
            f"setup_{dgp}_setup_s": float(row["median_setup_wall_s"]),
            f"setup_{dgp}_solve_s": float(row["median_solve_after_setup_wall_s"]),
            f"setup_{dgp}_share": float(row["median_setup_share_of_reused_total"]),
        }
        for key, value in fields.items():
            if prose.get(key) != value:
                prose[key] = value
                changed += 1
    return changed


def _synchronize_accuracy_frontier(document: dict) -> int:
    """Fill the frontier table from the tolerance sweep.

    Each package is swept over its own tolerance settings and the achieved
    external residual is recorded beside the wall time, so the reader sees the
    accuracy each runtime bought instead of a single matched point that some
    packages cannot reach.
    """
    measured = _latest_rows("accuracy_frontier.csv")
    table = document["tables"].get("accuracy_frontier")
    if table is None or measured is None:
        return 0
    rows = [row for row in measured if _row_success(row)]

    labels = {
        "pyfixest-rust-map": "`rust-map`",
        "pyfixest-within-additive": "`within-additive`",
    }
    rendered: list[list[str]] = []
    for design in ("simple", "difficult"):
        for package, label in labels.items():
            matching = [
                row
                for row in rows
                if row.get("dgp") == design and row.get("package") == package
            ]
            for index, row in enumerate(
                sorted(matching, key=lambda r: float(r["max_eta"]), reverse=True)
            ):
                eta = float(row["max_eta"])
                rendered.append(
                    [
                        f"{design}" if index == 0 and label == list(labels.values())[0] else "",
                        label if index == 0 else "",
                        row["setting"].split("=")[-1],
                        _format_seconds(float(row["time_s"])),
                        _format_typst_scientific(eta),
                    ]
                )
    if rendered == table["rows"]:
        return 0
    table["rows"] = rendered
    return len(rendered)


def _synchronize_ppml_inner_outer(document: dict) -> int:
    """Fill the PPML table separating outer IRLS steps from inner LSMR work.

    A common outer iteration cap is not a common accuracy condition, so the
    table reports whether the outer loop converged rather than only how many
    steps it took.
    """
    rows = _latest_rows("ppml_inner_outer.csv")
    table = document["tables"].get("ppml_inner_outer")
    if table is None or rows is None:
        return 0

    rendered: list[list[str]] = []
    for design in ("simple", "difficult"):
        first = True
        for row in [r for r in rows if r.get("dgp") == design]:
            converged = str(row.get("outer_converged", "")).lower() in {"true", "1"}
            steps = row.get("outer_iterations", "")
            rendered.append(
                [
                    design if first else "",
                    f"`{row['preconditioner']}`",
                    "yes" if str(row.get("rebuild_each_step", "")).lower() in {"true", "1"} else "no",
                    steps if converged else f"{steps} (capped)",
                    f"{int(float(row['inner_iterations_sum'])):,}".replace(",", "#h(0.18em)"),
                    _format_seconds(float(row["total_s"])),
                ]
            )
            first = False
    if rendered == table["rows"]:
        return 0
    table["rows"] = rendered
    return len(rendered)


def _synchronize_factor_scaling(document: dict) -> int:
    """Fill the table of setup and solve cost as the factor count grows.

    Q enters the construction through the Q(Q-1)/2 pair enumeration, so this is
    the axis on which the Schwarz preconditioner is most exposed, and every
    other experiment in the paper holds it at three.
    """
    rows = _latest_rows("factor_scaling.csv")
    table = document["tables"].get("factor_scaling")
    if table is None or rows is None:
        return 0

    by_q: dict[int, list[dict]] = {}
    for row in rows:
        by_q.setdefault(int(row["n_factors"]), []).append(row)

    rendered = []
    for n_factors in sorted(by_q):
        group = by_q[n_factors]

        def med(field: str) -> float:
            return float(median(float(item[field]) for item in group))

        rendered.append(
            [
                str(n_factors),
                str(n_factors * (n_factors - 1) // 2),
                f"{med('setup_s'):.3f}",
                f"{med('solve_s'):.3f}",
                f"{med('setup_share'):.0%}",
                f"{med('iterations_max'):.0f}",
            ]
        )
    if rendered == table["rows"]:
        return 0
    table["rows"] = rendered
    return len(rendered)


def _synchronize_amortization(document: dict) -> int:
    """Fill the table of total time against the number of right-hand sides.

    The break-even is read off the measurements rather than from a closed form,
    because marginal solve cost is not exactly linear in K.
    """
    rows = _latest_rows("amortization.csv")
    table = document["tables"].get("amortization")
    if table is None or rows is None:
        return 0

    def med(k: int, name: str, field: str) -> float | None:
        picked = [
            float(item[field])
            for item in rows
            if int(item["k_rhs"]) == k and item["preconditioner"] == name
        ]
        return float(median(picked)) if picked else None

    rendered = []
    for k in sorted({int(item["k_rhs"]) for item in rows}):
        diagonal, additive = med(k, "diagonal", "total_s"), med(k, "additive", "total_s")
        if diagonal is None or additive is None:
            continue
        rendered.append(
            [
                str(k),
                f"{diagonal:.2f}",
                f"{additive:.2f}",
                f"{diagonal / additive:.1f}x",
                f"{additive / k:.3f}",
            ]
        )
    if rendered == table["rows"]:
        return 0
    table["rows"] = rendered
    return len(rendered)


ITERATION_COLUMNS = (
    ("map", "map-sweep"),
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
            trials = [r for r in rows if r["design"] == design and r["solver_label"] == label]
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
            capped = any(r.get("censoring") == "capped" for r in trials)
            value = f"{median(counts):.0f}"
            cells.append(f"{value} (capped)" if capped else value)
        rendered.append(cells)
    return rendered


def _synchronize_zigzag(document: dict) -> int:
    """Store the synthetic-zigzag within/FEM.jl medians used in the manuscript.

    Read both times directly from the raw benchmark output.
    """
    path = ROOT / "benchmarks" / "results" / "correia-benchmarks.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == "synthetic-zigzag"
        ]
    times: dict[str, float] = {}
    for row in rows:
        backend = _backend_name(row.get("algo") or "")
        if backend in {"within", "FEM.jl"} and str(row.get("success", "")).lower() == "true":
            try:
                times[backend] = float(row["time"])
            except (KeyError, TypeError, ValueError):
                continue
    prose = document.setdefault("prose", {})
    changed = 0
    for backend, key in (("within", "zigzag_within_s"), ("FEM.jl", "zigzag_fem_s")):
        value = times.get(backend)
        if value is not None and prose.get(key) != value:
            prose[key] = value
            changed += 1
    return changed


def _reject_source_collision(
    table: str, dataset: str, backend: str, candidates: list[dict[str, str]]
) -> None:
    """Fail loudly when two result files claim the same cell.

    The harness writes one file per backend label, so this can only happen if a
    file was renamed without relabelling the `backend` column inside it. The
    renderer would otherwise report the cell as "incomplete" from duplicate
    trial ids, which does not say why.
    """
    per_backend_sources = {
        row.get("_source_file", "")
        for row in candidates
        if "__" in row.get("_source_file", "")
    }
    if len(per_backend_sources) > 1:
        listed = ", ".join(sorted(per_backend_sources))
        raise ValueError(
            f"{table}/{dataset}/{backend} draws on more than one per-backend "
            f"result file: {listed}. Two files carry the same `backend` label; "
            "relabel or remove one."
        )


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
    changed = 0
    for name, table in document["tables"].items():
        if name in {"memory", "agreement"}:
            continue
        headers = [_clean_cell(cell) for cell in table["header"]]
        for row in table["rows"]:
            dataset, requirements, source_marker = _paper_runtime_target(name, row)
            for column, backend in enumerate(headers[2:], start=2):
                candidates = [
                    source
                    for source in raw
                    if _matches_runtime_target(
                        source, dataset, backend, requirements, source_marker
                    )
                ]
                _reject_source_collision(name, dataset, backend, candidates)
                rendered = _render_trial_result(candidates)
                if row[column] != rendered:
                    row[column] = rendered
                    changed += 1
    measurements = _latest_rows("memory.csv")
    if measurements is not None:
        table = document["tables"]["memory"]
        for index, row in enumerate(table["rows"]):
            if row[0].startswith("#"):
                continue
            dgp = row[0].split()[0]
            size = "100k" if index < 4 else "1m"
            for column, backend in ((2, "rust"), (3, "rust-cg")):
                candidates = [
                    item
                    for item in measurements
                    if item["dgp"] == dgp
                    and item["size"] == size
                    and item["backend"] == backend
                    and _integer_field(item, "model_k") == 1
                ]
                match = next(
                    (
                        item
                        for item in candidates
                        if item["success"].lower() == "true"
                    ),
                    None,
                )
                if match and match["rss_mb"]:
                    rendered = f"{int(float(match['rss_mb'])):,} MiB"
                elif candidates:
                    rendered = "failed"
                else:
                    rendered = "#miss"
                if row[column] != rendered:
                    row[column] = rendered
                    changed += 1
    changed += _synchronize_hardness(document)
    changed += _synchronize_agreement(document)
    changed += _synchronize_setup_cost(document)
    changed += _synchronize_accuracy_frontier(document)
    changed += _synchronize_ppml_inner_outer(document)
    changed += _synchronize_iterations(document)
    changed += _synchronize_factor_scaling(document)
    changed += _synchronize_amortization(document)
    changed += _synchronize_zigzag(document)
    if write:
        _write_json(TABLES_PATH, document)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-external-runtimes").set_defaults(func=check_external_runtimes)
    sub.add_parser("setup-julia-env").set_defaults(func=setup_julia_env)
    fetch = sub.add_parser("fetch-correia")
    fetch.add_argument("--datasets", nargs="*", help="Dataset metadata IDs to fetch (default: all)")
    fetch.add_argument("--offline", action="store_true", help="Validate local CSVs without network access")
    fetch.set_defaults(func=fetch_correia)
    sub.add_parser("collect").set_defaults(func=collect)
    sub.add_parser("archive-legacy-results").set_defaults(func=archive_legacy_results)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--output-dir", type=Path)
    render_parser.set_defaults(func=render)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
