#!/usr/bin/env python3
"""Manage reproducible benchmark results for the paper.

This script uses only the Python standard library, so runtime checks can run before the
Pixi environment is available. Paper table data is stored in
``results/paper/benchmark_tables.json``; ``render`` writes the Typst includes, and
``collect`` records raw outputs and runtime information.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import time
import urllib.request
import zipfile
from statistics import median
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The timing rules live with the benchmark harness that produces the numbers,
# not beside the renderer that formats them, so that "median over converged
# trials, failures kept in the denominator" has one implementation rather than
# one per consumer. The module is standard-library only, which is what lets
# this script keep running before the Pixi environment exists; a test pins
# that property.
from benchmarks.modular.timing import summarize_times
from scripts.figure_style import METHOD_TABLE_HEADER
TABLES_PATH = ROOT / "results" / "paper" / "benchmark_tables.json"
CLAIMS_PATH = ROOT / "results" / "paper" / "claim_registry.json"
GENERATED_DIR = ROOT / "generated" / "tables"
RUNTIME_CONFIG = ROOT / "config" / "external_runtimes.json"
EXTERNAL_RESULTS_PATH = ROOT / "results" / "external" / "cuda.json"
CORREIA_DIR = ROOT / "data" / "correia_data"
EXPECTED_TRIALS = 3
# The legacy CUDA timings are quoted only in Appendix C, never in a main
# runtime table, so they resolve to generated prose values rather than to a
# table cell. They moved out of the OLS table when the appendix was written.
EXPECTED_EXTERNAL_CUDA_TARGETS = {
    ("simple", "torch-cuda"),
    ("difficult", "torch-cuda"),
}
RAW_GLOBS = (
    "benchmarks/results/*.csv",
    "results/runs/latest/*.csv",
    # The calibration pilot and the pooled gap analysis write JSON, not CSV.
    # Without these the run that froze Gate A and the diagnostic behind the
    # spectral-gap caveat would carry no provenance.
    "results/runs/latest/*.json",
    "figures/results/*.svg",
)
CODE_GLOBS = (
    "pixi.toml",
    "pixi.lock",
    "scripts/*.py",
    "benchmarks/**/*.py",
    "benchmarks/**/*.R",
    "benchmarks/**/*.jl",
    "benchmarks/julia-env/Project.toml",
    "benchmarks/julia-env/Manifest.toml",
    "config/*.json",
    "data/correia_data/metadata/*.json",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def _code_fingerprint() -> str:
    """Hash the code, locks, and metadata that define a benchmark run."""
    paths: set[Path] = set()
    for pattern in CODE_GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _git_dirty() -> bool:
    try:
        return bool(_run(["git", "status", "--porcelain", "--untracked-files=all"]))
    except (OSError, subprocess.CalledProcessError):
        return True


def _module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return spec.origin if spec else None


def _runtime_provenance() -> dict:
    packages = {}
    for name in ("pyfixest", "within-py", "pyarrow", "numpy", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "python_packages": packages,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "bench_threads": os.environ.get("BENCH_THREADS"),
        "julia_num_threads": os.environ.get("JULIA_NUM_THREADS"),
        "within_repo": os.environ.get("WITHIN_REPO"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "code_sha256": _code_fingerprint(),
        "module_origins": {
            "pyfixest": _module_origin("pyfixest"),
            "within": _module_origin("within"),
        },
        "r_version": _optional_version(["Rscript", "--version"]),
        "julia_version": _optional_version(["julia", "--version"]),
    }


def _git_commit() -> str | None:
    try:
        return _run(["git", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        return None


def _optional_version(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        return _run(command)
    except subprocess.CalledProcessError:
        return None


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
    ols_simple = tables["ols"]["rows"][0]
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

    def largest_metric(rows: list[list[str]], column: int) -> str:
        candidates = [
            row[column]
            for row in rows
            if _numeric_cell(row[column]) is not None
        ]
        if not candidates:
            return "--"
        return max(candidates, key=lambda value: _numeric_cell(value) or 0.0)

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
        "result_akm_mobility_first_gap": tables["akm_mobility"]["rows"][0][1],
        # The limitations section names the regime where the method loses, so
        # those numbers have to move with the run like every other quoted value.
        **_scaling_prose_values(),
        **_ppml_reuse_prose_values(),
        **_gap_prose_values(),
        "result_ols_simple_within": ols_simple[5],
        "result_ols_simple_rust_map": ols_simple[2],
        "result_ols_simple_best_alternative": _format_seconds(
            min(
                value
                for value in (_numeric_cell(ols_simple[i]) for i in (2, 3, 4))
                if value is not None
            )
        ),
        "result_ols_difficult_gap": gap_without_share(ols_difficult[1]),
        "result_ols_difficult_rust_map": ols_difficult[2],
        "result_ols_difficult_fixest": ols_difficult[3],
        "result_ols_difficult_fem": ols_difficult[4],
        "result_ols_difficult_within": ols_difficult[5],
        "result_correia_uniform_harder_gap": tables["correia_synthetic"]["rows"][3][1],
        "result_correia_enron_fem": enron[4],
        "result_correia_enron_within": enron[5],
        "result_ppml_simple_range": seconds_range(ppml_simple, range(2, 6)),
        "result_ppml_difficult_three_fixest": ppml_difficult[3],
        "result_ppml_difficult_three_glfem": ppml_difficult[4],
        "result_ppml_difficult_three_within": ppml_difficult[5],
        "result_agreement_simple_gap": gap_without_share(memory_rows[1][1]),
        "result_agreement_difficult_gap": gap_without_share(memory_rows[2][1]),
        "result_agreement_simple_max": largest_metric(agreement_rows[:4], 3),
        "result_agreement_difficult_max": largest_metric(agreement_rows[4:], 3),
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
        "result_ppml_within_vs_glfem": _format_ratio(_numeric_cell(ppml_difficult[4]), _numeric_cell(ppml_difficult[5])),
        "result_memory_100k_overhead": f"{min(memory_100k):.0f}--{max(memory_100k):.0f} MiB" if memory_100k else "--",
        "result_memory_1m_overhead": f"{min(memory_1m):.0f}--{max(memory_1m):.0f} MiB" if memory_1m else "--",
        "result_directors_component_share": (
            f"{directors_share:.0%}" if directors_share is not None else "--"
        ),
        "result_zigzag_within": _format_seconds(float(prose["zigzag_within_s"])),
        "result_zigzag_fem": _format_seconds(float(prose["zigzag_fem_s"])),
        "result_zigzag_speedup": _format_ratio(
            float(prose["zigzag_fem_s"]), float(prose["zigzag_within_s"])
        ),
    }
    # Values synchronized straight into the result file, such as the legacy CUDA
    # timings of Appendix C, are published without further formatting.
    prose_values.update(
        {name: value for name, value in prose.items() if name.startswith("result_")}
    )
    values.extend(f"#let {name} = [{_prose_cell(str(value))}]" for name, value in prose_values.items())
    (destination.parent / "paper_values.typ").write_text("\n".join(values) + "\n", encoding="utf-8")
    print(f"[render] wrote {len(tables)} table fragments to {destination}")


def collect(args: argparse.Namespace) -> None:
    run_dir = ROOT / "results" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_provenance()
    if runtime["within_repo"]:
        raise SystemExit(
            "Refusing to collect paper results with WITHIN_REPO set; "
            "the paper run must use the locked within-py package"
        )
    if runtime["git_dirty"]:
        raise SystemExit(
            "Refusing to collect paper results from a dirty tracked worktree; "
            "commit benchmark code and documentation first"
        )
    updated = _synchronize_canonical_tables()
    artifacts = []
    for pattern in RAW_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_json(run_dir / "provenance.json", {"runtime": runtime, "artifacts": artifacts})
    print(
        f"[collect] recorded {len(artifacts)} raw result files in {run_dir}; "
        f"updated {updated} paper table cells"
    )


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


def _largest_backend_metric(
    rows: list[list[str]], backend: str, column: int
) -> str:
    candidates = [
        row[column]
        for row in rows
        if _clean_cell(row[1]) == backend and _numeric_cell(row[column]) is not None
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
    path = ROOT / "results" / "runs" / "latest" / "hardness.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
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
    path = ROOT / "results" / "runs" / "latest" / "agreement.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        observations = list(csv.DictReader(handle))
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
    path = ROOT / "results" / "runs" / "latest" / "within_setup_cost_summary.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        summary = {row["dgp"]: row for row in csv.DictReader(handle)}
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


def _scaling_prose_values() -> dict[str, str]:
    """Quoted numbers from the factor-scaling and amortization sweeps."""
    values: dict[str, str] = {}
    runs = ROOT / "results" / "runs" / "latest"

    scaling = runs / "factor_scaling.csv"
    if scaling.exists():
        with scaling.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_q: dict[int, list[dict]] = {}
        for row in rows:
            by_q.setdefault(int(row["n_factors"]), []).append(row)
        if 2 in by_q and 5 in by_q:
            def med(q: int, field: str) -> float:
                return float(median(float(r[field]) for r in by_q[q]))

            values["result_qscale_setup_ratio"] = _format_ratio(med(5, "setup_s"), med(2, "setup_s"))
            values["result_qscale_solve_ratio"] = _format_ratio(med(5, "solve_s"), med(2, "solve_s"))
            values["result_qscale_setup_share_q5"] = f"{med(5, 'setup_share'):.0%}"
            values["result_qscale_iters_q2"] = f"{med(2, 'iterations_max'):.0f}"
            values["result_qscale_iters_q5"] = f"{med(5, 'iterations_max'):.0f}"

    amort = runs / "amortization.csv"
    if amort.exists():
        with amort.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        def total(k: int, name: str) -> float | None:
            picked = [
                float(r["total_s"])
                for r in rows
                if int(r["k_rhs"]) == k and r["preconditioner"] == name
            ]
            return float(median(picked)) if picked else None

        for k in (1, 25):
            diagonal, additive = total(k, "diagonal"), total(k, "additive")
            if diagonal and additive:
                values[f"result_amortize_ratio_k{k}"] = _format_ratio(diagonal, additive)
    return values


def _gap_prose_values() -> dict[str, str]:
    """Publish the worker-firm gap of the difficult design at each measured size.

    Connectivity is size-dependent, so any claim about the difficult design has
    to name its sample size. Quoting these from one place keeps the sizes in the
    prose tied to the hardness table.
    """
    path = ROOT / "results" / "runs" / "latest" / "hardness.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = {100_000: "100k", 1_000_000: "1m", 10_000_000: "10m"}
    values: dict[str, str] = {}
    for row in rows:
        if (
            "difficult" not in row["dataset_id"]
            or row["fe_a"] != "indiv_id"
            or row["fe_b"] != "firm_id"
        ):
            continue
        label = labels.get(int(row["n_obs"]))
        if label is None:
            continue
        values[f"result_gap_difficult_{label}"] = _format_typst_scientific(
            float(row["one_minus_rho"])
        )
    return values


def _ppml_reuse_prose_values() -> dict[str, str]:
    """Quoted numbers from the IRLS preconditioner-reuse experiment."""
    path = ROOT / "results" / "runs" / "latest" / "ppml_inner_outer.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    values: dict[str, str] = {}
    for design in ("simple", "difficult"):
        group = [row for row in rows if row.get("dgp") == design]
        rebuilt = next(
            (r for r in group if str(r["rebuild_each_step"]).lower() in {"true", "1"}),
            None,
        )
        reused = next(
            (
                r
                for r in group
                if r["preconditioner"] == "additive"
                and str(r["rebuild_each_step"]).lower() not in {"true", "1"}
            ),
            None,
        )
        if rebuilt is None or reused is None:
            continue
        values[f"result_ppml_reuse_speedup_{design}"] = _format_ratio(
            float(reused["total_s"]), float(rebuilt["total_s"])
        )
        values[f"result_ppml_rebuild_outer_{design}"] = str(
            int(float(rebuilt["outer_iterations"]))
        )
        values[f"result_ppml_rebuild_inner_{design}"] = f"{int(float(rebuilt['inner_iterations_sum'])):,}"
        values[f"result_ppml_rebuild_total_{design}"] = _format_seconds(
            float(rebuilt["total_s"])
        )
        deviations = [
            abs(float(r["deviance"]) / float(rebuilt["deviance"]) - 1.0)
            for r in group
            if r is not rebuilt
        ]
        if deviations:
            values[f"result_ppml_reuse_deviance_gap_{design}"] = (
                _format_typst_scientific(max(deviations))
            )
    return values


def _synchronize_accuracy_frontier(document: dict) -> int:
    """Fill the frontier table from the tolerance sweep.

    Each package is swept over its own tolerance settings and the achieved
    external residual is recorded beside the wall time, so the reader sees the
    accuracy each runtime bought instead of a single matched point that some
    packages cannot reach.
    """
    path = ROOT / "results" / "runs" / "latest" / "accuracy_frontier.csv"
    table = document["tables"].get("accuracy_frontier")
    if table is None or not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if _row_success(row)]

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
    path = ROOT / "results" / "runs" / "latest" / "ppml_inner_outer.csv"
    table = document["tables"].get("ppml_inner_outer")
    if table is None or not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

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
    path = ROOT / "results" / "runs" / "latest" / "factor_scaling.csv"
    table = document["tables"].get("factor_scaling")
    if table is None or not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

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
    path = ROOT / "results" / "runs" / "latest" / "amortization.csv"
    table = document["tables"].get("amortization")
    if table is None or not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

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
    path = ROOT / "results" / "runs" / "latest" / "within_preconditioners.csv"
    table = document["tables"].get("iterations")
    if table is None or not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rendered = _iteration_rows(list(csv.DictReader(handle)))
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


def _validate_external_results(external: dict) -> list[dict]:
    required_metadata = {
        "schema_version": 3,
        "source": "legacy PyFixest benchmark suite",
        "status": "indicative_only",
        "exact_run_provenance_available": False,
        "cross_machine_comparison_allowed": False,
        "local_reproducible": False,
    }
    for field, expected in required_metadata.items():
        actual = external.get(field)
        matches = actual is expected if isinstance(expected, bool) else actual == expected
        if not matches:
            raise ValueError(
                f"External CUDA metadata {field!r} must be {expected!r}"
            )
    provenance_note = external.get("provenance_note")
    if not isinstance(provenance_note, str) or not provenance_note.strip():
        raise ValueError("External CUDA metadata requires a nonempty provenance_note")

    measurements = external.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("External CUDA measurements must be a list")

    targets = []
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise ValueError("Each external CUDA measurement must be an object")
        target = tuple(measurement.get(field) for field in ("design", "backend"))
        if not all(isinstance(value, str) for value in target):
            raise ValueError("External CUDA targets must use string identifiers")
        targets.append(target)
        time_s = measurement.get("time_s")
        if (
            isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(time_s)
            or time_s <= 0
        ):
            raise ValueError(
                f"External CUDA timing must be finite and positive: {time_s!r}"
            )

    if len(targets) != len(set(targets)):
        raise ValueError("External CUDA measurements contain duplicate targets")
    if set(targets) != EXPECTED_EXTERNAL_CUDA_TARGETS:
        raise ValueError(
            "External CUDA measurements must contain exactly the simple and difficult "
            "torch-cuda designs"
        )
    return measurements


def _synchronize_external_results(document: dict) -> int:
    """Publish the legacy CUDA timings as prose values for Appendix C.

    They are deliberately not written into a runtime table. The run recorded
    neither hardware nor package versions, so placing it in a column next to
    reproducible timings would invite exactly the comparison the appendix says
    cannot be made.
    """
    measurements = _validate_external_results(_read_json(EXTERNAL_RESULTS_PATH))
    prose = document.setdefault("prose", {})
    changed = 0
    for measurement in measurements:
        key = f"result_cuda_{measurement['design']}"
        rendered = _format_seconds(float(measurement["time_s"]))
        if prose.get(key) != rendered:
            prose[key] = rendered
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
    memory_path = ROOT / "results" / "runs" / "latest" / "memory.csv"
    if memory_path.exists():
        with memory_path.open(newline="", encoding="utf-8") as handle:
            measurements = list(csv.DictReader(handle))
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
    changed += _synchronize_external_results(document)
    if write:
        _write_json(TABLES_PATH, document)
    return changed


def verify(_: argparse.Namespace) -> None:
    document = _read_json(TABLES_PATH)
    tables = document["tables"]
    registry = _read_json(CLAIMS_PATH)
    claims = registry["claims"]
    claimed = {claim["table"] for claim in claims}
    missing = sorted(set(tables) - claimed)
    if missing:
        raise SystemExit(f"Missing claim registry entries: {', '.join(missing)}")

    provenance_path = ROOT / "results" / "runs" / "latest" / "provenance.json"
    if not provenance_path.exists():
        raise SystemExit(f"Missing benchmark provenance: {provenance_path}")
    provenance = _read_json(provenance_path)
    runtime = provenance.get("runtime", {})
    required_runtime = (
        "git_commit",
        "git_dirty",
        "code_sha256",
        "bench_threads",
        "julia_num_threads",
        "r_version",
        "julia_version",
        "module_origins",
    )
    missing_runtime = [name for name in required_runtime if runtime.get(name) is None]
    if missing_runtime:
        raise SystemExit("Incomplete benchmark provenance: " + ", ".join(missing_runtime))
    if runtime["git_dirty"]:
        raise SystemExit("Benchmark provenance records a dirty tracked worktree")
    current_code_hash = _code_fingerprint()
    if runtime["code_sha256"] != current_code_hash:
        raise SystemExit(
            "Benchmark code fingerprint differs from provenance: "
            f"expected {runtime['code_sha256']}, found {current_code_hash}"
        )

    artifacts = {
        artifact.get("path", ""): artifact
        for artifact in provenance.get("artifacts", [])
        if artifact.get("path")
    }
    artifact_errors: list[str] = []
    for relative, artifact in artifacts.items():
        path = ROOT / relative
        if not path.is_file():
            artifact_errors.append(f"missing {relative}")
        elif _sha256(path) != artifact.get("sha256"):
            artifact_errors.append(f"hash mismatch {relative}")
    external_sources = set(registry.get("external_sources", []))
    source_errors: list[str] = []
    for claim in claims:
        for pattern in claim.get("sources", []):
            matches = sorted(path for path in ROOT.glob(pattern) if path.is_file())
            if not matches:
                source_errors.append(f"{claim['id']}: no files match {pattern}")
                continue
            for path in matches:
                relative = path.relative_to(ROOT).as_posix()
                if relative not in external_sources and relative not in artifacts:
                    source_errors.append(f"{claim['id']}: {relative} absent from provenance")
    if artifact_errors or source_errors:
        raise SystemExit(
            "Invalid result provenance:\n- " + "\n- ".join(artifact_errors + source_errors)
        )

    expected_document = copy.deepcopy(document)
    try:
        _synchronize_canonical_tables(expected_document, write=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Cannot reconstruct paper tables from raw results: {exc}") from exc
    if expected_document != document:
        raise SystemExit(
            "Paper table values do not match the raw results. Run "
            "`pixi run render-paper-results` after collecting the benchmark results."
        )

    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        render(argparse.Namespace(output_dir=temp_root / "tables"))
        for name in tables:
            expected = (temp_root / "tables" / f"{name}.typ").read_text(encoding="utf-8")
            actual_path = GENERATED_DIR / f"{name}.typ"
            if not actual_path.exists() or actual_path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"Generated table is stale: {actual_path}")
        expected_values = (temp_root / "paper_values.typ").read_text(encoding="utf-8")
        actual_values = GENERATED_DIR.parent / "paper_values.typ"
        if not actual_values.exists() or actual_values.read_text(encoding="utf-8") != expected_values:
            raise SystemExit(f"Generated paper values are stale: {actual_values}")
    manuscript = (ROOT / "graph_preconditioner_hdfe.typ").read_text(encoding="utf-8")
    required_includes = [f'generated/tables/{name}.typ' for name in tables]
    absent = [item for item in required_includes if item not in manuscript]
    if absent:
        raise SystemExit("Manuscript is missing generated table includes: " + ", ".join(absent))
    incomplete = []
    for table_name, table in tables.items():
        for row_number, row in enumerate(table["rows"], start=1):
            for column, cell in enumerate(row, start=1):
                if cell in {"#miss", "incomplete"}:
                    incomplete.append(f"{table_name}[{row_number},{column}]={cell}")
    if incomplete:
        raise SystemExit("Required paper table cells are missing or incomplete: " + ", ".join(incomplete))
    print("[verify] raw results, hashes, code, generated tables, and manuscript includes are current")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-external-runtimes").set_defaults(func=check_external_runtimes)
    sub.add_parser("setup-julia-env").set_defaults(func=setup_julia_env)
    fetch = sub.add_parser("fetch-correia")
    fetch.add_argument("--datasets", nargs="*", help="Dataset metadata IDs to fetch (default: all)")
    fetch.add_argument("--offline", action="store_true", help="Validate local CSVs without network access")
    fetch.set_defaults(func=fetch_correia)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--run-id", default="latest")
    collect_parser.set_defaults(func=collect)
    sub.add_parser("archive-legacy-results").set_defaults(func=archive_legacy_results)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--output-dir", type=Path)
    render_parser.set_defaults(func=render)
    sub.add_parser("verify").set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
