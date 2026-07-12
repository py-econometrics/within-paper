#!/usr/bin/env python3
"""Reproducible result management for the paper.

This module deliberately uses only the Python standard library so it can be run
both inside the Pixi environment and as a preflight helper before a full run.
Canonical table values live in ``results/paper/benchmark_tables.json``; render
turns them into the Typst fragments included by the manuscript. ``collect``
records the raw output files and execution environment for a benchmark run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
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
TABLES_PATH = ROOT / "results" / "paper" / "benchmark_tables.json"
CLAIMS_PATH = ROOT / "results" / "paper" / "claim_registry.json"
GENERATED_DIR = ROOT / "generated" / "tables"
RUNTIME_CONFIG = ROOT / "config" / "external_runtimes.json"
EXTERNAL_RESULTS_PATH = ROOT / "results" / "external" / "cuda.json"
CORREIA_DIR = ROOT / "data" / "correia_data"
EXPECTED_TRIALS = 3
RAW_GLOBS = (
    "benchmarks/results/*.csv",
    "results/runs/latest/*.csv",
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
                            f"Julia thread check: JULIA_NUM_THREADS={julia_threads}, "
                            f"but Julia started with {configured_threads.strip()} thread(s)"
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
    """Fetch each metadata-described zip and verify its source checksum."""
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
        "  table.header(" + ", ".join(f"th[{cell}]" for cell in table["header"]) + "),",
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
            # design cell, so subsequent rows contain only four cells.
            row = row[1:]
        cells = [cell if cell else "" for cell in row]
        rendered_cells = [
            cell if index == 0 and cell.startswith("table.cell(") else f"[{cell}]"
            for index, cell in enumerate(cells)
        ]
        lines.append("  " + ", ".join(rendered_cells) + ",")
    lines.extend(["  table.hline(stroke: 0.8pt + table-rule),", ")", ""])
    return "\n".join(lines)


def render(args: argparse.Namespace) -> None:
    document = _read_json(TABLES_PATH)
    tables = document["tables"]
    prose = document.get("prose", {})
    destination = Path(args.output_dir) if args.output_dir else GENERATED_DIR
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (destination / f"{name}.typ").write_text(_table_fragment(name, table), encoding="utf-8")
    values = ["// Generated result values; do not edit by hand."]
    for name in tables:
        values.append(f'#let paper_{name}_source = "results/paper/benchmark_tables.json"')
    ols_difficult = tables["ols"]["rows"][1]
    ppml_difficult = tables["ppml"]["rows"][3]
    memory_rows = tables["memory"]["rows"]

    def memory_overheads(rows: list[list[str]]) -> list[float]:
        values = []
        for row in rows:
            map_memory, within_memory = _numeric_cell(row[2]), _numeric_cell(row[3])
            if map_memory is not None and within_memory is not None:
                values.append(within_memory - map_memory)
        return values

    memory_100k = memory_overheads(memory_rows[1:3])
    memory_1m = memory_overheads(memory_rows[4:6])
    directors_share = _component_share(tables["correia_real"]["rows"][-1][1])
    prose_values = {
        "result_akm_mobility_first_gap": tables["akm_mobility"]["rows"][0][1],
        "result_ols_difficult_within": tables["ols"]["rows"][1][5],
        "result_ols_difficult_gpu": tables["ols"]["rows"][1][6],
        "result_ols_difficult_rust_map": tables["ols"]["rows"][1][2],
        "result_correia_uniform_harder_gap": tables["correia_synthetic"]["rows"][3][1],
        "result_ppml_simple_three_map": tables["ppml"]["rows"][2][3],
        "result_ppml_simple_three_glfem": tables["ppml"]["rows"][2][4],
        "result_ppml_simple_three_within": tables["ppml"]["rows"][2][5],
        "result_ppml_difficult_three_fixest": tables["ppml"]["rows"][3][2],
        "result_ppml_difficult_three_glfem": tables["ppml"]["rows"][3][4],
        "result_ppml_difficult_three_within": tables["ppml"]["rows"][3][5],
        "result_agreement_fixest_max": _largest_backend_metric(
            tables["agreement"]["rows"], "fixest", 4
        ),
        "result_setup_simple_setup": _format_seconds(float(prose["setup_simple_setup_s"])),
        "result_setup_simple_solve": _format_seconds(float(prose["setup_simple_solve_s"])),
        "result_setup_simple_share": f"{float(prose['setup_simple_share']):.0%}",
        "result_setup_difficult_setup": _format_seconds(float(prose["setup_difficult_setup_s"])),
        "result_setup_difficult_solve": _format_seconds(float(prose["setup_difficult_solve_s"])),
        "result_setup_difficult_share": f"{float(prose['setup_difficult_share']):.0%}",
        "result_ols_gpu_vs_fem": _format_ratio(_numeric_cell(ols_difficult[4]), _numeric_cell(ols_difficult[6])),
        "result_ppml_within_vs_fixest": _format_ratio(_numeric_cell(ppml_difficult[2]), _numeric_cell(ppml_difficult[5])),
        "result_ppml_within_vs_glfem": _format_ratio(_numeric_cell(ppml_difficult[4]), _numeric_cell(ppml_difficult[5])),
        "result_memory_100k_overhead": f"{min(memory_100k):.0f}--{max(memory_100k):.0f} MB" if memory_100k else "--",
        "result_memory_1m_overhead": f"{min(memory_1m):.0f}--{max(memory_1m):.0f} MB" if memory_1m else "--",
        "result_directors_component_share": (
            f"{directors_share:.0%}" if directors_share is not None else "--"
        ),
    }
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
        f"[collect] recorded {len(artifacts)} raw result artifacts in {run_dir}; "
        f"updated {updated} canonical timing cells"
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
    """Archive untracked generated artifacts without touching input caches."""
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
        print(f"[archive] moved {moved} generated artifacts to {archive_root}")
    else:
        print("[archive] no untracked generated artifacts found")
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
    if "within" in text or "rust-cg" in text:
        return "within"
    if "rust-map" in text or text in {"rust", "pyfixest-map"}:
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


def _render_trial_result(candidates: list[dict[str, str]]) -> str:
    """Render a complete three-trial result, including measured non-convergence."""
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
        trial_ids = [_integer_field(row, "iter_num") for row in candidates]
        if any(trial_id is None for trial_id in trial_ids) or len(set(trial_ids)) != len(trial_ids):
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

    if total != EXPECTED_TRIALS or successful is None or not 0 <= successful <= total:
        return "incomplete"
    if successful == 0:
        return f"failed (0/{total})"
    if not values:
        return "incomplete"
    rendered = _format_seconds(float(median(values)))
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


def _paper_runtime_target(
    table_name: str, row: list[str]
) -> tuple[str, dict[str, int], str]:
    """Return the exact run specification represented by a paper timing cell."""
    dataset = _clean_cell(row[0]).split(" ")[0]
    if table_name in {"ols", "ppml"}:
        dataset = dataset.split("(")[0]
    if table_name in {"akm_mobility", "akm_sorting"}:
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
    """Prevent failed table-cell markers from becoming invalid Typst prose."""
    return "--" if value in {"#miss", "failed", "--"} else value


def _format_hardness(gap: float, share: float) -> str:
    """Use compact Typst-compatible formatting for a gap and component share."""
    if gap and abs(gap) < 1e-2:
        exponent = int(f"{gap:.0e}".split("e")[1])
        mantissa = gap / (10**exponent)
        gap_text = f"${mantissa:.2f} times 10^({exponent})$"
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
    for table_name in ("akm_mobility", "akm_sorting"):
        for row in document["tables"][table_name]["rows"]:
            scenario = _clean_cell(row[0])
            changed += update(table_name, f"{scenario}_1000000_k1_iter_1", row)
    for row in document["tables"]["ols"]["rows"]:
        family = row[0].split()[0]
        changed += update("ols", f"{family}_1000000_k1_iter_1", row)
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
            replacement = ["#miss", "#miss", "#miss"]
        elif source.get("success", "").lower() != "true":
            replacement = ["failed", "failed", "failed"]
        else:
            replacement = [
                f"{float(source['x1']):.8f}",
                "--" if backend == "rust-map" else _format_typst_scientific(float(source["avg_abs_diff"])),
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


def _synchronize_external_results(document: dict) -> int:
    external = _read_json(EXTERNAL_RESULTS_PATH)
    changed = 0
    for measurement in external.get("measurements", []):
        table_name = measurement["table"]
        row_name = measurement["row"]
        backend = measurement["backend"]
        table = document["tables"][table_name]
        headers = [_clean_cell(cell) for cell in table["header"]]
        try:
            column = headers.index(backend)
            row = next(item for item in table["rows"] if item[0].split()[0] == row_name)
        except (ValueError, StopIteration) as exc:
            raise ValueError(f"External result target not found: {measurement}") from exc
        rendered = _format_seconds(float(measurement["time_s"]))
        if row[column] != rendered:
            row[column] = rendered
            changed += 1
    return changed


def _synchronize_canonical_tables(
    document: dict | None = None, *, write: bool = True
) -> int:
    """Update timing cells from current raw result CSVs, preserving diagnostics.

    Gap/component diagnostics are calculated in their dedicated diagnostic pipeline
    and intentionally live in the canonical table store. Runtime columns are
    replaced whenever a complete median is present in a new benchmark output.
    """
    raw = _rows_from_csvs()
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
                if backend == "torch-cuda":
                    continue
                rendered = _render_trial_result(candidates)
                if row[column] != rendered:
                    row[column] = rendered
                    changed += 1
    memory_path = ROOT / "results" / "runs" / "latest" / "memory.csv"
    if memory_path.exists():
        with memory_path.open(newline="", encoding="utf-8") as handle:
            measurements = list(csv.DictReader(handle))
        table = document["tables"]["memory"]
        for row in table["rows"]:
            if row[0].startswith("#"):
                continue
            dgp = row[0].split()[0]
            size = "100k" if table["rows"].index(row) < 4 else "1m"
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
                    rendered = f"{int(float(match['rss_mb'])):,} MB"
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
            "Canonical paper tables do not match the current raw results; "
            "run pixi run render-paper-results after collecting a clean run"
        )

    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        render(argparse.Namespace(output_dir=temp_root / "tables"))
        for name in tables:
            expected = (temp_root / "tables" / f"{name}.typ").read_text(encoding="utf-8")
            actual_path = GENERATED_DIR / f"{name}.typ"
            if not actual_path.exists() or actual_path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"Stale generated fragment: {actual_path}")
        expected_values = (temp_root / "paper_values.typ").read_text(encoding="utf-8")
        actual_values = GENERATED_DIR.parent / "paper_values.typ"
        if not actual_values.exists() or actual_values.read_text(encoding="utf-8") != expected_values:
            raise SystemExit(f"Stale generated prose values: {actual_values}")
    manuscript = (ROOT / "graph_preconditioner_hdfe.typ").read_text(encoding="utf-8")
    required_includes = [f'generated/tables/{name}.typ' for name in tables]
    absent = [item for item in required_includes if item not in manuscript]
    if absent:
        raise SystemExit("Manuscript is not wired to generated tables: " + ", ".join(absent))
    incomplete = []
    for table_name, table in tables.items():
        for row_number, row in enumerate(table["rows"], start=1):
            for column, cell in enumerate(row, start=1):
                if cell in {"#miss", "incomplete"}:
                    incomplete.append(f"{table_name}[{row_number},{column}]={cell}")
    if incomplete:
        raise SystemExit("Incomplete locally reproducible paper results: " + ", ".join(incomplete))
    print("[verify] raw results, hashes, code, generated fragments, and manuscript wiring are current")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-external-runtimes").set_defaults(func=check_external_runtimes)
    sub.add_parser("setup-julia-env").set_defaults(func=setup_julia_env)
    fetch = sub.add_parser("fetch-correia")
    fetch.add_argument("--datasets", nargs="*", help="Optional metadata slugs to fetch")
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
