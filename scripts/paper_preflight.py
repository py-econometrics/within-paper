"""Environment preflight and benchmark-data acquisition.

These commands run before the long benchmark suite: they check the external R
and Julia runtimes, install the tracked Julia project, and download and verify
the Correia HDFE datasets. None of them touch the paper table pipeline, so they
live apart from the table synchronizers in ``paper_results``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tomllib
import urllib.request
import zipfile

from scripts.paper_io import ROOT, _read_json, _run, _sha256

CORREIA_DIR = ROOT / "benchmarks" / "data" / "correia_data"
RUNTIME_CONFIG = ROOT / "config" / "external_runtimes.json"


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
