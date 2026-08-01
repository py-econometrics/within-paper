"""Running a benchmark backend in another language.

The R and Julia backends are driven the same way: write the manifest and the
model spec to a JSON config, run the interpreter on a driver script, and read
one JSON record per fit back. Long runs also append to a result log, so a
process that dies before flushing stdout still reports the fits it finished.

This lives apart from the in-process PyFixest backend because the two share
nothing but the result type, and because the tolerance benchmark needs the
handshake without the rest of the benchmark layer.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.core.interfaces import BenchmarkDataset, FeolsResult, FeolsSpec
from benchmarks.core.runtime import configure_benchmark_runtime
from benchmarks.solvers.common import (
    as_bool,
    normalize_vcov,
    result_from_dataset,
    safe_cast,
)

# ---------------------------------------------------------------------------


def _parse_subprocess_output(
    *,
    datasets: list[BenchmarkDataset],
    spec: FeolsSpec,
    backend: str,
    completed_process: subprocess.CompletedProcess[str],
) -> list[FeolsResult]:
    # Index strictly by (dataset_id, iter_num). Both the feols/fepois and the
    # Correia manifests emit one JSON line per trial with its iteration number,
    # so a crashed run leaves the missing trials unmatched (and therefore failed)
    # rather than reusing another trial's timing for a trial that never ran.
    parsed_by_key: dict[tuple[str, int | None], dict] = {}

    for line in completed_process.stdout.splitlines():
        payload = line.strip()
        if not payload:
            continue
        try:
            entry = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        dataset_id = entry.get("dataset_id")
        if isinstance(dataset_id, str):
            iter_num = safe_cast(entry.get("iter_num"), int)
            key = (dataset_id, iter_num)
            previous = parsed_by_key.get(key)
            if previous is not None and previous != entry:
                raise ValueError(
                    "Subprocess emitted conflicting records for "
                    f"dataset_id={dataset_id!r}, iter_num={iter_num!r}"
                )
            parsed_by_key[key] = entry

    # Warn when a successful subprocess omits one or more datasets.
    n_emitted = len(parsed_by_key)
    n_missing = len(datasets) - n_emitted
    if n_missing > 0 and completed_process.returncode == 0:
        warnings.warn(
            f"Subprocess returned results for {n_emitted}/{len(datasets)} datasets"
        )

    stderr_text = (completed_process.stderr or "").strip()
    # Store only the final 4,000 characters of stderr in each failed CSV row.
    if len(stderr_text) > 4_000:
        stderr_text = stderr_text[-4_000:]
    if completed_process.returncode != 0:
        default_error = f"Subprocess exited with code {completed_process.returncode}"
        if stderr_text:
            default_error = f"{default_error}: {stderr_text}"
    else:
        default_error = stderr_text or None
    results: list[FeolsResult] = []

    for dataset in datasets:
        entry = parsed_by_key.get((dataset.dataset_id, dataset.iter_num))
        if entry is None:
            missing_error = default_error or "The subprocess returned no result for this dataset."
            results.append(
                result_from_dataset(
                    dataset,
                    spec,
                    backend=backend,
                    elapsed=None,
                    success=False,
                    error=missing_error,
                )
            )
            continue

        elapsed = safe_cast(entry.get("time"), float)
        n_obs_override = safe_cast(entry.get("n_obs"), int)

        results.append(
            result_from_dataset(
                dataset,
                spec,
                backend=backend,
                elapsed=elapsed,
                success=as_bool(entry.get("success"), default=elapsed is not None),
                error=entry.get("error"),
                n_obs_override=n_obs_override,
                **_diagnostics_from_entry(entry),
            )
        )

    return results


def _diagnostics_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Keep optional diagnostics an external driver records in the common result."""
    integer_fields = (
        "n_retained",
        "outer_iterations",
        "inner_iterations_sum",
        "inner_iterations_max",
    )
    float_fields = (
        "preconditioner_build_s",
        "deviance",
        "loglik",
        "max_eta",
        "beta_x1",
    )
    diagnostics: dict[str, Any] = {
        field: value
        for field in integer_fields
        if (value := safe_cast(entry.get(field), int)) is not None
    }
    diagnostics.update(
        {
            field: value
            for field in float_fields
            if (value := safe_cast(entry.get(field), float)) is not None
        }
    )
    censoring = entry.get("censoring")
    if isinstance(censoring, str):
        diagnostics["censoring"] = censoring
    return diagnostics


class SubprocessFeolsBenchmarker:
    """Generic subprocess backend for feols (R/Julia)."""

    def __init__(
        self,
        *,
        name: str,
        command_prefix: Sequence[str],
        script_path: Path,
        model: str = "feols",
    ):
        self._name = name
        self._command_prefix = tuple(command_prefix)
        self._script_path = script_path.resolve()
        # Selects the fit family inside a driver that serves more than one.
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    def cache_key(self) -> dict[str, Any]:
        """Settings that make one subprocess result file reusable."""
        return {
            "adapter": "subprocess",
            "name": self.name,
            "command_prefix": self._command_prefix,
            "script_path": str(self._script_path),
            "model": self._model,
        }

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        configure_benchmark_runtime()
        manifest = [
            {
                "dataset_id": dataset.dataset_id,
                "data_path": str(dataset.data_path.resolve()),
                "dgp": dataset.dgp,
                "n_obs": dataset.n_obs,
                "iter_type": dataset.iter_type,
                "iter_num": dataset.iter_num,
            }
            for dataset in datasets
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            result_log_path = Path(tmpdir) / "results.jsonl"
            config_path.write_text(
                json.dumps(
                    {
                        "manifest": manifest,
                        "formula": spec.formula,
                        "depvar": spec.depvar,
                        "covariates": list(spec.covariates),
                        "fe_cols": list(spec.fe_cols),
                        "vcov": spec.vcov,
                        "vcov_type": normalize_vcov(spec.vcov),
                        "model": self._model,
                        # Julia PPML can run for several minutes. Write each result
                        # to a file in case the process exits before flushing stdout.
                        "result_log_path": str(result_log_path),
                    }
                ),
                encoding="utf-8",
            )

            command = [
                *self._command_prefix,
                str(self._script_path),
                str(config_path),
            ]

            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout_text, stderr_text = proc.communicate()
                stdout_lines = stdout_text.splitlines(keepends=True)
                stderr_lines = stderr_text.splitlines(keepends=True)
                if stderr_text:
                    sys.stderr.write(stderr_text)
                    sys.stderr.flush()
                if result_log_path.exists():
                    stdout_lines.extend(
                        result_log_path.read_text(encoding="utf-8").splitlines(keepends=True)
                    )
            except Exception as exc:
                return [
                    result_from_dataset(
                        dataset,
                        spec,
                        backend=self.name,
                        elapsed=None,
                        success=False,
                        error=str(exc),
                    )
                    for dataset in datasets
                ]

        completed = subprocess.CompletedProcess(
            args=command,
            returncode=proc.returncode,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )
        return _parse_subprocess_output(
            datasets=datasets,
            spec=spec,
            backend=self.name,
            completed_process=completed,
        )
