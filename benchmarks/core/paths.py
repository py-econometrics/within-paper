"""Every filesystem location the project uses, defined once.

Each module used to derive the repository root itself, with
``Path(__file__).resolve().parents[N]`` and an N that depended on how deep the
file happened to sit. That made the directory layout part of two dozen files'
correctness: moving a module one level changed the meaning of its N, and the
symptom was a path that silently pointed at the wrong place rather than an
import error.

Standard library only, and no imports from the rest of the package.
``scripts/paper_results.py`` runs before the Pixi environment exists, so
anything it reaches has to work with a bare interpreter.
"""

from __future__ import annotations

from pathlib import Path

# benchmarks/core/paths.py -> benchmarks/core -> benchmarks -> repository root
ROOT = Path(__file__).resolve().parents[2]

# Generated parquet samples and the downloaded Correia collection.
DATA_DIR = ROOT / "benchmarks" / "data"
CORREIA_DIR = DATA_DIR / "correia_data"

# Raw per-backend sweep output. Untracked; the paper tables are built from it.
RESULTS_DIR = ROOT / "benchmarks" / "results"

# Where the standalone diagnostics write, and where paper_results reads.
LATEST_RUN = ROOT / "results" / "runs" / "latest"

FIGURES_DIR = ROOT / "figures" / "results"

# Pinned Julia depot for the external drivers.
JULIA_ENV = ROOT / "benchmarks" / "julia-env"

# The R and Julia benchmark drivers. They are executed as scripts, never
# imported, so they live outside the Python packages rather than beside the
# module that happens to launch them.
EXTERNAL_DIR = ROOT / "benchmarks" / "external"
