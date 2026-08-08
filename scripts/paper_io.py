"""Low-level filesystem and process helpers shared by the paper scripts.

These have no benchmark-specific knowledge: JSON round-tripping, checksums, and
running a subprocess. Keeping them in one dependency-free module lets both the
environment preflight and the table pipeline import them without a cycle.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]


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


def _run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(
        command, text=True, stderr=subprocess.STDOUT, env=env
    ).strip()


def _git_dirty() -> bool:
    try:
        return bool(_run(["git", "status", "--porcelain", "--untracked-files=all"]))
    except (OSError, subprocess.CalledProcessError):
        return True
