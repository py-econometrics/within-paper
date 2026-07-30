"""Writing measured rows to disk.

Every experiment ends by writing a CSV that scripts/paper_results.py later
reads. Three drivers had grown their own copy of that step, and they disagreed
on the details that matter when a schema changes: whether a row missing an
optional key is an error or a blank cell, and whether the header comes from the
first row or from all of them.

Taking the union of keys in first-seen order is the behaviour the pipeline
needs. A driver that records an optional diagnostic only when the backend
exposes it still produces one rectangular table, and adding a field does not
silently truncate the rows written before it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path


def write_rows(
    path: Path, rows: Sequence[dict], *, fieldnames: Iterable[str] | None = None
) -> None:
    """Write dict rows to ``path``, unioning keys so optional fields stay aligned.

    ``fieldnames`` pins the column order when a consumer depends on it. Omit it
    to take every key in first-seen order. Missing keys are written as blanks;
    an empty ``rows`` is an error, because a driver that measured nothing should
    say so rather than leave a header-only file that reads as a completed run.
    """
    if not rows:
        raise ValueError("no rows to write")
    if fieldnames is None:
        collected: list[str] = []
        for row in rows:
            for key in row:
                if key not in collected:
                    collected.append(key)
        fieldnames = collected
    else:
        fieldnames = list(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
