"""Download the Correia collection and verify every recorded checksum."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORREIA_DIR = ROOT / "benchmarks" / "data" / "correia_data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(datasets: list[str], offline: bool) -> None:
    selected = set(datasets)
    for metadata_path in sorted((CORREIA_DIR / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        slug = metadata["slug"]
        if selected and slug not in selected:
            continue
        destination = CORREIA_DIR / f"{slug}.csv"
        if destination.exists() and _sha256(destination) == metadata["checksum_csv"]:
            print(f"[ok] {slug}: existing CSV checksum verified")
            continue
        if offline:
            raise SystemExit(f"Missing or invalid {destination}; --offline forbids download")

        archive = CORREIA_DIR / ".downloads" / f"{slug}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {slug}: {metadata['package_url']}")
        urllib.request.urlretrieve(metadata["package_url"], archive)
        if archive.stat().st_size != metadata["package_bytes"] or _sha256(archive) != metadata["package_sha256"]:
            archive.unlink(missing_ok=True)
            raise SystemExit(f"Checksum or size mismatch for {slug} package")

        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise SystemExit(f"Expected one CSV in {archive}, found {members}")
            with bundle.open(members[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        if destination.stat().st_size != metadata["csv_bytes"] or _sha256(destination) != metadata["checksum_csv"]:
            destination.unlink(missing_ok=True)
            raise SystemExit(f"CSV checksum or size mismatch after extracting {slug}")
        print(f"[ok] {slug}: verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    fetch(args.datasets, args.offline)
