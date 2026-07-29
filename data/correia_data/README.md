# Correia benchmark data

The CSV files in this directory are not tracked by Git. From the repository root, download
and verify the full collection with:

```bash
pixi run fetch-correia
```

The download metadata and expected checksums are stored in `metadata/`. To verify an
existing local copy without downloading anything:

```bash
pixi run python scripts/paper_results.py fetch-correia --offline
```

The collection is described on [Sergio Correia's HDFE data page](https://scorreia.com/data/hdfe/index.html).
