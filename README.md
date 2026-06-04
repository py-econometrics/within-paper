# within-paper

Source for the paper *A Fast Graph-Based Solver for Fixed-Effects Regressions* (Fischer & Schröder). The Typst source (`main.typ`), benchmarks, and figures used to compile `main.pdf`.

## Setup

Dependencies (Typst, R + `fixest`, Python + `pyfixest`) are managed with [pixi](https://pixi.sh):

```bash
pixi install
pixi run compile   # builds main.pdf from main.typ
```
