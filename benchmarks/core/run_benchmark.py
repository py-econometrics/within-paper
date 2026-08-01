"""Run one benchmark module after applying the shared runtime configuration."""

from __future__ import annotations

import argparse
import runpy
import sys

from benchmarks.core.runtime import configure_benchmark_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="Python module containing a benchmark entry point")
    args, remaining = parser.parse_known_args()
    configure_benchmark_runtime()
    sys.argv = [args.module, *remaining]
    runpy.run_module(args.module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
