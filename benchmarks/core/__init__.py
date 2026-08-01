"""Primitives with no knowledge of a solver, a DGP or an experiment.

Everything here is importable from anywhere else in the package. Nothing here
imports from `benchmarks.dgp`, `benchmarks.solvers` or `benchmarks.drivers`; a
test enforces that, so the layering cannot quietly invert.
"""
