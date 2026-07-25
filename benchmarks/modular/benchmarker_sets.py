from __future__ import annotations

from dataclasses import dataclass

from feols_benchmarkers import (
    FixestFeolsBenchmarker,
    JuliaFeolsBenchmarker,
    PyFeolsBenchmarkerFullApi,
    detect_torch_runtime_availability,
)
from fepois_benchmarkers import (
    FixestFepoisBenchmarker,
    GLFixedEffectModelsBenchmarker,
    PyFepoisBenchmarkerFullApi,
)


@dataclass(frozen=True)
class BenchmarkerBundle:
    benchmarkers: list


def build_standard_feols_benchmarkers(
    *,
    include_pyfixest: bool = True,
    include_fixest: bool = True,
    include_julia: bool = True,
    include_torch: bool = True,
) -> BenchmarkerBundle:
    """Create the enabled feols benchmark backends."""
    pyfixest_benchmarkers = []
    if include_pyfixest:
        pyfixest_benchmarkers.extend(
            [
                PyFeolsBenchmarkerFullApi("pyfixest (within)", "within"),
                PyFeolsBenchmarkerFullApi("pyfixest (rust-map)", "rust"),
            ]
        )

    if include_torch:
        availability = detect_torch_runtime_availability()
        if not availability.has_torch:
            print(
                "[bench] skipping Torch backends: Torch is not installed",
                flush=True,
            )
        else:
            pyfixest_benchmarkers.append(
                PyFeolsBenchmarkerFullApi(
                    "pyfixest (torch-cpu)",
                    "torch_cpu",
                )
            )
            if availability.has_mps:
                pyfixest_benchmarkers.append(
                    PyFeolsBenchmarkerFullApi(
                        "pyfixest (torch-mps)",
                        "torch_mps",
                    )
                )
            else:
                print(
                    "[bench] skipping torch-mps: MPS unavailable",
                    flush=True,
                )

            if availability.has_cuda:
                pyfixest_benchmarkers.append(
                    PyFeolsBenchmarkerFullApi(
                        "pyfixest (torch-cuda)",
                        "torch_cuda",
                    )
                )
            else:
                print(
                    "[bench] skipping torch-cuda: CUDA unavailable",
                    flush=True,
                )

    benchmarkers = list(pyfixest_benchmarkers)
    if include_fixest:
        benchmarkers.append(FixestFeolsBenchmarker("fixest-map"))
    if include_julia:
        benchmarkers.append(JuliaFeolsBenchmarker("FEM.jl (lsmr)"))

    if not benchmarkers:
        raise ValueError("No requested benchmark backend is available.")

    return BenchmarkerBundle(benchmarkers=benchmarkers)


def build_standard_fepois_benchmarkers(
    *,
    include_pyfixest: bool = True,
    include_fixest: bool = True,
    include_julia: bool = True,
) -> BenchmarkerBundle:
    """Create the enabled fepois benchmark backends."""
    benchmarkers = []
    if include_pyfixest:
        benchmarkers.extend(
            [
                PyFepoisBenchmarkerFullApi(
                    "pyfixest (within)", "within", iwls_maxiter=100
                ),
                PyFepoisBenchmarkerFullApi(
                    "pyfixest (rust-map)", "rust", iwls_maxiter=100
                ),
            ]
        )
    if include_fixest:
        benchmarkers.append(FixestFepoisBenchmarker("fixest-fepois"))
    if include_julia:
        benchmarkers.append(GLFixedEffectModelsBenchmarker("glfixedeffectmodels.jl"))

    if not benchmarkers:
        raise ValueError("No requested benchmark backend is available.")

    return BenchmarkerBundle(benchmarkers=benchmarkers)
