from __future__ import annotations

from dataclasses import dataclass

from feols_benchmarkers import (
    FixestFeolsBenchmarker,
    JuliaFeolsBenchmarker,
    PyFeolsBenchmarkerFullApi,
    detect_torch_runtime_availability,
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
    """Build the shared feols benchmark runner set used by modular benchmarks."""
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
                "[bench] skipping torch benchmarkers: torch is not installed",
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
                    "[bench] skipping torch-mps benchmarker: MPS unavailable",
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
                    "[bench] skipping torch-cuda benchmarker: CUDA unavailable",
                    flush=True,
                )

    benchmarkers = list(pyfixest_benchmarkers)
    if include_fixest:
        benchmarkers.append(FixestFeolsBenchmarker("fixest-map"))
    if include_julia:
        benchmarkers.append(JuliaFeolsBenchmarker("FEM.jl (lsmr)"))

    if not benchmarkers:
        raise ValueError(
            "No benchmarkers available after applying include flags and runtime "
            "availability checks."
        )

    return BenchmarkerBundle(benchmarkers=benchmarkers)
