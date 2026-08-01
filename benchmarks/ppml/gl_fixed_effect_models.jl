#!/usr/bin/env julia

using CSV
using DataFrames
using GLFixedEffectModels
using GLM
using Parquet2
using StatsModels

data_path, output_path, requested = ARGS
threads = parse(Int, ENV["BENCH_THREADS"])
Threads.nthreads() == threads || error("Julia thread count does not match BENCH_THREADS")
frame = DataFrame(Parquet2.Dataset(data_path))
formula = term(:negbin_y) ~ term(:x1) + fe(:indiv_id) + fe(:firm_id) + fe(:year)
fit_once() = nlreg(
    frame, formula, Poisson(), LogLink(), Vcov.simple();
    maxiter=100, nthreads=threads,
)
fit_once()
rows = NamedTuple[]
for repetition in 0:(parse(Int, requested) - 1)
    local trial_started = time_ns()
    try
        fit = fit_once()
        fit.converged || error("GLFixedEffectModels returned without convergence")
        push!(rows, (
            backend="GLFEM.jl", repetition=repetition,
            runtime_s=(time_ns() - trial_started) / 1e9, n_retained=nobs(fit),
            beta_x1=Float64(coef(fit)[1]),
            converged=true, error="",
        ))
    catch error_value
        push!(rows, (
            backend="GLFEM.jl", repetition=repetition,
            runtime_s=(time_ns() - trial_started) / 1e9, n_retained=missing,
            beta_x1=missing, converged=false,
            error=sprint(showerror, error_value),
        ))
    end
end
CSV.write(output_path, DataFrame(rows))
