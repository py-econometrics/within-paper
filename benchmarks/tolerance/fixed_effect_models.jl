#!/usr/bin/env julia

using CSV
using DataFrames
using FixedEffectModels
using LinearAlgebra
using Parquet2
using StatsModels

data_path, output_path, design, tolerance_text, default_text,
    repetitions_text, maxiter_text, reference_beta_text, reference_se_text = ARGS
tolerances = parse.(Float64, split(tolerance_text, ","))
default_tolerance = parse(Float64, default_text)
repetitions = parse(Int, repetitions_text)
maxiter = parse(Int, maxiter_text)
reference_beta = parse(Float64, reference_beta_text)
reference_se = parse(Float64, reference_se_text)
threads = parse(Int, ENV["BENCH_THREADS"])
Threads.nthreads() == threads || error("Julia thread count does not match BENCH_THREADS")
frame = DataFrame(Parquet2.Dataset(data_path))
reference_residual = Float64.(frame.reference_residual)
reference_norm = norm(reference_residual)
formula = @formula(y ~ x1 + fe(indiv_id) + fe(firm_id) + fe(year))
fit_once(tolerance) = reg(
    frame, formula, Vcov.simple(); save=:residuals, tol=tolerance,
    maxiter=maxiter, nthreads=threads, double_precision=true, progress_bar=false,
)
fit_once(tolerances[1])
rows = NamedTuple[]
for tolerance in tolerances, repetition in 0:(repetitions - 1)
    local trial_started = time_ns()
    try
        fit = fit_once(tolerance)
        fit.converged || error("FixedEffectModels returned without convergence")
        beta = Float64(coef(fit)[1])
        residual = Float64.(residuals(fit))
        push!(rows, (
            design=design, backend="FEM.jl", tolerance=tolerance,
            default_tolerance=default_tolerance, repetition=repetition,
            runtime_s=(time_ns() - trial_started) / 1e9, beta_x1=beta,
            coefficient_error_se=abs(beta - reference_beta) / reference_se,
            residual_error=norm(residual - reference_residual) / reference_norm,
            converged=true, capped=false, error="",
        ))
    catch error_value
        message = sprint(showerror, error_value)
        push!(rows, (
            design=design, backend="FEM.jl", tolerance=tolerance,
            default_tolerance=default_tolerance, repetition=repetition,
            runtime_s=(time_ns() - trial_started) / 1e9, beta_x1=missing,
            coefficient_error_se=missing, residual_error=missing,
            converged=false, capped=occursin(r"iter|converg|maxiter"i, message),
            error=message,
        ))
    end
end
CSV.write(output_path, DataFrame(rows))
