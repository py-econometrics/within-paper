#!/usr/bin/env julia

using CSV
using DataFrames
using FixedEffectModels
using LinearAlgebra
using Logging
using Parquet2
using StatsModels

mutable struct ConvergenceLogger <: AbstractLogger
    parent::AbstractLogger
    capped::Bool
end

Logging.min_enabled_level(logger::ConvergenceLogger) = Logging.min_enabled_level(logger.parent)
Logging.shouldlog(logger::ConvergenceLogger, args...) = Logging.shouldlog(logger.parent, args...)
Logging.catch_exceptions(logger::ConvergenceLogger) = Logging.catch_exceptions(logger.parent)

function Logging.handle_message(
    logger::ConvergenceLogger, level, message, module_name, group, id, file, line;
    kwargs...,
)
    if occursin("Convergence of annihilation procedure not achieved", string(message))
        logger.capped = true
    end
    Logging.handle_message(
        logger.parent, level, message, module_name, group, id, file, line; kwargs...,
    )
end

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
function fit_once(tolerance)
    logger = ConvergenceLogger(current_logger(), false)
    fit = with_logger(logger) do
        reg(
            frame, formula, Vcov.simple(); save=:residuals, tol=tolerance,
            maxiter=maxiter, nthreads=threads, double_precision=true, progress_bar=false,
        )
    end
    logger.capped && error("FixedEffectModels reached the maximum number of iterations")
    fit.converged || error("FixedEffectModels returned without convergence")
    fit
end
try
    fit_once(tolerances[1])
catch
end
rows = NamedTuple[]
for tolerance in tolerances, repetition in 0:(repetitions - 1)
    local trial_started = time_ns()
    try
        fit = fit_once(tolerance)
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
            converged=false,
            capped=occursin(r"maximum.*iterations|maxiter|iteration cap"i, message),
            error=message,
        ))
    end
end
CSV.write(output_path, DataFrame(rows))
