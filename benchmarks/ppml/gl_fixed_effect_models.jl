#!/usr/bin/env julia

using CSV
using DataFrames
using GLFixedEffectModels
using GLM
using Logging
using Parquet2
using StatsModels

mutable struct ConvergenceLogger <: AbstractLogger
    parent::AbstractLogger
    inner_capped::Bool
end

Logging.min_enabled_level(logger::ConvergenceLogger) = Logging.min_enabled_level(logger.parent)
Logging.shouldlog(logger::ConvergenceLogger, args...) = Logging.shouldlog(logger.parent, args...)
Logging.catch_exceptions(logger::ConvergenceLogger) = Logging.catch_exceptions(logger.parent)

function Logging.handle_message(
    logger::ConvergenceLogger, level, message, module_name, group, id, file, line;
    kwargs...,
)
    if occursin("Convergence of annihilation procedure not achieved", string(message))
        logger.inner_capped = true
    end
    Logging.handle_message(
        logger.parent, level, message, module_name, group, id, file, line; kwargs...,
    )
end

data_path, output_path, requested, outer_maxiter_text = ARGS
outer_maxiter = parse(Int, outer_maxiter_text)
threads = parse(Int, ENV["BENCH_THREADS"])
Threads.nthreads() == threads || error("Julia thread count does not match BENCH_THREADS")
frame = DataFrame(Parquet2.Dataset(data_path))
formula = term(:negbin_y) ~ term(:x1) + fe(:indiv_id) + fe(:firm_id) + fe(:year)
function fit_once()
    logger = ConvergenceLogger(current_logger(), false)
    fit = with_logger(logger) do
        nlreg(
            frame, formula, Poisson(), LogLink(), Vcov.simple();
            maxiter=outer_maxiter, nthreads=threads,
        )
    end
    return fit, logger.inner_capped
end
try
    fit_once()
catch
end
rows = NamedTuple[]
for repetition in 0:(parse(Int, requested) - 1)
    local trial_started = time_ns()
    try
        fit, inner_capped = fit_once()
        inner_capped && error("GLFixedEffectModels inner demeaning returned without convergence")
        fit.converged || error("GLFixedEffectModels returned without convergence")
        push!(rows, (
            backend="GLFEM.jl", repetition=repetition,
            runtime_s=(time_ns() - trial_started) / 1e9, n_retained=nobs(fit),
            beta_x1=Float64(coef(fit)[1]),
            converged=true, capped=false, error="",
        ))
    catch error_value
        message = sprint(showerror, error_value)
        push!(rows, (
            backend="GLFEM.jl", repetition=repetition,
            runtime_s=(time_ns() - trial_started) / 1e9, n_retained=missing,
            beta_x1=missing, converged=false,
            capped=occursin("without convergence", message), error=message,
        ))
    end
end
CSV.write(output_path, DataFrame(rows))
