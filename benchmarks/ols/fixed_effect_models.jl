#!/usr/bin/env julia

using CSV
using DataFrames
using FixedEffectModels
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

data_path, output_path, fixed_text, requested = ARGS[1:4]
specification = length(ARGS) >= 5 ? ARGS[5] : "intercepts"
threads = parse(Int, ENV["BENCH_THREADS"])
Threads.nthreads() == threads || error("Julia thread count does not match BENCH_THREADS")
frame = DataFrame(Parquet2.Dataset(data_path))
fixed_effects = split(fixed_text, ",")
formula = if specification == "intercepts"
    fixed_terms = foldl(+, [fe(Symbol(name)) for name in fixed_effects])
    term(:y) ~ term(:x1) + fixed_terms
elseif specification == "worker-firm-year-slopes"
    term(:y) ~ term(:x1) + fe(:indiv_id) +
        (fe(:indiv_id) & term(:year)) + fe(:firm_id) +
        (fe(:firm_id) & term(:year)) + fe(:year)
else
    error("unknown OLS specification $specification")
end

function fit_once()
    logger = ConvergenceLogger(current_logger(), false)
    fit = with_logger(logger) do
        reg(frame, formula, Vcov.simple(); nthreads=threads, progress_bar=false)
    end
    logger.capped && error("FixedEffectModels demeaning returned without convergence")
    fit
end

warm_started = time_ns()
try
    fit_once()
catch
end
warmup = (time_ns() - warm_started) / 1e9
repetitions = requested == "adaptive" ? (warmup < 1 ? 20 : warmup < 10 ? 7 : 3) : parse(Int, requested)
rows = NamedTuple[]
for repetition in 0:(repetitions - 1)
    local trial_started = time_ns()
    try
        fit = fit_once()
        push!(rows, (
            backend="FEM.jl", preconditioner="",
            package_version=string(Base.pkgversion(FixedEffectModels)),
            repetition=repetition,
            n_planned=repetitions,
            runtime_s=(time_ns() - trial_started) / 1e9, n_retained=nobs(fit),
            beta_x1=Float64(coef(fit)[1]),
            converged=true, capped=false, error="",
        ))
    catch error_value
        push!(rows, (
            backend="FEM.jl", preconditioner="",
            package_version=string(Base.pkgversion(FixedEffectModels)),
            repetition=repetition,
            n_planned=repetitions,
            runtime_s=(time_ns() - trial_started) / 1e9, n_retained=missing,
            beta_x1=missing, converged=false,
            capped=occursin("without convergence", sprint(showerror, error_value)),
            error=sprint(showerror, error_value),
        ))
    end
end
CSV.write(output_path, DataFrame(rows))
