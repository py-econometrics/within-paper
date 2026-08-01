#!/usr/bin/env julia

using CSV
using DataFrames
using FixedEffectModels
using Parquet2
using StatsModels

data_path, output_path, fixed_text, requested = ARGS
threads = parse(Int, ENV["BENCH_THREADS"])
Threads.nthreads() == threads || error("Julia thread count does not match BENCH_THREADS")
frame = DataFrame(Parquet2.Dataset(data_path))
fixed_effects = split(fixed_text, ",")
fixed_terms = foldl(+, [fe(Symbol(name)) for name in fixed_effects])
formula = term(:y) ~ term(:x1) + fixed_terms

function fit_once()
    reg(frame, formula, Vcov.simple(); nthreads=threads, progress_bar=false)
end

warm_started = time_ns()
fit_once()
warmup = (time_ns() - warm_started) / 1e9
repetitions = requested == "adaptive" ? (warmup < 1 ? 20 : warmup < 10 ? 7 : 3) : parse(Int, requested)
rows = NamedTuple[]
for repetition in 0:(repetitions - 1)
    local trial_started = time_ns()
    try
        fit = fit_once()
        push!(rows, (
            backend="FEM.jl", repetition=repetition,
            runtime_s=(time_ns() - trial_started) / 1e9, n_retained=nobs(fit),
            beta_x1=Float64(coef(fit)[1]), max_eta=missing,
            converged=true, error="",
        ))
    catch error_value
        push!(rows, (
            backend="FEM.jl", repetition=repetition,
            runtime_s=(time_ns() - trial_started) / 1e9, n_retained=missing,
            beta_x1=missing, max_eta=missing, converged=false,
            error=sprint(showerror, error_value),
        ))
    end
end
CSV.write(output_path, DataFrame(rows))
