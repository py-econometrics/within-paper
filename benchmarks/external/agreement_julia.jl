#!/usr/bin/env julia

using DataFrames
using FixedEffectModels
using JSON3
using Parquet2
using StatsModels

function benchmark_threads()
    requested = tryparse(Int, get(ENV, "JULIA_NUM_THREADS", ""))
    if requested === nothing || requested < 1
        error("JULIA_NUM_THREADS must be set to a positive integer before running benchmarks")
    end
    actual = Threads.nthreads()
    if actual != requested
        error("Julia started with $actual thread(s), but JULIA_NUM_THREADS=$requested; set it before Julia starts")
    end
    return actual
end

if length(ARGS) != 1
    error("Expected exactly one argument: path to JSON config.")
end

config = JSON3.read(read(ARGS[1], String))
df = DataFrame(Parquet2.Dataset(String(config[:data_path])))

depvar = term(Symbol(String(config[:depvar])))
rhs_expr = foldl(+, [term(Symbol(String(c))) for c in config[:covariates]])
fe_expr = foldl(+, [fe(Symbol(String(c))) for c in config[:fe_cols]])

julia_nthreads = benchmark_threads()
println(stderr, "[bench] julia.FixedEffectModels agreement check using $(julia_nthreads) thread(s)")
model = reg(df, depvar ~ rhs_expr + fe_expr, Vcov.simple(), nthreads=julia_nthreads, progress_bar=false)
if hasproperty(model, :converged) && !getproperty(model, :converged)
    error("FixedEffectModels agreement model did not converge")
end
coefs = coef(model)

println(JSON3.write(Dict(
    "backend" => "FixedEffectModels",
    "names" => coefnames(model),
    "coefficients" => collect(coefs),
)))
