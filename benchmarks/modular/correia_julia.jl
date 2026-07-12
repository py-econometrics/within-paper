#!/usr/bin/env julia

using CSV
using DataFrames
using FixedEffectModels
using JSON3
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

function main()
    if length(ARGS) != 1
        error("Expected exactly one argument: path to JSON config.")
    end

    config = JSON3.read(read(ARGS[1], String))
    manifest = config[:manifest]
    depvar = String(config[:depvar])
    covariates = String.(config[:covariates])
    fe_cols = String.(config[:fe_cols])
    tolerance = Float64(config[:tolerance])

    lhs_term = term(Symbol(depvar))
    rhs_expr = foldl(+, [term(Symbol(c)) for c in covariates])
    fe_expr = foldl(+, [fe(Symbol(col)) for col in fe_cols])
    formula = lhs_term ~ rhs_expr + fe_expr
    julia_nthreads = benchmark_threads()
    println(stderr, "[bench] julia.FixedEffectModels using $(julia_nthreads) thread(s) (Threads.nthreads(); Sys.CPU_THREADS=$(Sys.CPU_THREADS))")

    for entry in manifest
        elapsed = nothing
        success = true
        error_msg = nothing
        n_obs = Int(entry[:n_obs])

        try
            df = CSV.read(String(entry[:data_path]), DataFrame)
            n_obs = nrow(df)
            start_time = time()
            model = reg(df, formula; tol=tolerance, nthreads=julia_nthreads, progress_bar=false)
            if hasproperty(model, :converged) && !getproperty(model, :converged)
                error("FixedEffectModels model returned without convergence")
            end
            elapsed = time() - start_time
        catch e
            success = false
            error_msg = sprint(showerror, e)
        end

        result = Dict(
            "dataset_id" => String(entry[:dataset_id]),
            "dgp" => String(entry[:dgp]),
            "n_obs" => n_obs,
            "iter_type" => String(entry[:iter_type]),
            "iter_num" => Int(entry[:iter_num]),
            "time" => elapsed,
            "success" => success,
            "error" => error_msg,
        )
        println(stdout, JSON3.write(result))
        flush(stdout)
        GC.gc()
    end
end

main()
