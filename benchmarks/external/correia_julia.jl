#!/usr/bin/env julia

# The Correia collection ships as CSV, not Parquet, so this driver reads its
# own data and cannot use run_manifest. It shares the thread check, which is
# the part that must agree across every driver: a sweep whose drivers disagree
# about thread count is not a like-for-like comparison.
using CSV
using FixedEffectModels
using StatsModels

include(joinpath(@__DIR__, "bench_common.jl"))

function main()
    if length(ARGS) != 1
        error("Expected exactly one argument: path to JSON config.")
    end

    config = JSON3.read(read(ARGS[1], String))
    manifest = config[:manifest]
    depvar = String(config[:depvar])
    covariates = String.(config[:covariates])
    fe_cols = String.(config[:fe_cols])

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
            model = reg(df, formula; nthreads=julia_nthreads, progress_bar=false)
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
