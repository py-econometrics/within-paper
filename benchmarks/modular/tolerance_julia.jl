#!/usr/bin/env julia

using DataFrames
using FixedEffectModels
using JSON3
using LinearAlgebra
using Parquet2
using Random
using StatsModels

function main()
    length(ARGS) == 1 || error("Expected one argument: path to the JSON configuration.")
    config = JSON3.read(read(ARGS[1], String))

    threads = Int(config[:thread_count])
    Threads.nthreads() == threads || error(
        "Julia is using $(Threads.nthreads()) threads; expected $threads"
    )

    df = DataFrame(Parquet2.Dataset(String(config[:data_path])))
    nrow(df) == Int(config[:n_obs]) || error(
        "Prepared sample row count does not match the configuration"
    )
    formula = @formula(y ~ x1 + fe(indiv_id) + fe(firm_id) + fe(year))
    vcov = Vcov.simple()
    maxiter = Int(config[:maxiter])

    function fit_once(tolerance::Float64)
        started = time_ns()
        model = reg(
            df,
            formula,
            vcov;
            save = :residuals,
            tol = tolerance,
            maxiter = maxiter,
            nthreads = threads,
            double_precision = true,
            progress_bar = false,
        )
        elapsed = (time_ns() - started) / 1e9
        converged = !hasproperty(model, :converged) || getproperty(model, :converged)
        converged || error("FixedEffectModels model did not converge")
        return model, elapsed
    end

    # Discard one fit before timing the requested settings.
    try
        fit_once(Float64(config[:tolerances][1]))
    catch error_message
        println(stderr, "[warn] FEM.jl warm-up failed: ", error_message)
    end
    GC.gc()

    schedule = [
        (Float64(tolerance), repetition)
        for tolerance in config[:tolerances]
        for repetition in 1:Int(config[:repetitions])
    ]
    Random.seed!(Int(config[:seed]))
    Random.shuffle!(schedule)

    reference_residual = Float64.(df.reference_residual)
    reference_beta = Float64(config[:reference_beta_x1])
    reference_se = Float64(config[:reference_se_x1])
    reference_residual_norm = Float64(config[:reference_residual_norm])

    for (tolerance, repetition) in schedule
        println(
            stderr,
            "[fit] $(config[:design]) $(config[:label]) tol=$tolerance " *
            "rep=$repetition/$(config[:repetitions])",
        )
        flush(stderr)

        started = time_ns()
        elapsed = nothing
        success = false
        converged = false
        beta = nothing
        coefficient_error = nothing
        residual_accuracy = nothing
        iterations = nothing
        error_message = nothing

        try
            model, elapsed = fit_once(tolerance)
            converged = true
            beta = Float64(coef(model)[1])
            residual = Float64.(residuals(model))
            length(residual) == nrow(df) || error(
                "fit did not retain the pre-pruned sample"
            )
            coefficient_error = abs(beta - reference_beta) / reference_se
            residual_accuracy = norm(residual - reference_residual) /
                reference_residual_norm
            if hasproperty(model, :iterations)
                raw_iterations = getproperty(model, :iterations)
                iterations = raw_iterations isa Number ?
                    Int(raw_iterations) : maximum(Int.(raw_iterations))
            end
            success = true
        catch error_value
            elapsed = (time_ns() - started) / 1e9
            error_message = sprint(showerror, error_value)
        end

        capped = error_message !== nothing && occursin(
            r"iter|converg|maxiter"i,
            error_message,
        )
        row = Dict(
            "design" => String(config[:design]),
            "n_obs_source" => Int(config[:n_obs_source]),
            "n_obs" => Int(config[:n_obs]),
            "n_singletons_dropped" => Int(config[:n_singletons_dropped]),
            "sample_hash" => String(config[:sample_hash]),
            "source_path" => String(config[:source_path]),
            "method" => String(config[:method]),
            "label" => String(config[:label]),
            "package" => String(config[:package]),
            "solver" => String(config[:solver]),
            "preconditioner" => String(config[:preconditioner]),
            "tolerance" => tolerance,
            "default_tolerance" => Float64(config[:default_tolerance]),
            "is_default_tolerance" => tolerance == Float64(config[:default_tolerance]),
            "maxiter" => maxiter,
            "repetition" => repetition,
            "time_s" => elapsed,
            "success" => success,
            "converged" => converged,
            "capped" => capped,
            "coefficient_error_se" => coefficient_error,
            "residual_error" => residual_accuracy,
            "beta_x1" => beta,
            "reference_beta_x1" => reference_beta,
            "reference_se_x1" => reference_se,
            "reference_residual_norm" => reference_residual_norm,
            "reference_fe_eta" => Float64(config[:reference_fe_eta]),
            "reference_x_score" => Float64(config[:reference_x_score]),
            "reference_tolerance" => Float64(config[:reference_tolerance]),
            "reference_maxiter" => Int(config[:reference_maxiter]),
            "thread_count" => threads,
            "iterations" => iterations,
            "error" => error_message,
        )
        println(stdout, JSON3.write(row))
        flush(stdout)
        GC.gc()
    end
end

main()
