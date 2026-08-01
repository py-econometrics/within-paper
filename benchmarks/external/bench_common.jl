#!/usr/bin/env julia

# Shared scaffolding for the Julia benchmark drivers.
#
# The linear and Poisson drivers differ only in which package fits the model:
# FixedEffectModels and GLFixedEffectModels have different call signatures and
# report different convergence diagnostics. Everything around the fit was
# duplicated between them, so a change to the record schema or the thread check
# had to be made twice and could silently be made only once.
#
# A driver includes this file, then calls `run_manifest` with a closure that
# performs its own fit. The closure raises on non-convergence; the record,
# timing, grouping and progress table are handled here.

using JSON3
using DataFrames
using Parquet2
using Printf
using Statistics

function benchmark_threads()
    requested = tryparse(Int, get(ENV, "BENCH_THREADS", ""))
    if requested === nothing || requested < 1
        error("BENCH_THREADS must be set to a positive integer before running benchmarks")
    end
    actual = Threads.nthreads()
    if actual != requested
        error("Julia started with $actual thread(s), but BENCH_THREADS=$requested")
    end
    return actual
end

# ── table formatting ──
function fmt_time(t::Float64)
    if t < 1.0
        return @sprintf("%.1fms", t * 1000)
    else
        return @sprintf("%.3fs", t)
    end
end

function format_number(n::Int)
    s = string(n)
    result = ""
    for (i, c) in enumerate(reverse(s))
        if i > 1 && (i - 1) % 3 == 0
            result = "," * result
        end
        result = c * result
    end
    return result
end

DGP_W = Ref(16)

function print_header(name::String)
    w = DGP_W[]
    hdr = "  " * rpad("dgp", w) * @sprintf(" %12s %4s %10s %10s %10s  %s", "n_obs", "n_fe", "min", "median", "max", "status")
    sep = "  " * "-"^(length(hdr) - 2)
    println(stderr, "\n  ", name)
    println(stderr, sep)
    println(stderr, hdr)
    println(stderr, sep)
    flush(stderr)
end

function print_row(dgp::String, n_obs::Int, n_fe::Int, times::Vector{Float64})
    w = DGP_W[]
    if length(times) > 0
        mn, md, mx, status = fmt_time(minimum(times)), fmt_time(median(times)), fmt_time(maximum(times)), "ok"
    else
        mn, md, mx, status = "—", "—", "—", "FAIL"
    end
    n_obs_str = format_number(n_obs)
    println(stderr, "  " * rpad(dgp, w) * @sprintf(" %12s %4d %10s %10s %10s  %s", n_obs_str, n_fe, mn, md, mx, status))
    flush(stderr)
end

# ── parse normalized vcov_type: "iid", "hetero", or "cluster:<colname>" ──
function parse_vcov(vcov_type::String)
    if startswith(vcov_type, "cluster:")
        cluster_col = replace(vcov_type, "cluster:" => "")
        return Vcov.cluster(Symbol(cluster_col))
    elseif vcov_type == "iid"
        return Vcov.simple()
    elseif vcov_type == "hetero"
        return Vcov.robust()
    else
        error("Unknown vcov_type: $vcov_type")
    end
end

function sum_terms(terms)
    isempty(terms) && return ConstantTerm(1)
    return foldl(+, terms)
end

"""
    run_manifest(fit, config, name)

Walk the manifest, timing `fit(df)` on each entry and emitting one JSON record
per fit. `fit` raises to signal a failed or non-converged run; the error is
recorded rather than propagated, so one bad cell does not abort a sweep.

The function comes first so that callers can use `do` block syntax.
"""
function run_manifest(fit::Function, config, name::String)
    manifest = config[:manifest]
    fe_cols = String.(config[:fe_cols])
    n_fe = length(fe_cols)
    # A long sweep can die before stdout is flushed. The reader indexes records
    # by (dataset_id, iter_num), so emitting to both places cannot double count.
    result_log_path = haskey(config, :result_log_path) ? String(config[:result_log_path]) : nothing

    DGP_W[] = max(16, maximum(length(String(entry[:dgp])) for entry in manifest))

    julia_nthreads = benchmark_threads()
    println(stderr, "[bench] $(name) using $(julia_nthreads) thread(s)")
    print_header(name)

    prev_dgp = nothing
    prev_nobs = nothing
    group_times = Float64[]

    for entry in manifest
        cur_dgp = String(entry[:dgp])
        cur_nobs = Int(entry[:n_obs])

        # flush previous group when key changes
        if prev_dgp !== nothing && (cur_dgp != prev_dgp || cur_nobs != prev_nobs)
            print_row(prev_dgp, prev_nobs, n_fe, group_times)
            group_times = Float64[]
        end
        prev_dgp = cur_dgp
        prev_nobs = cur_nobs

        dataset_id = String(entry[:dataset_id])
        iter_type = String(entry[:iter_type])
        iter_num = Int(entry[:iter_num])
        data_path = String(entry[:data_path])

        elapsed = nothing
        success = true
        error_msg = nothing
        extra = Dict{String,Any}()

        try
            df = DataFrame(Parquet2.Dataset(data_path))
            start_time = time()
            extra = fit(df, julia_nthreads)
            elapsed = time() - start_time
        catch e
            success = false
            error_msg = string(e)
        end

        # collect trial times (skip burnin)
        if iter_type != "burnin" && elapsed !== nothing
            push!(group_times, elapsed)
        end

        result = Dict{String,Any}(
            "dataset_id" => dataset_id,
            "dgp" => cur_dgp,
            "n_obs" => cur_nobs,
            "iter_type" => iter_type,
            "iter_num" => iter_num,
            "time" => elapsed,
            "success" => success,
            "thread_count" => julia_nthreads,
            "error" => error_msg,
        )
        merge!(result, extra === nothing ? Dict{String,Any}() : extra)
        payload = JSON3.write(result)
        println(stdout, payload)
        flush(stdout)
        if result_log_path !== nothing
            open(result_log_path, "a") do io
                println(io, payload)
            end
        end
    end

    # flush last group
    if prev_dgp !== nothing
        print_row(prev_dgp, prev_nobs, n_fe, group_times)
    end
end
