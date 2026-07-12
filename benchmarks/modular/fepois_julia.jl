#!/usr/bin/env julia

using JSON3
using DataFrames
using Parquet2
using GLFixedEffectModels
using GLM
using Distributions
using StatsModels
using Printf
using Statistics
using Logging

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

# Suppress GLFixedEffectModels `@info` notices (e.g. "N observations detected
# as separated using the FE method. Dropping them ...") so the benchmark table
# is not interleaved with them.
Logging.disable_logging(Logging.Info)

# ── table formatting ──
function fmt_time(t::Float64)
    if t < 1.0
        return @sprintf("%.1fms", t * 1000)
    else
        return @sprintf("%.3fs", t)
    end
end

DGP_W = Ref(16)

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
    if isempty(times)
        println(stderr, "  ", rpad(dgp, w),
            @sprintf(" %12s %4d %10s %10s %10s  %s", format_number(n_obs), n_fe, "—", "—", "—", "FAIL"))
    else
        mn = fmt_time(minimum(times))
        md = fmt_time(median(times))
        mx = fmt_time(maximum(times))
        println(stderr, "  ", rpad(dgp, w),
            @sprintf(" %12s %4d %10s %10s %10s  %s", format_number(n_obs), n_fe, mn, md, mx, "ok"))
    end
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
    if isempty(terms)
        return ConstantTerm(1)
    end
    return foldl(+, terms)
end

# ── main ──
function main()
    if length(ARGS) != 1
        error("Expected exactly one argument: path to JSON config.")
    end

    config = JSON3.read(read(ARGS[1], String))
    manifest = config[:manifest]
    fe_cols = String.(config[:fe_cols])
    n_fe = length(fe_cols)
    vcov_type = String(config[:vcov_type])
    vcov_spec = parse_vcov(vcov_type)
    result_log_path = haskey(config, :result_log_path) ? String(config[:result_log_path]) : nothing

    DGP_W[] = max(16, maximum(length(String(entry[:dgp])) for entry in manifest))

    depvar = String(config[:depvar])
    covariates = String.(config[:covariates])

    lhs_term = term(Symbol(depvar))
    rhs_expr = sum_terms([term(Symbol(c)) for c in covariates])
    fe_expr = sum_terms([fe(Symbol(col)) for col in fe_cols])
    formula = lhs_term ~ rhs_expr + fe_expr
    start = zeros(length(covariates))

    julia_nthreads = benchmark_threads()
    println(stderr, @sprintf(
        "[bench] julia.GLFixedEffectModels using %d thread(s) (Threads.nthreads(); Sys.CPU_THREADS=%d)",
        julia_nthreads, Sys.CPU_THREADS,
    ))

    print_header("julia.GLFixedEffectModels (fepois)")

    prev_dgp = nothing
    prev_nobs = nothing
    group_times = Float64[]

    for entry in manifest
        cur_dgp = String(entry[:dgp])
        cur_nobs = Int(entry[:n_obs])

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

        try
            println(stderr, @sprintf(
                "[bench] GLFEM starting dgp=%s n_obs=%d n_fe=%d iter=%d",
                cur_dgp, cur_nobs, n_fe, iter_num,
            ))
            flush(stderr)
            df = DataFrame(Parquet2.Dataset(data_path))
            start_time = time()
            model = nlreg(
                df,
                formula,
                Poisson(),
                LogLink(),
                vcov_spec,
                start=start,
                maxiter=100,
                maxiter_center=10000,
                center_tol=1e-8,
                separation=[:fe],
                nthreads=julia_nthreads,
            )
            if !model.converged
                error("GLFixedEffectModels model returned without convergence")
            end
            elapsed = time() - start_time
        catch e
            success = false
            error_msg = string(e)
        end

        if iter_type != "burnin" && elapsed !== nothing
            push!(group_times, elapsed)
        end

        result = Dict(
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
        payload = JSON3.write(result)
        println(stdout, payload)
        flush(stdout)
        if result_log_path !== nothing
            open(result_log_path, "a") do io
                println(io, payload)
                flush(io)
            end
        end
    end

    if prev_dgp !== nothing
        print_row(prev_dgp, prev_nobs, n_fe, group_times)
    end
end

main()
