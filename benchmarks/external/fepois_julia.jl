#!/usr/bin/env julia

# Poisson benchmark driver for GLFixedEffectModels.jl.
#
# The manifest walk, timing, record schema, result log, thread check and
# progress table live in bench_common.jl. Only the formula and the fit are
# here, because nlreg takes a different signature from reg and reports its own
# convergence flag.

using GLFixedEffectModels
using GLM
using Distributions
using StatsModels
using Logging

include(joinpath(@__DIR__, "bench_common.jl"))

# Suppress GLFixedEffectModels `@info` notices (e.g. "N observations detected
# as separated using the FE method. Dropping them ...") so the benchmark table
# is not interleaved with them.
Logging.disable_logging(Logging.Info)

function main()
    if length(ARGS) != 1
        error("Expected exactly one argument: path to JSON config.")
    end

    config = JSON3.read(read(ARGS[1], String))
    vcov_spec = parse_vcov(String(config[:vcov_type]))

    depvar = String(config[:depvar])
    covariates = String.(config[:covariates])
    fe_cols = String.(config[:fe_cols])

    lhs_term = term(Symbol(depvar))
    rhs_expr = sum_terms([term(Symbol(c)) for c in covariates])
    fe_expr = sum_terms([fe(Symbol(col)) for col in fe_cols])
    formula = lhs_term ~ rhs_expr + fe_expr
    start = zeros(length(covariates))

    run_manifest(config, "julia.GLFixedEffectModels (fepois)") do df, nthreads
        model = nlreg(
            df,
            formula,
            Poisson(),
            LogLink(),
            vcov_spec,
            start=start,
            maxiter=100,
            separation=[:fe],
            nthreads=nthreads,
        )
        if !model.converged
            error("GLFixedEffectModels model returned without convergence")
        end
        return Dict{String,Any}()
    end
end

main()
