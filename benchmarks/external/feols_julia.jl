#!/usr/bin/env julia

# Linear benchmark driver for FixedEffectModels.jl.
#
# The manifest walk, timing, record schema, thread check and progress table
# live in bench_common.jl. Only the formula and the fit are here.

using FixedEffectModels
using StatsModels

include(joinpath(@__DIR__, "bench_common.jl"))

function main()
    if length(ARGS) != 1
        error("Expected exactly one argument: path to JSON config.")
    end

    config = JSON3.read(read(ARGS[1], String))
    vcov_spec = parse_vcov(String(config[:vcov_type]))

    # Build the reg formula: y ~ x1 + fe(indiv_id) + fe(year)
    depvar = String(config[:depvar])
    covariates = String.(config[:covariates])
    fe_cols = String.(config[:fe_cols])

    lhs_term = term(Symbol(depvar))
    rhs_expr = sum_terms([term(Symbol(c)) for c in covariates])
    fe_expr = sum_terms([fe(Symbol(col)) for col in fe_cols])
    formula = lhs_term ~ rhs_expr + fe_expr

    run_manifest(config, "julia.FixedEffectModels (feols)") do df, nthreads
        model = reg(df, formula, vcov_spec, nthreads=nthreads, progress_bar=false)
        if hasproperty(model, :converged) && !getproperty(model, :converged)
            error("FixedEffectModels model returned without convergence")
        end
        return Dict{String,Any}()
    end
end

main()
