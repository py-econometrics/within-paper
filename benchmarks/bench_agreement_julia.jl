#!/usr/bin/env julia

using DataFrames
using FixedEffectModels
using JSON3
using Parquet2
using StatsModels

if length(ARGS) != 1
    error("Expected exactly one argument: path to JSON config.")
end

config = JSON3.read(read(ARGS[1], String))
df = DataFrame(Parquet2.Dataset(String(config[:data_path])))

depvar = term(Symbol(String(config[:depvar])))
rhs_expr = foldl(+, [term(Symbol(String(c))) for c in config[:covariates]])
fe_expr = foldl(+, [fe(Symbol(String(c))) for c in config[:fe_cols]])

model = reg(df, depvar ~ rhs_expr + fe_expr, Vcov.simple(), progress_bar=false)
coefs = coef(model)

println(JSON3.write(Dict(
    "backend" => "FixedEffectModels",
    "names" => coefnames(model),
    "coefficients" => collect(coefs),
)))
