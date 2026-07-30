#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(jsonlite)
})

bench_threads <- suppressWarnings(as.integer(Sys.getenv("BENCH_THREADS", unset = "")))
if (is.na(bench_threads) || bench_threads < 1) stop("BENCH_THREADS must be set to a positive integer")
setFixest_nthreads(bench_threads)
if (getFixest_nthreads() != bench_threads) stop("fixest did not accept BENCH_THREADS")
message(sprintf("[bench] r.fixest agreement check using %d thread(s)", getFixest_nthreads()))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Expected exactly one argument: path to JSON config.")
}

config <- fromJSON(args[[1]], simplifyVector = FALSE)
df <- as.data.frame(read_parquet(config$data_path))
fit <- feols(as.formula(config$formula), data = df, vcov = "iid", nthreads = bench_threads)
if (!isTRUE(fit$convStatus)) stop("fixest agreement model did not converge")
coefs <- coef(fit)

cat(
  toJSON(
    list(
      backend = "fixest",
      names = names(coefs),
      coefficients = as.numeric(coefs)
    ),
    auto_unbox = TRUE,
    digits = NA
  ),
  "\n"
)
