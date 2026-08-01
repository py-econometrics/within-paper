#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
})

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[[1]]
output_path <- args[[2]]
design <- args[[3]]
fixed_effects <- strsplit(args[[4]], ",", fixed = TRUE)[[1]]
requested <- args[[5]]
threads <- as.integer(Sys.getenv("BENCH_THREADS"))
setFixest_nthreads(threads)
if (getFixest_nthreads() != threads) stop("fixest did not accept BENCH_THREADS")

frame <- as.data.frame(read_parquet(data_path))
formula <- as.formula(paste("y ~ x1 |", paste(fixed_effects, collapse = " + ")))
fit_once <- function() feols(formula, frame, vcov = "iid", nthreads = threads)
started <- proc.time()[["elapsed"]]
warmup_fit <- fit_once()
rm(warmup_fit)
warmup <- proc.time()[["elapsed"]] - started
repetitions <- if (requested == "adaptive") {
  if (warmup < 1) 20L else if (warmup < 10) 7L else 3L
} else as.integer(requested)

rows <- vector("list", repetitions)
for (index in seq_len(repetitions)) {
  started <- proc.time()[["elapsed"]]
  rows[[index]] <- tryCatch({
    fit <- fit_once()
    data.frame(
      backend = "fixest", repetition = index - 1L,
      runtime_s = proc.time()[["elapsed"]] - started,
      n_retained = nobs(fit), beta_x1 = unname(coef(fit)[["x1"]]),
      max_eta = NA_real_, converged = TRUE, error = ""
    )
  }, error = function(error) data.frame(
    backend = "fixest", repetition = index - 1L,
    runtime_s = proc.time()[["elapsed"]] - started,
    n_retained = NA_integer_, beta_x1 = NA_real_, max_eta = NA_real_,
    converged = FALSE, error = conditionMessage(error)
  ))
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), output_path, row.names = FALSE, na = "")
