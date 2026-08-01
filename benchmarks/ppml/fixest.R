#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
})
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[[1]]
output_path <- args[[2]]
repetitions <- as.integer(args[[4]])
threads <- as.integer(Sys.getenv("BENCH_THREADS"))
setFixest_nthreads(threads)
if (getFixest_nthreads() != threads) stop("fixest did not accept BENCH_THREADS")
frame <- as.data.frame(read_parquet(data_path))
formula <- negbin_y ~ x1 | indiv_id + firm_id + year
fit_once <- function() {
  fit <- fepois(
    formula, frame, vcov = "iid", nthreads = threads, glm.iter = 100L,
    notes = FALSE, warn = FALSE
  )
  if (!isTRUE(fit$convStatus)) stop("fixest PPML model returned without convergence")
  fit
}
warmup_fit <- fit_once()
rm(warmup_fit)
rows <- vector("list", repetitions)
for (index in seq_len(repetitions)) {
  started <- proc.time()[["elapsed"]]
  rows[[index]] <- tryCatch({
    fit <- fit_once()
    data.frame(
      backend = "fixest", repetition = index - 1L,
      runtime_s = proc.time()[["elapsed"]] - started,
      n_retained = nobs(fit), beta_x1 = unname(coef(fit)[["x1"]]),
      deviance = as.numeric(deviance(fit)), converged = TRUE, error = ""
    )
  }, error = function(error) data.frame(
    backend = "fixest", repetition = index - 1L,
    runtime_s = proc.time()[["elapsed"]] - started,
    n_retained = NA_integer_, beta_x1 = NA_real_, deviance = NA_real_,
    converged = FALSE, error = conditionMessage(error)
  ))
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), output_path, row.names = FALSE, na = "")
