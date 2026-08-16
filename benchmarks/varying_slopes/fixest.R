#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
})

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[[1]]
output_path <- args[[2]]
repetitions <- as.integer(args[[3]])
threads <- as.integer(Sys.getenv("BENCH_THREADS"))
setFixest_nthreads(threads)
if (getFixest_nthreads() != threads) stop("fixest did not accept BENCH_THREADS")

frame <- as.data.frame(read_parquet(data_path))
formula <- y ~ x1 | indiv_id[year] + firm_id + year
fit_once <- function() {
  capped <- FALSE
  fit <- withCallingHandlers(
    feols(formula, frame, vcov = "iid", nthreads = threads),
    warning = function(warning) {
      if (grepl("Absence of convergence", conditionMessage(warning), fixed = TRUE)) {
        capped <<- TRUE
        invokeRestart("muffleWarning")
      }
    }
  )
  if (capped) stop("fixest demeaning returned without convergence")
  fit
}

warmup_fit <- try(fit_once(), silent = TRUE)
if (!inherits(warmup_fit, "try-error")) rm(warmup_fit)
rows <- vector("list", repetitions)
for (index in seq_len(repetitions)) {
  started <- proc.time()[["elapsed"]]
  rows[[index]] <- tryCatch({
    fit <- fit_once()
    data.frame(
      backend = "fixest", preconditioner = "",
      package_version = as.character(packageVersion("fixest")),
      repetition = index - 1L, n_planned = repetitions,
      runtime_s = proc.time()[["elapsed"]] - started,
      n_retained = nobs(fit), beta_x1 = unname(coef(fit)[["x1"]]),
      converged = TRUE, capped = FALSE, error = ""
    )
  }, error = function(error) {
    message <- conditionMessage(error)
    data.frame(
      backend = "fixest", preconditioner = "",
      package_version = as.character(packageVersion("fixest")),
      repetition = index - 1L, n_planned = repetitions,
      runtime_s = proc.time()[["elapsed"]] - started,
      n_retained = NA_integer_, beta_x1 = NA_real_, converged = FALSE,
      capped = grepl("without convergence", message, fixed = TRUE),
      error = message
    )
  })
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), output_path, row.names = FALSE, na = "")
