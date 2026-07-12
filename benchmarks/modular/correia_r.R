#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})

bench_threads <- suppressWarnings(as.integer(Sys.getenv("BENCH_THREADS", unset = "")))
if (is.na(bench_threads) || bench_threads < 1) stop("BENCH_THREADS must be set to a positive integer")
setFixest_nthreads(bench_threads)
if (getFixest_nthreads() != bench_threads) stop("fixest did not accept BENCH_THREADS")
message(sprintf("[bench] r.fixest using %d thread(s)", getFixest_nthreads()))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Expected exactly one argument: path to JSON config.")
}

config <- fromJSON(args[[1]], simplifyVector = FALSE)
manifest <- config$manifest
formula <- as.formula(config$formula)
tolerance <- config$tolerance
fixef_iterations <- config$fixef_iterations

for (entry in manifest) {
  elapsed <- NULL
  success <- TRUE
  error_msg <- NULL
  n_obs <- entry$n_obs

  tryCatch(
    {
      df <- utils::read.csv(entry$data_path)
      n_obs <- nrow(df)
      elapsed <- unname(system.time({
        suppressMessages(
          fit <- feols(
            formula,
            data = df,
            fixef.tol = tolerance,
            fixef.iter = fixef_iterations,
            nthreads = bench_threads
          )
        )
        if (!is.null(fit$convStatus) && !isTRUE(fit$convStatus)) {
          stop("fixest model returned without convergence")
        }
      })[["elapsed"]])
    },
    error = function(e) {
      success <<- FALSE
      error_msg <<- conditionMessage(e)
      elapsed <<- NULL
    }
  )

  cat(
    toJSON(
      list(
        dataset_id = entry$dataset_id,
        dgp = entry$dgp,
        n_obs = n_obs,
        iter_type = entry$iter_type,
        iter_num = entry$iter_num,
        time = elapsed,
        success = success,
        error = error_msg
      ),
      auto_unbox = TRUE,
      null = "null"
    ),
    "\n"
  )

  if (exists("df")) {
    rm(df)
  }
  gc(verbose = FALSE)
}
