#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Expected one argument: path to the JSON configuration.")
}

config <- fromJSON(args[[1]], simplifyVector = TRUE)
threads <- as.integer(config$thread_count)
setFixest_nthreads(threads)
if (getFixest_nthreads() != threads) {
  stop("fixest did not accept the requested thread count")
}

df <- as.data.frame(read_parquet(config$data_path))
if (nrow(df) != as.integer(config$n_obs)) {
  stop("Prepared sample row count does not match the configuration")
}

formula <- y ~ x1 | indiv_id + firm_id + year

fit_once <- function(tolerance) {
  started <- proc.time()[["elapsed"]]
  fit <- feols(
    formula,
    data = df,
    vcov = "iid",
    fixef.tol = tolerance,
    fixef.iter = as.integer(config$maxiter),
    nthreads = threads,
    warn = FALSE,
    notes = FALSE
  )
  elapsed <- unname(proc.time()[["elapsed"]] - started)
  converged <- is.null(fit$convStatus) || isTRUE(fit$convStatus)
  if (!converged) {
    stop("fixest model did not converge")
  }
  list(fit = fit, elapsed = elapsed)
}

# Discard one fit before timing the requested settings.
warmup <- try(fit_once(config$tolerances[[1]]), silent = TRUE)
rm(warmup)
gc()

schedule <- expand.grid(
  tolerance = as.numeric(config$tolerances),
  repetition = seq_len(as.integer(config$repetitions))
)
set.seed(as.integer(config$seed))
schedule <- schedule[sample.int(nrow(schedule)), , drop = FALSE]

for (idx in seq_len(nrow(schedule))) {
  tolerance <- schedule$tolerance[[idx]]
  repetition <- schedule$repetition[[idx]]
  message(sprintf(
    "[fit] %s %s tol=%g rep=%d/%d",
    config$design,
    config$label,
    tolerance,
    repetition,
    config$repetitions
  ))

  started <- proc.time()[["elapsed"]]
  elapsed <- NULL
  success <- FALSE
  converged <- FALSE
  beta <- NULL
  coefficient_error <- NULL
  residual_accuracy <- NULL
  iterations <- NULL
  error_msg <- NULL

  tryCatch(
    {
      result <- fit_once(tolerance)
      elapsed <- result$elapsed
      fit <- result$fit
      converged <- TRUE
      beta <- unname(coef(fit)[["x1"]])
      residual <- as.numeric(resid(fit))
      if (length(residual) != nrow(df)) {
        stop("fit did not retain the pre-pruned sample")
      }
      coefficient_error <- abs(beta - config$reference_beta_x1) /
        config$reference_se_x1
      residual_accuracy <- sqrt(sum(
        (residual - df$reference_residual)^2
      )) / config$reference_residual_norm
      iterations <- max(as.integer(fit$iterations))
      success <- TRUE
    },
    error = function(e) {
      elapsed <<- unname(proc.time()[["elapsed"]] - started)
      error_msg <<- conditionMessage(e)
    }
  )

  capped <- !is.null(error_msg) &&
    grepl("iter|converg|maxiter", error_msg, ignore.case = TRUE)

  row <- list(
    design = config$design,
    n_obs_source = config$n_obs_source,
    n_obs = config$n_obs,
    n_singletons_dropped = config$n_singletons_dropped,
    sample_hash = config$sample_hash,
    source_path = config$source_path,
    method = config$method,
    label = config$label,
    package = config$package,
    solver = config$solver,
    preconditioner = config$preconditioner,
    tolerance = tolerance,
    default_tolerance = config$default_tolerance,
    is_default_tolerance = tolerance == config$default_tolerance,
    maxiter = config$maxiter,
    repetition = repetition,
    time_s = elapsed,
    success = success,
    converged = converged,
    capped = capped,
    coefficient_error_se = coefficient_error,
    residual_error = residual_accuracy,
    beta_x1 = beta,
    reference_beta_x1 = config$reference_beta_x1,
    reference_se_x1 = config$reference_se_x1,
    reference_residual_norm = config$reference_residual_norm,
    reference_fe_eta = config$reference_fe_eta,
    reference_x_score = config$reference_x_score,
    reference_tolerance = config$reference_tolerance,
    reference_maxiter = config$reference_maxiter,
    thread_count = threads,
    iterations = iterations,
    error = error_msg
  )
  cat(toJSON(row, auto_unbox = TRUE, null = "null", digits = NA), "\n")
  rm(list = intersect(c("fit", "residual", "result"), ls()))
  gc()
}
