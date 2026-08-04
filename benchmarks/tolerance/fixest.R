#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
})
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[[1]]
output_path <- args[[2]]
design <- args[[3]]
tolerances <- as.numeric(strsplit(args[[4]], ",", fixed = TRUE)[[1]])
default_tolerance <- as.numeric(args[[5]])
repetitions <- as.integer(args[[6]])
maxiter <- as.integer(args[[7]])
reference_beta <- as.numeric(args[[8]])
reference_se <- as.numeric(args[[9]])
threads <- as.integer(Sys.getenv("BENCH_THREADS"))
setFixest_nthreads(threads)
if (getFixest_nthreads() != threads) stop("fixest did not accept BENCH_THREADS")

frame <- as.data.frame(read_parquet(data_path))
reference_norm <- sqrt(sum(frame$reference_residual^2))
formula <- y ~ x1 | indiv_id + firm_id + year
fit_once <- function(tolerance) {
  capped <- FALSE
  fit <- withCallingHandlers(
    feols(
      formula, frame, vcov = "iid", fixef.tol = tolerance,
      fixef.iter = maxiter, nthreads = threads, warn = TRUE, notes = FALSE
    ),
    warning = function(warning) {
      if (grepl("Absence of convergence", conditionMessage(warning), fixed = TRUE)) {
        capped <<- TRUE
        invokeRestart("muffleWarning")
      }
    }
  )
  if (capped) stop("fixest demeaning reached the maximum number of iterations")
  if (!is.null(fit$convStatus) && !isTRUE(fit$convStatus)) {
    stop("fixest model reached maxiter without convergence")
  }
  fit
}
warmup_fit <- try(fit_once(tolerances[[1]]), silent = TRUE)
if (!inherits(warmup_fit, "try-error")) rm(warmup_fit)
rows <- list()
index <- 1L
for (tolerance in tolerances) {
  for (repetition in seq_len(repetitions)) {
    started <- proc.time()[["elapsed"]]
    rows[[index]] <- tryCatch({
      fit <- fit_once(tolerance)
      residual <- as.numeric(resid(fit))
      if (length(residual) != nrow(frame)) stop("fit did not retain the pre-pruned sample")
      beta <- unname(coef(fit)[["x1"]])
      data.frame(
        design = design, backend = "fixest", tolerance = tolerance,
        default_tolerance = default_tolerance, repetition = repetition - 1L,
        runtime_s = proc.time()[["elapsed"]] - started, beta_x1 = beta,
        coefficient_error_se = abs(beta - reference_beta) / reference_se,
        residual_error = sqrt(sum((residual - frame$reference_residual)^2)) / reference_norm,
        converged = TRUE, capped = FALSE, error = ""
      )
    }, error = function(error) {
      message <- conditionMessage(error)
      data.frame(
        design = design, backend = "fixest", tolerance = tolerance,
        default_tolerance = default_tolerance, repetition = repetition - 1L,
        runtime_s = proc.time()[["elapsed"]] - started, beta_x1 = NA_real_,
        coefficient_error_se = NA_real_, residual_error = NA_real_,
        converged = FALSE,
        capped = grepl("maximum.*iterations|maxiter|iteration cap", message, ignore.case = TRUE),
        error = message
      )
    })
    index <- index + 1L
  }
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), output_path, row.names = FALSE, na = "")
