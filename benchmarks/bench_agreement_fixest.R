#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Expected exactly one argument: path to JSON config.")
}

config <- fromJSON(args[[1]], simplifyVector = FALSE)
df <- as.data.frame(read_parquet(config$data_path))
fit <- feols(as.formula(config$formula), data = df, vcov = "iid")
coefs <- coef(fit)

cat(
  toJSON(
    list(
      backend = "fixest",
      names = names(coefs),
      coefficients = as.numeric(coefs)
    ),
    auto_unbox = TRUE
  ),
  "\n"
)
