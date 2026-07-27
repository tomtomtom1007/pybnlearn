#!/usr/bin/env Rscript
# pybnlearn: what R's direct.lingam actually returns, so the one documented
# structural divergence is measured rather than asserted.
#
# R selects a node's parents with glmnet's adaptive lasso; this package uses
# a score-based search, because glmnet is not vendored.  The README says so,
# but "chooses the arcs differently" is a claim about how *often* and how
# *much*, and that was worth measuring rather than leaving to the reader's
# imagination -- particularly since the answer turns out to be "rarely".
#
# Reproducing glmnet's path exactly is not on the table: its stopping rule
# compares the gain in deviance against 1e-5 times the deviance, and on the
# first data set tried that comparison came down to 7.865e-06 against
# 7.634e-06.  A three percent margin means the number of lambdas returned
# flips with the convergence tolerance, so matching it needs glmnet's
# arithmetic and not merely its algorithm.  What can be compared is the
# answer: which arcs come out.
#
# Usage: Rscript tools/gen_r_lingam_arcs_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages({
  library(bnlearn)
  library(glmnet)
})

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"

jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
field = function(name, value) paste0(jstr(name), ":", value)
jarcs = function(a) {
  if (is.null(a) || length(a) == 0) return("[]")
  if (is.null(dim(a))) a = matrix(a, ncol = 2)
  if (nrow(a) == 0) return("[]")
  paste0("[", paste(apply(a, 1, function(r)
    paste0("[", jstr(r[1]), ",", jstr(r[2]), "]")), collapse = ","), "]")
}

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

# The continuous data sets already committed, at more than one sample size:
# the divergence that exists shows up at one size and not another, which is
# itself worth pinning down.
sets = list(
  list(name = "nongaussian", file = "nongaussian.csv", rows = c(200, 400)),
  list(name = "gaussian.test", file = "gaussian.test.csv", rows = c(200, 1000, 5000)),
  list(name = "collinear", file = "sweep.collinear.csv", rows = c(200)),
  list(name = "heavy-tailed", file = "sweep.heavy-tailed.csv", rows = c(180)),
  list(name = "scaled", file = "sweep.scaled.csv", rows = c(150)))

for (spec in sets) {
  full = read.csv(file.path(outdir, spec$file))

  for (n in spec$rows) {
    d = head(full, n)
    learned = tryCatch(direct.lingam(d), error = function(e) NULL)
    if (is.null(learned)) next

    add(field("kind", jstr("lingam.arcs")),
        field("dataset", jstr(spec$name)),
        field("file", jstr(spec$file)),
        field("rows", sprintf("%d", n)),
        field("ordering", jarr(learned$learning$ordering)),
        field("arcs", jarcs(arcs(learned))))
  }
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "lingam_arcs.json"))
cat("wrote", length(records), "records to",
    file.path(outdir, "lingam_arcs.json"), "\n")
