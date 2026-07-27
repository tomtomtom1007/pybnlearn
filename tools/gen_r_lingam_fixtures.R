#!/usr/bin/env Rscript
# pybnlearn: reference results for the direct LiNGAM causal ordering.
#
# Only the ordering.  R's direct.lingam() then picks the arcs with glmnet's
# adaptive lasso, whose coordinate-descent solver and lambda path are not
# vendored here, so there is nothing to compare arcs against.
#
# The ordering is the part that is LiNGAM: it is what non-Gaussianity buys
# you over conditional independence.  It is also greedy and entirely
# deterministic, so a single wrong entry cascades into every later one --
# which makes the whole sequence worth comparing rather than a summary of it.
#
# Data with genuinely non-Gaussian noise is included alongside bnlearn's own
# Gaussian data sets.  On Gaussian data the ordering is arbitrary in theory,
# and comparing it is a test of the arithmetic rather than of the method --
# which is exactly what a port needs.
#
# Usage: Rscript tools/gen_r_lingam_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

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

e = new.env()
data(gaussian.test, envir = e)
data(marks, envir = e)

# A network with genuinely non-Gaussian noise, which is the assumption
# LiNGAM rests on: uniform, exponential and Laplace errors, so the ordering
# is identifiable rather than arbitrary.
set.seed(21)
n = 400
laplace = function(n) {
  u = runif(n) - 0.5
  -sign(u) * log(1 - 2 * abs(u))
}
V1 = runif(n, -2, 2)
V2 = 0.8 * V1 + rexp(n) - 1
V3 = -0.5 * V1 + 0.6 * V2 + laplace(n)
V4 = 0.9 * V3 + runif(n, -1, 1)
V5 = 0.4 * V2 - 0.7 * V4 + rexp(n) - 1
nongauss = data.frame(V1 = V1, V2 = V2, V3 = V3, V4 = V4, V5 = V5)

exact = nongauss
for (col in names(exact))
  exact[[col]] = sprintf("%.17g", exact[[col]])
write.csv(exact, file.path(outdir, "nongaussian.csv"), row.names = FALSE,
          quote = FALSE)

sets = list("gaussian.test" = get("gaussian.test", envir = e),
            "marks" = get("marks", envir = e),
            "nongaussian" = nongauss)

for (name in names(sets)) {
  d = sets[[name]]

  # the whole data set, and a couple of subsets: the ordering is greedy, so
  # a different sample size exercises a different sequence of decisions.
  for (rows in unique(c(nrow(d), min(200, nrow(d)), min(80, nrow(d))))) {
    subset = head(d, rows)

    add(field("kind", jstr("ordering")),
        field("dataset", jstr(name)),
        field("rows", sprintf("%d", rows)),
        field("nodes", jarr(names(d))),
        field("ordering", jarr(bnlearn:::dlingam.ordering(
          subset, mi = "pwling", whitelist = NULL, blacklist = NULL))))
  }

  # constraints, which fix part of the ordering before the search starts.
  ns = names(d)

  wl = matrix(c(ns[2], ns[1]), ncol = 2)
  add(field("kind", jstr("ordering")),
      field("dataset", jstr(name)),
      field("rows", sprintf("%d", nrow(d))),
      field("nodes", jarr(ns)),
      field("whitelist", jarcs(wl)),
      field("ordering", jarr(bnlearn:::dlingam.ordering(
        d, mi = "pwling",
        whitelist = bnlearn:::build.whitelist(wl, nodes = ns, data = d,
                      algo = "direct.lingam", criterion = "pwling"),
        blacklist = NULL))))

  # blacklist every arc out of one node, which makes it a leaf.
  bl = matrix(c(rep(ns[1], length(ns) - 1), ns[-1]), ncol = 2)
  add(field("kind", jstr("ordering")),
      field("dataset", jstr(name)),
      field("rows", sprintf("%d", nrow(d))),
      field("nodes", jarr(ns)),
      field("blacklist", jarcs(bl)),
      field("ordering", jarr(bnlearn:::dlingam.ordering(
        d, mi = "pwling", whitelist = NULL,
        blacklist = bnlearn:::build.blacklist(bl, NULL, ns,
                      algo = "direct.lingam")))))
}

# the pieces the ordering is built from, checked on their own so that a
# disagreement can be localised rather than merely observed.
d = head(get("gaussian.test", envir = e), 200)
for (pair in list(c("A", "B"), c("A", "C"), c("C", "F"), c("B", "D"))) {
  xi = as.numeric(bnlearn:::.scale(d[, pair[1]]))
  xj = as.numeric(bnlearn:::.scale(d[, pair[2]]))

  add(field("kind", jstr("pairwise")),
      field("x", jstr(pair[1])), field("y", jstr(pair[2])),
      field("rows", "200"),
      field("value", sprintf("%.17g",
        bnlearn:::approx.mutual.information(xi, xj))),
      field("residual.sd", sprintf("%.17g",
        bnlearn:::cgsd(bnlearn:::dlingam.remove.effect(xi, xj)))))
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "lingam.json"))
cat("wrote", length(records), "records to", file.path(outdir, "lingam.json"),
    "\n")
