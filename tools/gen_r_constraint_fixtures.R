#!/usr/bin/env Rscript
# pybnlearn: reference results for the constraint-based algorithms.
#
# These are the ones where the order of the tests is part of the algorithm --
# gs breaks out of its grow loop at the first node that passes, so iterating
# candidates differently gives a different Markov blanket and, eventually, a
# different graph that still looks entirely reasonable.  Only a comparison
# against R catches that.
#
# Usage: Rscript tools/gen_r_constraint_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

num = function(x) sprintf("%.17g", as.numeric(x))
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
jarcs = function(a) if (is.null(a) || nrow(a) == 0) "[]" else
                      jarr(paste(a[, "from"], a[, "to"], sep = ">"))
jbool = function(b) if (isTRUE(b)) "true" else "false"

datasets = list()
for (nm in c("learning.test", "asia", "coronary", "lizards",
             "gaussian.test", "marks")) {
  e = new.env()
  data(list = nm, envir = e)
  datasets[[nm]] = get(nm, envir = e)
}

records = character(0)

run = function(dataset, algorithm, test, alpha = 0.05, wl = NULL, bl = NULL,
               undirected = FALSE) {
  data = datasets[[dataset]]
  argv = list(x = data, test = test, alpha = alpha, undirected = undirected)
  if (!is.null(wl)) argv$whitelist = wl
  if (!is.null(bl)) argv$blacklist = bl

  cat("  ", dataset, algorithm, test, alpha, "\n", file = stderr())
  net = do.call(algorithm, argv)

  records <<- c(records, sprintf(paste0(
      '{"dataset":%s,"algorithm":%s,"test":%s,"alpha":%s,"whitelist":%s,',
      '"blacklist":%s,"undirected":%s,"arcs":%s,"nodes":%s}'),
    jstr(dataset), jstr(algorithm), jstr(test), num(alpha),
    jarcs(wl), jarcs(bl), jbool(undirected),
    jarcs(net$arcs), jarr(names(net$nodes))))
}

# mmpc and si.hiton.pc default to undirected = TRUE upstream, but every case
# here passes `undirected` explicitly so the fixtures cover both modes for all
# of them and do not silently depend on a default.
algorithms = c("gs", "iamb", "inter.iamb", "iamb.fdr", "mmpc", "si.hiton.pc",
               "pc.stable")
discrete = c("learning.test", "asia", "coronary", "lizards")
continuous = c("gaussian.test", "marks")

for (algo in algorithms) {
  for (ds in discrete)
    for (tst in c("mi", "x2", "mi-adf", "mi-sh"))
      run(ds, algo, tst)
  for (ds in continuous)
    for (tst in c("cor", "zf", "mi-g"))
      run(ds, algo, tst)
}

# alpha changes which edges survive, so it changes the graph.
for (algo in algorithms)
  for (a in c(0.01, 0.1, 0.2))
    run("learning.test", algo, "mi", alpha = a)

# undirected output skips the orientation phase entirely.
for (algo in algorithms)
  for (ds in c("learning.test", "asia", "coronary"))
    run(ds, algo, "mi", undirected = TRUE)

# constraints.
wl = data.frame(from = "A", to = "F", stringsAsFactors = FALSE)
bl = data.frame(from = c("A", "C"), to = c("B", "D"),
                stringsAsFactors = FALSE)
for (algo in algorithms) {
  run("learning.test", algo, "mi", wl = wl)
  run("learning.test", algo, "mi", bl = bl)
  run("learning.test", algo, "mi", wl = wl, bl = bl)
}

wl2 = data.frame(from = "A", to = "T", stringsAsFactors = FALSE)
bl2 = data.frame(from = "S", to = "L", stringsAsFactors = FALSE)
for (algo in algorithms) {
  run("asia", algo, "mi", wl = wl2)
  run("asia", algo, "x2", bl = bl2)
}

out = file.path(outdir, "constraint.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)

cat(sprintf("wrote %s: %d reference networks\n", out, length(records)))
cat(sprintf("R %s, bnlearn %s\n", getRversion(),
            as.character(packageVersion("bnlearn"))))
