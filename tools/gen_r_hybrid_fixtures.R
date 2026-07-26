#!/usr/bin/env Rscript
# pybnlearn: reference results for the hybrid algorithms.
#
# Usage: Rscript tools/gen_r_hybrid_fixtures.R [outdir]
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

datasets = list()
for (nm in c("learning.test", "asia", "coronary", "lizards",
             "gaussian.test", "marks")) {
  e = new.env(); data(list = nm, envir = e)
  datasets[[nm]] = get(nm, envir = e)
}

records = character(0)

run = function(dataset, algorithm, restrict = NULL, maximize = NULL,
               restrict.args = list(), maximize.args = list(),
               wl = NULL, bl = NULL) {
  d = datasets[[dataset]]

  argv = list(x = d, restrict.args = restrict.args,
              maximize.args = maximize.args)
  if (!is.null(restrict)) argv$restrict = restrict
  if (!is.null(maximize)) argv$maximize = maximize
  if (!is.null(wl)) argv$whitelist = wl
  if (!is.null(bl)) argv$blacklist = bl

  cat("  ", dataset, algorithm, restrict, maximize, "\n", file = stderr())
  net = do.call(algorithm, argv)

  ra = if (length(restrict.args) == 0) "{}" else
    paste0("{", paste(sprintf('%s:%s', sapply(names(restrict.args), jstr),
      sapply(restrict.args, function(v) if (is.character(v)) jstr(v) else num(v))),
      collapse = ","), "}")
  ma = if (length(maximize.args) == 0) "{}" else
    paste0("{", paste(sprintf('%s:%s', sapply(names(maximize.args), jstr),
      sapply(maximize.args, function(v) if (is.character(v)) jstr(v) else num(v))),
      collapse = ","), "}")

  records <<- c(records, sprintf(paste0(
      '{"dataset":%s,"algorithm":%s,"restrict":%s,"maximize":%s,',
      '"restrict.args":%s,"maximize.args":%s,"whitelist":%s,"blacklist":%s,',
      '"modelstring":%s,"arcs":%s,"nodes":%s}'),
    jstr(dataset), jstr(algorithm),
    if (is.null(restrict)) "null" else jstr(restrict),
    if (is.null(maximize)) "null" else jstr(maximize),
    ra, ma, jarcs(wl), jarcs(bl),
    jstr(modelstring(net)), jarcs(net$arcs), jarr(names(net$nodes))))
}

discrete = c("learning.test", "asia", "coronary", "lizards")
continuous = c("gaussian.test", "marks")

for (ds in c(discrete, continuous))
  run(ds, "mmhc")

# rsmax2 over the restrict/maximize combinations that are ported.
for (ds in c("learning.test", "asia", "coronary")) {
  for (rst in c("mmpc", "si.hiton.pc", "gs", "iamb", "inter.iamb",
                "iamb.fdr", "pc.stable")) {
    run(ds, "rsmax2", restrict = rst, maximize = "hc")
    run(ds, "rsmax2", restrict = rst, maximize = "tabu")
  }
}
for (ds in continuous)
  for (rst in c("mmpc", "si.hiton.pc", "pc.stable"))
    run(ds, "rsmax2", restrict = rst, maximize = "hc")

# arguments passed through to each phase.
run("learning.test", "mmhc", restrict.args = list(alpha = 0.01))
run("learning.test", "mmhc", restrict.args = list(test = "x2"))
run("learning.test", "mmhc", maximize.args = list(score = "bde", iss = 5))
run("learning.test", "rsmax2", restrict = "mmpc", maximize = "hc",
    restrict.args = list(alpha = 0.2), maximize.args = list(score = "aic"))
run("learning.test", "rsmax2", restrict = "aracne", maximize = "hc")
run("learning.test", "rsmax2", restrict = "chow.liu", maximize = "hc")

# constraints.
wl = data.frame(from = "A", to = "F", stringsAsFactors = FALSE)
bl = data.frame(from = c("A", "C"), to = c("B", "D"), stringsAsFactors = FALSE)
run("learning.test", "mmhc", wl = wl)
run("learning.test", "mmhc", bl = bl)
run("learning.test", "mmhc", wl = wl, bl = bl)
run("learning.test", "rsmax2", restrict = "si.hiton.pc", maximize = "hc",
    bl = bl)

out = file.path(outdir, "hybrid.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)
cat(sprintf("wrote %s: %d reference networks\n", out, length(records)))
