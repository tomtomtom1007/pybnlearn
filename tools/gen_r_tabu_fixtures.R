#!/usr/bin/env Rscript
# pybnlearn: reference results for tabu search.
#
# Tabu search only differs from hill climbing once it has to take a move that
# makes the score worse, so the fixtures record whether each case actually
# diverged from hc.  A suite where every case agrees with hc would pass without
# exercising the tabu list, the loss-iteration counter or the best-network
# bookkeeping at all.
#
# Usage: Rscript tools/gen_r_tabu_fixtures.R [outdir]
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
for (nm in c("learning.test", "asia", "coronary", "lizards", "insurance",
             "alarm", "gaussian.test", "marks")) {
  e = new.env(); data(list = nm, envir = e)
  datasets[[nm]] = get(nm, envir = e)
}

records = character(0)

run = function(dataset, score, tabu = 10, extra = list(), wl = NULL, bl = NULL,
               maxp = Inf) {
  data = datasets[[dataset]]

  argv = c(list(x = data, score = score, tabu = tabu, maxp = maxp), extra)
  if (!is.null(wl)) argv$whitelist = wl
  if (!is.null(bl)) argv$blacklist = bl
  cat("  ", dataset, score, "tabu =", tabu, "\n", file = stderr())
  net = do.call(bnlearn::tabu, argv)

  # the same setup under hc, to record whether tabu went anywhere different.
  hcargv = c(list(x = data, score = score, maxp = maxp), extra)
  if (!is.null(wl)) hcargv$whitelist = wl
  if (!is.null(bl)) hcargv$blacklist = bl
  hcnet = do.call(bnlearn::hc, hcargv)
  same = isTRUE(all.equal(sort(paste(net$arcs[, 1], net$arcs[, 2])),
                          sort(paste(hcnet$arcs[, 1], hcnet$arcs[, 2]))))

  scores = do.call(bnlearn::score,
             c(list(x = net, data = data, type = score, by.node = TRUE), extra))

  extra.json = if (length(extra) == 0) "{}" else
    paste0("{", paste(sprintf('%s:%s', sapply(names(extra), jstr),
                              sapply(extra, num)), collapse = ","), "}")

  records <<- c(records, sprintf(paste0(
      '{"dataset":%s,"score":%s,"tabu":%s,"extra":%s,"whitelist":%s,',
      '"blacklist":%s,"maxp":%s,"modelstring":%s,"arcs":%s,"nodes":%s,',
      '"node.scores":[%s],"same.as.hc":%s}'),
    jstr(dataset), jstr(score), num(tabu), extra.json,
    jarcs(wl), jarcs(bl), if (is.infinite(maxp)) "null" else num(maxp),
    jstr(modelstring(net)), jarcs(net$arcs), jarr(names(net$nodes)),
    paste(sapply(scores, num), collapse = ","), jbool(same)))
}

# --- the tabu list size is the main knob -----------------------------------

for (ds in c("learning.test", "asia", "coronary", "lizards"))
  for (t in c(1, 2, 5, 10, 30))
    run(ds, "bic", tabu = t)

# --- across scores ----------------------------------------------------------

for (ds in c("learning.test", "asia", "coronary", "lizards")) {
  for (sc in c("aic", "loglik", "k2"))
    run(ds, sc)
  run(ds, "bde", extra = list(iss = 1))
  run(ds, "bde", extra = list(iss = 10))
}

for (ds in c("gaussian.test", "marks")) {
  for (sc in c("bic-g", "aic-g", "loglik-g"))
    run(ds, sc)
}

# --- larger networks, where the search has room to wander --------------------

for (ds in c("insurance", "alarm")) {
  run(ds, "bic")
  run(ds, "bde", extra = list(iss = 1))
  run(ds, "bic", tabu = 30)
}

# --- constraints and parent limits ------------------------------------------

wl = data.frame(from = "A", to = "F", stringsAsFactors = FALSE)
bl = data.frame(from = c("A", "C"), to = c("B", "D"), stringsAsFactors = FALSE)
run("learning.test", "bic", wl = wl)
run("learning.test", "bic", bl = bl)
run("learning.test", "bic", wl = wl, bl = bl)
for (mp in c(1, 2, 3))
  run("learning.test", "bic", maxp = mp)
run("asia", "bic", maxp = 2)
run("insurance", "bic", maxp = 3)

# --- non-default penalties --------------------------------------------------

run("learning.test", "bic", extra = list(k = 1))
run("learning.test", "bic", extra = list(k = 5))
run("gaussian.test", "bic-g", extra = list(k = 2))

out = file.path(outdir, "tabu.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)

cat(sprintf("wrote %s: %d reference networks\n", out, length(records)))
cat(sprintf("R %s, bnlearn %s\n", getRversion(),
            as.character(packageVersion("bnlearn"))))
