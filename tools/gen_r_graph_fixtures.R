#!/usr/bin/env Rscript
# pybnlearn: reference results for the graph utilities and the pairwise
# mutual-information learners.
#
# Usage: Rscript tools/gen_r_graph_fixtures.R [outdir]
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
jarcs = function(a) if (is.null(a) || nrow(a) == 0) "[]" else
                      jarr(paste(a[, "from"], a[, "to"], sep = ">"))
num = function(x) sprintf("%.17g", as.numeric(x))

datasets = list()
for (nm in c("learning.test", "asia", "coronary", "lizards",
             "gaussian.test", "marks")) {
  e = new.env(); data(list = nm, envir = e)
  datasets[[nm]] = get(nm, envir = e)
}

records = character(0)
emit = function(...) records <<- c(records, sprintf(...))

# --- graph transformations, on networks learned by hc ----------------------

for (ds in names(datasets)) {
  d = datasets[[ds]]
  net = hc(d)

  emit(paste0('{"kind":"transform","dataset":%s,"modelstring":%s,',
              '"arcs":%s,"cpdag":%s,"skeleton":%s,"moral":%s,"nparams":%s,',
              '"nodes":%s}'),
    jstr(ds), jstr(modelstring(net)), jarcs(net$arcs),
    jarcs(cpdag(net)$arcs),
    jarcs(skeleton(net)$arcs),
    jarcs(moral(net)$arcs),
    num(nparams(net, d)),
    jarr(names(net$nodes)))
}

# --- comparisons between pairs of networks ---------------------------------

pairs = list(
  list("learning.test", "[A][C][F][B|A][D|A:C][E|B:F]", "[A][B][C][D][E][F]"),
  list("learning.test", "[A][C][F][B|A][D|A:C][E|B:F]", "[A][C][F][B|A][D|A:C][E|B:F]"),
  list("learning.test", "[A][C][F][B|A][D|A:C][E|B:F]", "[A][B|A][C][D|A][E|B][F]"),
  list("learning.test", "[A][B|A][C|B][D|C][E|D][F|E]", "[F][E|F][D|E][C|D][B|C][A|B]"),
  list("asia", "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|E:B]", "[A][S][T][L][B][E][X][D]")
)

for (p in pairs) {
  a = model2network(p[[2]])
  b = model2network(p[[3]])
  cmp = compare(a, b)
  emit(paste0('{"kind":"compare","dataset":%s,"target":%s,"current":%s,',
              '"shd":%s,"shd.nocpdag":%s,"hamming":%s,',
              '"tp":%s,"fp":%s,"fn":%s}'),
    jstr(p[[1]]), jstr(p[[2]]), jstr(p[[3]]),
    num(shd(b, a)), num(shd(b, a, cpdag = FALSE)), num(hamming(b, a)),
    num(cmp$tp), num(cmp$fp), num(cmp$fn))
}

# --- model string round trips ----------------------------------------------

for (ms in c("[A][C][F][B|A][D|A:C][E|B:F]",
             "[A][B|A][C|A:B][D|C]",
             "[X][Y|X][Z|X:Y]",
             "[A]")) {
  net = model2network(ms)
  emit('{"kind":"modelstring","input":%s,"output":%s,"arcs":%s,"nodes":%s}',
    jstr(ms), jstr(modelstring(net)), jarcs(net$arcs), jarr(names(net$nodes)))
}

# --- the pairwise learners --------------------------------------------------

for (ds in c("learning.test", "asia", "coronary", "lizards")) {
  d = datasets[[ds]]
  for (est in c("mi")) {
    emit('{"kind":"chow.liu","dataset":%s,"mi":%s,"arcs":%s}',
      jstr(ds), jstr(est), jarcs(chow.liu(d, mi = est)$arcs))
    emit('{"kind":"aracne","dataset":%s,"mi":%s,"arcs":%s}',
      jstr(ds), jstr(est), jarcs(aracne(d, mi = est)$arcs))
  }
}
for (ds in c("gaussian.test", "marks")) {
  d = datasets[[ds]]
  emit('{"kind":"chow.liu","dataset":%s,"mi":%s,"arcs":%s}',
    jstr(ds), jstr("mi-g"), jarcs(chow.liu(d, mi = "mi-g")$arcs))
  emit('{"kind":"aracne","dataset":%s,"mi":%s,"arcs":%s}',
    jstr(ds), jstr("mi-g"), jarcs(aracne(d, mi = "mi-g")$arcs))
}

out = file.path(outdir, "graph.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)
cat(sprintf("wrote %s: %d records\n", out, length(records)))
