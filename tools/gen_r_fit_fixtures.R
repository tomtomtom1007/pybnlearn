#!/usr/bin/env Rscript
# pybnlearn: reference parameters from R's bn.fit().
#
# Usage: Rscript tools/gen_r_fit_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

num = function(x) if (length(x) == 0) "null" else
       paste0("[", paste(sapply(x, function(v)
         if (is.na(v)) "null" else sprintf("%.17g", v)), collapse = ","), "]")
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")

datasets = list()
for (nm in c("learning.test", "asia", "coronary", "lizards",
             "gaussian.test", "marks")) {
  e = new.env(); data(list = nm, envir = e)
  datasets[[nm]] = get(nm, envir = e)
}

records = character(0)

emit.discrete = function(dataset, modelstring, method, iss) {
  d = datasets[[dataset]]
  net = model2network(modelstring)
  argv = list(x = net, data = d, method = method)
  if (method == "bayes") argv$iss = iss
  f = do.call(bn.fit, argv)

  nodes = sapply(nodes(net), function(n) {
    p = f[[n]]$prob
    # as.vector() unrolls in column-major order, which is how the values are
    # laid out on the C side too.
    sprintf('{"node":%s,"parents":%s,"dim":%s,"values":%s}',
      jstr(n), jarr(f[[n]]$parents),
      num(if (is.null(dim(p))) length(p) else dim(p)),
      num(as.vector(p)))
  })

  records <<- c(records, sprintf(
    paste0('{"kind":"discrete","dataset":%s,"modelstring":%s,"method":%s,',
           '"iss":%s,"nodes":[%s]}'),
    jstr(dataset), jstr(modelstring), jstr(method),
    if (method == "bayes") sprintf("%.17g", iss) else "null",
    paste(nodes, collapse = ",")))
}

emit.gaussian = function(dataset, modelstring) {
  d = datasets[[dataset]]
  net = model2network(modelstring)
  f = bn.fit(net, d, method = "mle-g")

  nodes = sapply(nodes(net), function(n) {
    sprintf('{"node":%s,"parents":%s,"coefnames":%s,"coefficients":%s,"sd":%s}',
      jstr(n), jarr(f[[n]]$parents), jarr(names(f[[n]]$coefficients)),
      num(unname(f[[n]]$coefficients)), sprintf("%.17g", f[[n]]$sd))
  })

  records <<- c(records, sprintf(
    '{"kind":"gaussian","dataset":%s,"modelstring":%s,"nodes":[%s]}',
    jstr(dataset), jstr(modelstring), paste(nodes, collapse = ",")))
}

discrete = c("learning.test", "asia", "coronary", "lizards")
continuous = c("gaussian.test", "marks")

for (ds in discrete) {
  d = datasets[[ds]]
  for (ms in c(modelstring(hc(d)), modelstring(tabu(d)),
               modelstring(empty.graph(names(d))))) {
    emit.discrete(ds, ms, "mle", NA)
    for (iss in c(1, 5, 20))
      emit.discrete(ds, ms, "bayes", iss)
  }
}

for (ds in continuous) {
  d = datasets[[ds]]
  for (ms in c(modelstring(hc(d)), modelstring(empty.graph(names(d)))))
    emit.gaussian(ds, ms)
}

# a hand-written structure, so the tests do not only ever see what hc found.
emit.discrete("learning.test", "[A][B][C][D|A:B:C][E|D][F|D]", "mle", NA)
emit.discrete("learning.test", "[A][B][C][D|A:B:C][E|D][F|D]", "bayes", 10)
emit.gaussian("gaussian.test", "[A][B][C|A:B][D|C][E][F|C:E][G|A:F]")

out = file.path(outdir, "fit.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)
cat(sprintf("wrote %s: %d records\n", out, length(records)))
