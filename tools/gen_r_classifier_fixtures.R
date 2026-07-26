#!/usr/bin/env Rscript
# pybnlearn: reference results for the Bayesian network classifiers.
#
# Usage: Rscript tools/gen_r_classifier_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"

num = function(x) sprintf("%.17g", as.numeric(x))
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
jnums = function(v) if (length(v) == 0) "[]" else
                      paste0("[", paste(sapply(v, num), collapse = ","), "]")
# naive.bayes.backend() builds its arc matrix without dimnames, so the columns
# are addressed by position rather than by name.
jarcs = function(a) if (is.null(a) || nrow(a) == 0) "[]" else
                      jarr(paste(a[, 1], a[, 2], sep = ">"))

datasets = list()
for (nm in c("learning.test", "asia", "coronary", "lizards")) {
  e = new.env(); data(list = nm, envir = e); datasets[[nm]] = get(nm, envir = e)
}

records = character(0)

emit.structure = function(dataset, algorithm, training, explanatory, root,
                          extra = list()) {
  d = datasets[[dataset]]
  argv = c(list(x = d, training = training), extra)
  if (!is.null(explanatory)) argv$explanatory = explanatory
  if (!is.null(root)) argv$root = root
  net = do.call(algorithm, argv)

  records <<- c(records, sprintf(paste0(
      '{"kind":"structure","dataset":%s,"algorithm":%s,"training":%s,',
      '"explanatory":%s,"root":%s,"modelstring":%s,"arcs":%s,"nodes":%s}'),
    jstr(dataset), jstr(algorithm), jstr(training),
    if (is.null(explanatory)) "null" else jarr(explanatory),
    if (is.null(root)) "null" else jstr(root),
    jstr(modelstring(net)), jarcs(net$arcs), jarr(names(net$nodes))))
}

emit.predictions = function(dataset, algorithm, training, n) {
  d = datasets[[dataset]]
  net = do.call(algorithm, list(x = d, training = training))
  f = bn.fit(net, d)
  head.d = head(d, n)
  p = predict(f, head.d, prob = TRUE)
  pr = attr(p, "prob")
  rows = sapply(rownames(pr), function(lv)
    sprintf("%s:%s", jstr(lv), jnums(pr[lv, ])))

  records <<- c(records, sprintf(paste0(
      '{"kind":"predict","dataset":%s,"algorithm":%s,"training":%s,"n":%s,',
      '"predicted":%s,"levels":%s,"probabilities":{%s}}'),
    jstr(dataset), jstr(algorithm), jstr(training), num(n),
    jarr(as.character(p)), jarr(rownames(pr)), paste(rows, collapse = ",")))
}

targets = list("learning.test" = c("A", "B", "F"),
               "asia" = c("D", "S", "X"),
               "coronary" = c("Smoking", "Family"),
               "lizards" = c("Species"))

for (ds in names(targets))
  for (target in targets[[ds]])
    for (algo in c("naive.bayes", "tree.bayes")) {
      emit.structure(ds, algo, target, NULL, NULL)
      emit.predictions(ds, algo, target, 300)
    }

# the root changes how the feature tree is oriented.
for (root in c("A", "C", "E"))
  emit.structure("learning.test", "tree.bayes", "F", NULL, root)

# an explicit, restricted set of explanatory variables.
emit.structure("learning.test", "naive.bayes", "A", c("B", "C"), NULL)
emit.structure("learning.test", "tree.bayes", "A", c("B", "C", "D"), NULL)
emit.structure("asia", "tree.bayes", "D", c("A", "S", "T", "L"), NULL)

out = file.path(outdir, "classifiers.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)
cat(sprintf("wrote %s: %d records\n", out, length(records)))
