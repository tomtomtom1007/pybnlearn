#!/usr/bin/env Rscript
# pybnlearn: reference results for predict() and the prediction-based
# cross-validation losses.
#
# Usage: Rscript tools/gen_r_predict_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"

num = function(x) {
  v = as.numeric(x)
  if (length(v) == 0) return("null")
  if (is.na(v)) return('"NaN"')
  if (is.infinite(v)) return(if (v > 0) '"Inf"' else '"-Inf"')
  sprintf("%.17g", v)
}
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
jnums = function(v) if (length(v) == 0) "[]" else
                      paste0("[", paste(sapply(v, num), collapse = ","), "]")

datasets = list()
for (nm in c("learning.test", "asia", "gaussian.test")) {
  e = new.env(); data(list = nm, envir = e); datasets[[nm]] = get(nm, envir = e)
}

structures = list(
  "learning.test" = "[A][C][F][B|A][D|A:C][E|B:F]",
  "asia" = "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
  "gaussian.test" = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"
)

fits = list()
for (nm in names(structures))
  fits[[nm]] = bn.fit(model2network(structures[[nm]]), datasets[[nm]],
                      method = if (nm == "gaussian.test") "mle-g" else "mle")

records = character(0)

# --- predict() --------------------------------------------------------------

for (nm in names(structures)) {
  d = head(datasets[[nm]], 200)
  discrete = nm != "gaussian.test"

  for (node in names(datasets[[nm]])[1:4]) {
    for (method in c("parents", "bayes-lw")) {
      set.seed(11)
      p = predict(fits[[nm]], node, d, method = method)
      records = c(records, sprintf(paste0(
          '{"kind":"predict","network":%s,"node":%s,"method":%s,"seed":%s,',
          '"n":%s,"values":%s}'),
        jstr(nm), jstr(node), jstr(method), num(11), num(nrow(d)),
        if (discrete) jarr(as.character(p)) else jnums(p)))
    }

    if (discrete) {
      set.seed(11)
      p = predict(fits[[nm]], node, d, prob = TRUE)
      pr = attr(p, "prob")
      # emitted level by level, so the comparison does not depend on how the
      # table happens to be laid out on either side.
      rows = sapply(rownames(pr), function(lv)
        sprintf("%s:%s", jstr(lv), jnums(pr[lv, ])))
      records = c(records, sprintf(paste0(
          '{"kind":"predict.prob","network":%s,"node":%s,"n":%s,',
          '"levels":%s,"probabilities":{%s}}'),
        jstr(nm), jstr(node), num(nrow(d)),
        jarr(rownames(pr)), paste(rows, collapse = ",")))
    }
  }
}

# --- the prediction-based cross-validation losses ---------------------------

emit.cv = function(dataset, loss, target, seed, k, method, m) {
  argv = list(data = datasets[[dataset]], bn = "hc", loss = loss,
              loss.args = list(target = target), k = k, method = method)
  if (method == "hold-out") argv$m = m
  set.seed(seed)
  r = do.call(bn.cv, argv)
  records <<- c(records, sprintf(paste0(
      '{"kind":"cvloss","dataset":%s,"loss":%s,"target":%s,"seed":%s,"k":%s,',
      '"method":%s,"m":%s,"mean":%s}'),
    jstr(dataset), jstr(loss), jstr(target), num(seed), num(k),
    jstr(method), if (method == "hold-out") num(m) else "null",
    num(attr(r, "mean"))))
}

for (seed in c(1, 4)) {
  for (loss in c("pred", "f1"))
    for (spec in list(list("learning.test", "B"), list("asia", "D")))
      emit.cv(spec[[1]], loss, spec[[2]], seed, 5, "k-fold", NA)
  emit.cv("asia", "auroc", "D", seed, 5, "k-fold", NA)
  for (loss in c("cor", "mse"))
    emit.cv("gaussian.test", loss, "B", seed, 5, "k-fold", NA)
}

# hold-out scores each repetition separately and averages, which is a
# different aggregation from k-fold's pooling.
for (loss in c("pred", "cor")) {
  spec = if (loss == "pred") list("learning.test", "B") else
                             list("gaussian.test", "B")
  emit.cv(spec[[1]], loss, spec[[2]], 6, 3, "hold-out", 500)
}

out = file.path(outdir, "predict.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)
cat(sprintf("wrote %s: %d records\n", out, length(records)))
