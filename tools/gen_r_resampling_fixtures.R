#!/usr/bin/env Rscript
# pybnlearn: reference results for the bootstrap and cross-validation.
#
# Both resample the data, so both are exact comparisons: the replicates only
# agree if the same rows are drawn in the same order from the same generator.
#
# Usage: Rscript tools/gen_r_resampling_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"

# JSON has no literal for Inf or NaN, and these genuinely occur -- a
# cross-validation fold whose Gaussian fit is singular has an infinite loss,
# and R reports it as such -- so they are written as strings and decoded on
# the Python side rather than dropped.
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
jbool = function(b) if (isTRUE(b)) "true" else "false"

datasets = list()
for (nm in c("learning.test", "asia", "coronary", "gaussian.test")) {
  e = new.env(); data(list = nm, envir = e); datasets[[nm]] = get(nm, envir = e)
}

records = character(0)

# --- sample(): the primitive both of them rest on ---------------------------

for (seed in c(1, 42)) {
  for (spec in list(list(10, 10, FALSE), list(100, 30, TRUE),
                    list(6, 6, FALSE), list(5000, 5000, TRUE))) {
    set.seed(seed)
    v = sample(spec[[1]], spec[[2]], replace = spec[[3]])
    records = c(records, sprintf(
      '{"kind":"sample","seed":%s,"n":%s,"k":%s,"replace":%s,"values":%s}',
      num(seed), num(spec[[1]]), num(spec[[2]]), jbool(spec[[3]]),
      jnums(v)))
  }
}

# --- boot.strength ----------------------------------------------------------

for (ds in c("learning.test", "asia", "coronary")) {
  d = datasets[[ds]]
  for (algo in c("hc", "tabu", "gs", "iamb", "mmpc")) {
    for (seed in c(1, 7)) {
      set.seed(seed)
      s = boot.strength(d, algorithm = algo, R = 20)
      records = c(records, sprintf(paste0(
          '{"kind":"boot","dataset":%s,"algorithm":%s,"seed":%s,"R":%s,',
          '"cpdag":true,"shuffle":true,"from":%s,"to":%s,"strength":%s,',
          '"direction":%s}'),
        jstr(ds), jstr(algo), num(seed), num(20),
        jarr(s$from), jarr(s$to), jnums(s$strength), jnums(s$direction)))
    }
  }
}

# cpdag and shuffle switched off, since both change what gets counted.
for (cp in c(TRUE, FALSE)) for (sh in c(TRUE, FALSE)) {
  set.seed(3)
  s = boot.strength(datasets[["learning.test"]], algorithm = "hc", R = 15,
                    cpdag = cp, shuffle = sh)
  records = c(records, sprintf(paste0(
      '{"kind":"boot","dataset":%s,"algorithm":%s,"seed":%s,"R":%s,',
      '"cpdag":%s,"shuffle":%s,"from":%s,"to":%s,"strength":%s,',
      '"direction":%s}'),
    jstr("learning.test"), jstr("hc"), num(3), num(15),
    jbool(cp), jbool(sh),
    jarr(s$from), jarr(s$to), jnums(s$strength), jnums(s$direction)))
}

# --- bn.cv ------------------------------------------------------------------

emit.cv = function(dataset, spec, seed, k, method, m) {
  d = datasets[[dataset]]
  argv = list(data = d, bn = spec, k = k, method = method)
  if (method == "hold-out") argv$m = m
  set.seed(seed)
  r = do.call(bn.cv, argv)
  records <<- c(records, sprintf(paste0(
      '{"kind":"cv","dataset":%s,"bn":%s,"seed":%s,"k":%s,"method":%s,',
      '"m":%s,"mean":%s,"losses":%s}'),
    jstr(dataset), jstr(if (is.character(spec)) spec else modelstring(spec)),
    num(seed), num(k), jstr(method),
    if (method == "hold-out") num(m) else "null",
    num(attr(r, "mean")), jnums(sapply(r, function(f) f$loss))))
}

for (ds in c("learning.test", "asia", "gaussian.test"))
  for (algo in c("hc", "tabu"))
    for (seed in c(1, 5))
      emit.cv(ds, algo, seed, 5, "k-fold", NA)

for (seed in c(1, 5))
  for (k in c(3, 10))
    emit.cv("learning.test", "hc", seed, k, "hold-out", 500)

emit.cv("learning.test", model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
        2, 3, "k-fold", NA)
emit.cv("asia", model2network("[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]"),
        2, 5, "k-fold", NA)
emit.cv("learning.test", "gs", 9, 4, "k-fold", NA)

out = file.path(outdir, "resampling.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)
cat(sprintf("wrote %s: %d records\n", out, length(records)))
