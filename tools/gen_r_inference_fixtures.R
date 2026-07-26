#!/usr/bin/env Rscript
# pybnlearn: reference results for simulation and approximate inference.
#
# Everything here is seeded.  Monte Carlo results only agree between two
# implementations if they consume the same random numbers in the same order,
# so these fixtures are exact comparisons, not statistical ones: any
# disagreement means the sampling diverged, not that the estimate was noisy.
#
# Usage: Rscript tools/gen_r_inference_fixtures.R [outdir]
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

datasets = list()
for (nm in c("learning.test", "asia", "gaussian.test")) {
  e = new.env(); data(list = nm, envir = e); datasets[[nm]] = get(nm, envir = e)
}

fits = list(
  "learning.test" = bn.fit(model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
                           datasets[["learning.test"]]),
  "asia" = bn.fit(model2network("[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]"),
                  datasets[["asia"]]),
  "gaussian.test" = bn.fit(model2network(
      "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"), datasets[["gaussian.test"]],
      method = "mle-g")
)

records = character(0)

# --- rbn: the generated data itself, column by column -----------------------

for (name in names(fits))
  for (seed in c(1, 42, 12345))
    for (n in c(1, 5, 100)) {
      set.seed(seed)
      s = rbn(fits[[name]], n)
      cols = sapply(names(s), function(cn)
        sprintf('%s:%s', jstr(cn),
          if (is.factor(s[[cn]])) jarr(as.character(s[[cn]]))
          else jnums(s[[cn]])))
      records <<- c(records, sprintf(
        '{"kind":"rbn","network":%s,"seed":%s,"n":%s,"columns":{%s}}',
        jstr(name), num(seed), num(n), paste(cols, collapse = ",")))
    }

# --- cpquery: discrete networks, equality conditions ------------------------

emit.query = function(network, event, evidence, method, seed, n,
                      event.expr, evidence.expr, evidence.list) {
  # cpquery() captures its arguments with substitute(), so the call has to be
  # built as a language object rather than passed through a variable.
  # Logic sampling takes an expression for the evidence; likelihood weighting
  # takes a list of the values it pins, so the two are not interchangeable.
  ev = if (method == "lw") evidence.list else evidence.expr
  call = substitute(
    cpquery(FIT, EVENT, EVIDENCE, method = METHOD, n = N),
    list(FIT = fits[[network]], EVENT = event.expr, EVIDENCE = ev,
         METHOD = method, N = n))
  set.seed(seed)
  p = eval(call)
  records <<- c(records, sprintf(paste0(
      '{"kind":"cpquery","network":%s,"method":%s,"seed":%s,"n":%s,',
      '"event":%s,"evidence":%s,"probability":%s}'),
    jstr(network), jstr(method), num(seed), num(n),
    event, evidence, num(p)))
}

# each case gives the JSON encoding and the matching R expression.
cases = list(
  list("learning.test", '{"B":"a"}', '{"A":"a"}',
       quote((B == "a")), quote((A == "a")), list(A = "a")),
  list("learning.test", '{"B":"b"}', '{"A":"c"}',
       quote((B == "b")), quote((A == "c")), list(A = "c")),
  list("learning.test", '{"E":"c"}', '{"A":"b","F":"a"}',
       quote((E == "c")), quote((A == "b" & F == "a")),
       list(A = "b", F = "a")),
  list("learning.test", '{"D":"a"}', '{"C":"b"}',
       quote((D == "a")), quote((C == "b")), list(C = "b")),
  list("learning.test", '{"A":"a"}', 'null',
       quote((A == "a")), TRUE, TRUE),
  list("asia", '{"D":"yes"}', '{"S":"yes"}',
       quote((D == "yes")), quote((S == "yes")), list(S = "yes")),
  list("asia", '{"X":"yes"}', '{"A":"yes","S":"yes"}',
       quote((X == "yes")), quote((A == "yes" & S == "yes")),
       list(A = "yes", S = "yes")),
  list("asia", '{"E":"yes"}', '{"T":"no","L":"no"}',
       quote((E == "yes")), quote((T == "no" & L == "no")),
       list(T = "no", L = "no"))
)

for (case in cases)
  for (method in c("ls", "lw"))
    for (seed in c(1, 99))
      emit.query(case[[1]], case[[2]], case[[3]], method, seed, 10000,
                 case[[4]], case[[5]], case[[6]])

# --- cpdist with likelihood weighting: the weights matter -------------------

for (seed in c(1, 42)) {
  set.seed(seed)
  d = cpdist(fits[["learning.test"]], nodes = c("B", "E"),
             evidence = list(A = "a"), method = "lw", n = 500)
  w = attr(d, "weights")
  records <<- c(records, sprintf(paste0(
      '{"kind":"cpdist","network":%s,"method":"lw","seed":%s,"n":%s,',
      '"nodes":%s,"evidence":%s,"B":%s,"E":%s,"weights":%s}'),
    jstr("learning.test"), num(seed), num(500),
    jarr(c("B", "E")), '{"A":"a"}',
    jarr(as.character(d$B)), jarr(as.character(d$E)), jnums(w)))
}

out = file.path(outdir, "inference.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)
cat(sprintf("wrote %s: %d records\n", out, length(records)))
