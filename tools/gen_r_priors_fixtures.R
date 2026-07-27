#!/usr/bin/env Rscript
# pybnlearn: reference results for the non-uniform graph priors.
#
# A graph prior says how likely a structure is before the data are seen,
# which is a different thing from the parameter priors the Bayesian scores
# already have.  bnlearn has four, and they differ in more than their
# values:
#
#   uniform   every structure equally likely; the score stays decomposable
#             and score equivalent.
#   vsp       each arc included independently with probability beta; still
#             decomposable, still score equivalent.
#   marginal  a probability on every *pair* of nodes, adjacent or not, so
#             the score stops being decomposable -- but stays score
#             equivalent, because it does not care which way an arc points.
#   cs        Castelo and Siebes: a probability per arc *direction*, so the
#             score is neither decomposable nor score equivalent.
#
# Those two flags are what the fixtures are really testing.  They change how
# the search reuses its cache, and getting either wrong gives a network that
# is wrong in a way no single score comparison would reveal -- which is why
# hc() and tabu() are both here, and why they are expected to disagree with
# each other under the marginal prior.
#
# Usage: Rscript tools/gen_r_priors_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

num = function(x) if (length(x) == 0) "[]" else
       paste0("[", paste(sapply(x, function(v)
         if (is.na(v)) "null" else sprintf("%.17g", v)), collapse = ","), "]")
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
jbool = function(b) if (isTRUE(b)) "true" else "false"
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
data(learning.test, envir = e)
data(asia, envir = e)
data(gaussian.test, envir = e)

# ---------------------------------------------------------------------------
# the completion of a Castelo & Siebes prior
# ---------------------------------------------------------------------------

completions = list(
  list(name = "empty", nodes = LETTERS[1:4],
       from = character(0), to = character(0), prob = numeric(0)),
  list(name = "one", nodes = LETTERS[1:4],
       from = "A", to = "B", prob = 0.8),
  list(name = "both-directions", nodes = LETTERS[1:4],
       from = c("A", "B"), to = c("B", "A"), prob = c(0.6, 0.3)),
  list(name = "several", nodes = LETTERS[1:6],
       from = c("A", "B", "C", "E"), to = c("B", "C", "E", "F"),
       prob = c(0.9, 0.05, 0.5, 0.25)))

for (spec in completions) {
  beta = data.frame(from = spec$from, to = spec$to, prob = spec$prob,
           stringsAsFactors = FALSE)
  completed = bnlearn:::cs.completed.prior(beta, spec$nodes)

  add(field("kind", jstr("completion")),
      field("name", jstr(spec$name)),
      field("nodes", jarr(spec$nodes)),
      field("from", jarr(spec$from)),
      field("to", jarr(spec$to)),
      field("prob", num(spec$prob)),
      field("out.from", jarr(completed$from)),
      field("out.to", jarr(completed$to)),
      field("aid", num(completed$aid)),
      field("fwd", num(completed$fwd)),
      field("bkwd", num(completed$bkwd)))
}

# ---------------------------------------------------------------------------
# scores under each prior
# ---------------------------------------------------------------------------

sets = list(
  "learning.test" = list(data = get("learning.test", envir = e),
                         scores = c("bde", "bds", "bdj"),
                         structures = c("[A][C][F][B|A][D|A:C][E|B:F]",
                                        "[B][C][F][A|B][E|B:F][D|A:C]",
                                        "[A][B][C][D][E][F]")),
  "asia" = list(data = get("asia", envir = e),
                scores = c("bde"),
                structures = c("[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
                               "[A][S][T][L][B][E][X][D]")),
  "gaussian.test" = list(data = get("gaussian.test", envir = e),
                         scores = c("bge"),
                         structures = c("[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
                                        "[A][B][C][D][E][F][G]")))

cs.beta = list(
  "learning.test" = data.frame(
    from = c("A", "B", "C"), to = c("B", "C", "E"),
    prob = c(0.9, 0.05, 0.9), stringsAsFactors = FALSE),
  "asia" = data.frame(from = c("A", "S"), to = c("T", "L"),
    prob = c(0.85, 0.7), stringsAsFactors = FALSE),
  "gaussian.test" = data.frame(from = c("A", "B"), to = c("C", "D"),
    prob = c(0.75, 0.2), stringsAsFactors = FALSE))

for (dname in names(sets)) {
  spec = sets[[dname]]
  d = spec$data

  for (sc in spec$scores) for (ms in spec$structures) {
    net = model2network(ms)

    for (prior in list(
           list(p = "uniform"),
           list(p = "vsp"), list(p = "vsp", b = 0.1), list(p = "vsp", b = 0.6),
           list(p = "marginal"), list(p = "marginal", b = 0.2),
           list(p = "marginal", b = 0.9),
           list(p = "cs"), list(p = "cs", b = cs.beta[[dname]]))) {

      value = if (is.null(prior$b))
                score(net, d, type = sc, prior = prior$p)
              else
                score(net, d, type = sc, prior = prior$p, beta = prior$b)

      add(field("kind", jstr("score")),
          field("dataset", jstr(dname)),
          field("score", jstr(sc)),
          field("modelstring", jstr(ms)),
          field("prior", jstr(prior$p)),
          field("beta", if (is.null(prior$b)) "null"
                        else if (is.data.frame(prior$b)) jstr("frame")
                        else sprintf("%.17g", prior$b)),
          field("value", sprintf("%.17g", value)))
    }
  }
}

# ---------------------------------------------------------------------------
# structure learning under each prior
# ---------------------------------------------------------------------------

for (dname in names(sets)) {
  spec = sets[[dname]]
  d = spec$data

  for (algo in c("hc", "tabu")) for (sc in spec$scores) {
    for (prior in list(
           list(p = "uniform"),
           list(p = "vsp"), list(p = "vsp", b = 0.1), list(p = "vsp", b = 0.6),
           list(p = "marginal"), list(p = "marginal", b = 0.2),
           list(p = "marginal", b = 0.9),
           list(p = "cs"), list(p = "cs", b = cs.beta[[dname]]))) {

      call = list(x = d, score = sc, prior = prior$p)
      if (!is.null(prior$b))
        call$beta = prior$b

      learned = do.call(algo, call)

      add(field("kind", jstr("learn")),
          field("dataset", jstr(dname)),
          field("algorithm", jstr(algo)),
          field("score", jstr(sc)),
          field("prior", jstr(prior$p)),
          field("beta", if (is.null(prior$b)) "null"
                        else if (is.data.frame(prior$b)) jstr("frame")
                        else sprintf("%.17g", prior$b)),
          field("modelstring", jstr(modelstring(learned))),
          field("arcs", jarcs(arcs(learned))))
    }
  }
}

# ---------------------------------------------------------------------------
# the two flags the priors change
# ---------------------------------------------------------------------------

for (sc in c("bde", "bds", "bdj", "bge", "bic", "loglik"))
  for (prior in c("uniform", "vsp", "marginal", "cs")) {
    d = if (sc == "bge") get("gaussian.test", envir = e)
        else get("learning.test", envir = e)
    extra = list(prior = prior)

    add(field("kind", jstr("flags")),
        field("score", jstr(sc)),
        field("prior", jstr(prior)),
        field("equivalent", jbool(
          bnlearn:::is.score.equivalent(sc, d, extra))),
        field("decomposable", jbool(
          bnlearn:::is.score.decomposable(sc, extra))))
  }

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "priors.json"))
cat("wrote", length(records), "records to", file.path(outdir, "priors.json"),
    "\n")
