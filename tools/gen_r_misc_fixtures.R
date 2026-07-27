#!/usr/bin/env Rscript
# pybnlearn: reference results for the remaining utilities.
#
# perturb() and the hill-climbing restarts that use it, cextend.all(),
# count.graphs(), alst(), dedup(), bn.boot(), loss() and bf.strength().
#
# Two of these need the random number generator and are therefore compared
# exactly rather than statistically: perturb() draws its operation and its
# arc, and bn.boot() draws its resamples.  The restarts matter most on data
# where hill climbing has somewhere to get stuck, so alarm and insurance are
# in the set rather than only the toy networks.
#
# bf.strength() is the one that needs care for a different reason: a Bayes
# factor between two networks routinely runs past what a double holds, so
# the weights are accumulated in extended precision and only the normalised
# result comes back as a double.
#
# Usage: Rscript tools/gen_r_misc_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressWarnings(suppressMessages({
  library(bnlearn)
  library(gmp)
  library(Rmpfr)
}))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

num = function(x) if (length(x) == 0) "[]" else
       paste0("[", paste(sapply(x, function(v)
         if (is.na(v)) "null" else sprintf("%.17g", v)), collapse = ","), "]")
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
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
for (nm in c("learning.test", "asia", "alarm", "insurance", "gaussian.test",
             "marks"))
  data(list = nm, envir = e)

# ---------------------------------------------------------------------------
# perturb
# ---------------------------------------------------------------------------

graphs = list(
  "learning.test" = "[A][C][F][B|A][D|A:C][E|B:F]",
  "chain" = "[A][B|A][C|B][D|C][E|D]",
  "empty" = "[A][B][C][D]")

for (name in names(graphs)) {
  g = model2network(graphs[[name]])

  for (seed in c(1, 42))
    for (nops in c(1, 3, 6)) {
      set.seed(seed)
      add(field("kind", jstr("perturb")),
          field("graph", jstr(name)),
          field("modelstring", jstr(graphs[[name]])),
          field("seed", sprintf("%d", seed)),
          field("nops", sprintf("%d", nops)),
          field("arcs", jarcs(arcs(perturb(g, nops)))))
    }

  # a restricted set of operations, which changes both what is chosen and
  # how the draws line up.
  for (ops in list(c("set"), c("drop", "reverse"))) {
    set.seed(1)
    add(field("kind", jstr("perturb")),
        field("graph", jstr(name)),
        field("modelstring", jstr(graphs[[name]])),
        field("seed", "1"),
        field("nops", "4"),
        field("ops", jarr(ops)),
        field("arcs", jarcs(arcs(perturb(g, 4, ops = ops)))))
  }
}

# ---------------------------------------------------------------------------
# hill climbing with random restarts
# ---------------------------------------------------------------------------

for (dname in c("learning.test", "asia", "alarm", "insurance",
                "gaussian.test")) {
  d = head(get(dname, envir = e), 300)

  for (seed in c(1, 7))
    for (spec in list(c(3, 1), c(5, 2), c(2, 4))) {
      set.seed(seed)
      learned = hc(d, restart = spec[1], perturb = spec[2])

      add(field("kind", jstr("restart")),
          field("dataset", jstr(dname)),
          field("seed", sprintf("%d", seed)),
          field("restart", sprintf("%d", spec[1])),
          field("perturb", sprintf("%d", spec[2])),
          field("modelstring", jstr(modelstring(learned))),
          field("arcs", jarcs(arcs(learned))))
    }
}

# ---------------------------------------------------------------------------
# all the extensions of an equivalence class
# ---------------------------------------------------------------------------

for (ms in c("[A][B|A][C|B]", "[A][B][C|A:B]", "[A][B|A][C|A][D|B:C]",
             "[A][C][F][B|A][D|A:C][E|B:F]", "[A][B|A][C|B][D|C]")) {
  eq = cpdag(model2network(ms))
  all = cextend.all(eq)
  # a fully directed equivalence class has exactly one extension, and
  # cextend.all() returns it bare rather than in a list of one.  A bn object
  # is itself a list, so is.list() cannot tell the two apart.
  if (is(all, "bn"))
    all = list(all)

  add(field("kind", jstr("cextend.all")),
      field("modelstring", jstr(ms)),
      field("cpdag", jarcs(arcs(eq))),
      field("count", sprintf("%d", length(all))),
      field("extensions", paste0("[", paste(sapply(all,
        function(g) jstr(modelstring(g))), collapse = ","), "]")))
}

# ---------------------------------------------------------------------------
# counting graphs
# ---------------------------------------------------------------------------

for (n in 1:7)
  add(field("kind", jstr("count")),
      field("type", jstr("all-dags")),
      field("nodes", sprintf("%d", n)),
      field("value", jstr(as.character(count.graphs("all-dags", nodes = n)))))

for (n in 1:7)
  add(field("kind", jstr("count")),
      field("type", jstr("dags-given-ordering")),
      field("nodes", sprintf("%d", n)),
      field("value", jstr(as.character(
        count.graphs("dags-given-ordering", nodes = n)))))

for (n in 2:6) for (k in 1:n)
  add(field("kind", jstr("count")),
      field("type", jstr("dags-with-k-roots")),
      field("nodes", sprintf("%d", n)),
      field("k", sprintf("%d", k)),
      field("value", jstr(as.character(
        count.graphs("dags-with-k-roots", nodes = n, k = k)))))

for (n in 2:5) for (r in 0:choose(n, 2))
  add(field("kind", jstr("count")),
      field("type", jstr("dags-with-r-arcs")),
      field("nodes", sprintf("%d", n)),
      field("r", sprintf("%d", r)),
      field("value", jstr(as.character(
        count.graphs("dags-with-r-arcs", nodes = n, r = r)))))

for (ms in c("[A][B|A][C|B]", "[A][B][C|A:B]", "[A][B|A][C|A][D|B:C]"))
  add(field("kind", jstr("count")),
      field("type", jstr("dags-in-equivalence-class")),
      field("modelstring", jstr(ms)),
      field("value", jstr(as.character(count.graphs(
        "dags-in-equivalence-class", eqclass = cpdag(model2network(ms)))))))

# ---------------------------------------------------------------------------
# adjacency lists and deduplication
# ---------------------------------------------------------------------------

for (ms in c("[A][C][F][B|A][D|A:C][E|B:F]", "[A][B|A][C|B]", "[A][B][C]")) {
  g = model2network(ms)
  lists = alst(g)

  add(field("kind", jstr("alst")),
      field("modelstring", jstr(ms)),
      field("nodes", jarr(nodes(g))),
      field("alst", paste0("{", paste(sapply(names(lists), function(n)
        paste0(jstr(n), ":", jarr(lists[[n]]))), collapse = ","), "}")))
}

# A data set with two near-duplicate columns: one an exact multiple of an
# existing variable, one a noisy copy.  The threshold decides how close is
# close enough, so it has to separate them.
set.seed(13)
gt = get("gaussian.test", envir = e)
redundant = gt
redundant$H = redundant$A * 1.0000001
redundant$I = redundant$B + rnorm(nrow(redundant), sd = 2)

exact = redundant
for (col in names(exact))
  exact[[col]] = sprintf("%.17g", exact[[col]])
write.csv(exact, file.path(outdir, "redundant.csv"), row.names = FALSE,
          quote = FALSE)

for (threshold in c(0.5, 0.9, 0.99, 0.999))
  add(field("kind", jstr("dedup")),
      field("threshold", sprintf("%.17g", threshold)),
      field("kept", jarr(names(dedup(redundant, threshold = threshold)))))

# ---------------------------------------------------------------------------
# bootstrap over a custom statistic
# ---------------------------------------------------------------------------

for (dname in c("learning.test", "asia")) {
  d = head(get(dname, envir = e), 500)

  for (seed in c(1, 5)) {
    set.seed(seed)
    counts = unlist(bn.boot(d, statistic = narcs, R = 10, algorithm = "hc"))

    add(field("kind", jstr("boot")),
        field("dataset", jstr(dname)),
        field("seed", sprintf("%d", seed)),
        field("statistic", jstr("narcs")),
        field("values", num(counts)))

    set.seed(seed)
    strings = unlist(bn.boot(d, statistic = modelstring, R = 10,
                       algorithm = "hc"))

    add(field("kind", jstr("boot")),
        field("dataset", jstr(dname)),
        field("seed", sprintf("%d", seed)),
        field("statistic", jstr("modelstring")),
        field("strings", jarr(strings)))
  }
}

# ---------------------------------------------------------------------------
# Bayes factor arc strengths
# ---------------------------------------------------------------------------

for (spec in list(
       list(d = "learning.test", n = 500, ms = "[A][C][F][B|A][D|A:C][E|B:F]",
            score = "bde"),
       list(d = "learning.test", n = 500, ms = "[A][C][F][B|A][D|A:C][E|B:F]",
            score = "bic"),
       list(d = "asia", n = 1000,
            ms = "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]", score = "bde"),
       list(d = "gaussian.test", n = 500,
            ms = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]", score = "bge"))) {

  d = head(get(spec$d, envir = e), spec$n)
  s = bf.strength(model2network(spec$ms), d, score = spec$score)

  add(field("kind", jstr("bf")),
      field("dataset", jstr(spec$d)),
      field("rows", sprintf("%d", spec$n)),
      field("modelstring", jstr(spec$ms)),
      field("score", jstr(spec$score)),
      field("from", jarr(s$from)),
      field("to", jarr(s$to)),
      field("strength", num(s$strength)),
      field("direction", num(s$direction)),
      field("threshold", sprintf("%.17g", attr(s, "threshold"))))
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "misc.json"))
cat("wrote", length(records), "records to", file.path(outdir, "misc.json"),
    "\n")
