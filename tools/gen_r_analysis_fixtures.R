#!/usr/bin/env Rscript
# pybnlearn: reference results for d-separation, colliders, extensions,
# structural intervention distance, random graph generation, discretization
# and the Bayesian score utilities.
#
# Two of these are worth more attention than the rest.  random.graph() draws
# from R's generator, so its fixtures are compared exactly rather than
# statistically -- the same seed has to give the same graphs.  And
# discretize(method = "hartemink") chooses the breakpoints of every variable
# jointly, so a single wrong collapse changes every column; the fixtures
# record the levels, not just the shape.
#
# Usage: Rscript tools/gen_r_analysis_fixtures.R [outdir]
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
jtriples = function(a) {
  if (is.null(a) || nrow(a) == 0) return("[]")
  paste0("[", paste(apply(a, 1, function(r)
    paste0("[", jstr(r[1]), ",", jstr(r[2]), ",", jstr(r[3]), "]")),
    collapse = ","), "]")
}
jarcs = function(a) {
  if (is.null(a) || nrow(a) == 0) return("[]")
  paste0("[", paste(apply(a, 1, function(r)
    paste0("[", jstr(r[1]), ",", jstr(r[2]), "]")), collapse = ","), "]")
}

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

e = new.env()
for (nm in c("learning.test", "asia", "gaussian.test", "marks", "lizards"))
  data(list = nm, envir = e)

graphs = list(
  "learning.test" = model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
  "asia" = model2network("[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]"),
  "chain" = model2network("[A][B|A][C|B][D|C][E|D]"),
  "collider" = model2network("[A][B][C|A:B][D|C][E|C]"),
  "diamond" = model2network("[A][B|A][C|A][D|B:C]"),
  # A -> C and B -> C with A -> B as well: the collider at C is shielded,
  # so it says nothing about the equivalence class.
  "shielded" = model2network("[A][B|A][C|A:B][D|C]"))

# ---------------------------------------------------------------------------
# colliders and v-structures
# ---------------------------------------------------------------------------

for (name in names(graphs)) {
  g = graphs[[name]]

  add(field("kind", jstr("colliders")),
      field("graph", jstr(name)),
      field("modelstring", jstr(modelstring(g))),
      field("nodes", jarr(nodes(g))),
      field("all", jtriples(colliders(g))),
      field("unshielded", jtriples(unshielded.colliders(g))),
      field("shielded", jtriples(shielded.colliders(g))),
      field("all.arcs", jarcs(colliders(g, arcs = TRUE))),
      field("unshielded.arcs", jarcs(unshielded.colliders(g, arcs = TRUE))),
      field("shielded.arcs", jarcs(shielded.colliders(g, arcs = TRUE))))

  # every triple, for d-separation.
  ns = nodes(g)
  for (x in ns) for (y in ns) {
    if (x >= y) next
    others = setdiff(ns, c(x, y))

    # the empty conditioning set, every single node, and one pair.
    sets = c(list(character(0)), lapply(others, function(z) z))
    if (length(others) >= 2)
      sets = c(sets, list(others[1:2]))

    for (z in sets)
      add(field("kind", jstr("dsep")),
          field("graph", jstr(name)),
          field("x", jstr(x)), field("y", jstr(y)),
          field("z", jarr(z)),
          field("dsep", jbool(dsep(g, x, y, z))))
  }

  # the extension of the equivalence class.
  add(field("kind", jstr("cextend")),
      field("graph", jstr(name)),
      field("modelstring", jstr(modelstring(g))),
      field("cpdag", jarcs(arcs(cpdag(g)))),
      field("extended", jstr(modelstring(cextend(cpdag(g))))))
}

# ---------------------------------------------------------------------------
# structural intervention distance
# ---------------------------------------------------------------------------

pairs = list(
  c("learning.test", "learning.test"), c("chain", "collider"),
  c("collider", "chain"), c("diamond", "diamond"))

for (p in pairs) {
  if (!setequal(nodes(graphs[[p[1]]]), nodes(graphs[[p[2]]]))) next
  add(field("kind", jstr("sid")),
      field("learned", jstr(p[1])), field("true", jstr(p[2])),
      field("value", sprintf("%d", as.integer(sid(graphs[[p[1]]],
                                                  graphs[[p[2]]])))))
}

# and against perturbed versions of the same graph, which is the case that
# matters: two graphs one arc apart can be very differently useful.
base = graphs[["learning.test"]]
for (arc in list(c("A", "B"), c("C", "D"), c("B", "E"))) {
  dropped = drop.arc(base, arc[1], arc[2])
  reversed = reverse.arc(base, arc[1], arc[2])

  add(field("kind", jstr("sid")),
      field("learned", jstr(paste0("drop-", arc[1], arc[2]))),
      field("true", jstr("learning.test")),
      field("arcs", jarcs(arcs(dropped))),
      field("value", sprintf("%d", as.integer(sid(dropped, base)))))
  add(field("kind", jstr("sid")),
      field("learned", jstr(paste0("reverse-", arc[1], arc[2]))),
      field("true", jstr("learning.test")),
      field("arcs", jarcs(arcs(reversed))),
      field("value", sprintf("%d", as.integer(sid(reversed, base)))))
}

# ---------------------------------------------------------------------------
# generating graphs
# ---------------------------------------------------------------------------

for (seed in c(1, 42)) {
  for (spec in list(
         list(m = "ordered", n = LETTERS[1:6], num = 3, extra = list()),
         list(m = "ordered", n = LETTERS[1:8], num = 1,
              extra = list(prob = 0.3)),
         list(m = "ic-dag", n = LETTERS[1:6], num = 2,
              extra = list(burn.in = 30)),
         list(m = "melancon", n = LETTERS[1:5], num = 2,
              extra = list(burn.in = 20, every = 2)),
         list(m = "ic-dag", n = LETTERS[1:6], num = 1,
              extra = list(burn.in = 10, max.in.degree = 2)))) {

    set.seed(seed)
    generated = do.call(random.graph,
                  c(list(nodes = spec$n, num = spec$num, method = spec$m),
                    spec$extra))
    if (spec$num == 1)
      generated = list(generated)

    add(field("kind", jstr("random")),
        field("method", jstr(spec$m)),
        field("seed", sprintf("%d", seed)),
        field("num", sprintf("%d", spec$num)),
        field("nodes", jarr(spec$n)),
        field("extra", paste0("{", paste(sapply(names(spec$extra),
          function(k) paste0(jstr(k), ":", sprintf("%.17g", spec$extra[[k]]))),
          collapse = ","), "}")),
        field("graphs", paste0("[", paste(sapply(generated,
          function(g) jarcs(arcs(g))), collapse = ","), "]")))
  }
}

for (n in list(LETTERS[1:3], LETTERS[1:5]))
  add(field("kind", jstr("complete")),
      field("nodes", jarr(n)),
      field("arcs", jarcs(arcs(complete.graph(n)))))

# ---------------------------------------------------------------------------
# discretization
# ---------------------------------------------------------------------------

for (dname in c("gaussian.test", "marks")) {
  d = get(dname, envir = e)

  for (spec in list(list(m = "quantile", b = 3), list(m = "quantile", b = 5),
                    list(m = "interval", b = 3), list(m = "interval", b = 4),
                    list(m = "hartemink", b = 3, i = "quantile", ib = 30),
                    list(m = "hartemink", b = 2, i = "interval", ib = 20))) {

    if (is.null(spec$i))
      out = discretize(d, method = spec$m, breaks = spec$b)
    else
      out = discretize(d, method = spec$m, breaks = spec$b, idisc = spec$i,
              ibreaks = spec$ib)

    columns = sapply(names(out), function(v)
      paste0(jstr(v), ":", jarr(as.character(head(out[[v]], 30)))))
    levels = sapply(names(out), function(v)
      paste0(jstr(v), ":", jarr(levels(out[[v]]))))

    add(field("kind", jstr("discretize")),
        field("dataset", jstr(dname)),
        field("method", jstr(spec$m)),
        field("breaks", sprintf("%d", spec$b)),
        field("idisc", if (is.null(spec$i)) "null" else jstr(spec$i)),
        field("ibreaks", if (is.null(spec$ib)) "null" else
                sprintf("%d", spec$ib)),
        field("levels", paste0("{", paste(levels, collapse = ","), "}")),
        field("head", paste0("{", paste(columns, collapse = ","), "}")))
  }
}

# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------

lt = get("learning.test", envir = e)
for (spec in list(list(v = c("A", "B"), all = TRUE),
                  list(v = c("A", "B"), all = FALSE),
                  list(v = c("A", "B", "C"), all = TRUE),
                  list(v = c("B"), all = TRUE)))
  add(field("kind", jstr("configs")),
      field("variables", jarr(spec$v)),
      field("all", jbool(spec$all)),
      field("levels", jarr(levels(configs(lt[, spec$v, drop = FALSE],
              all = spec$all)))),
      field("head", jarr(as.character(head(configs(
              lt[, spec$v, drop = FALSE], all = spec$all), 40)))))

# ---------------------------------------------------------------------------
# the Bayesian score utilities
# ---------------------------------------------------------------------------

for (dname in c("learning.test", "asia", "lizards")) {
  d = get(dname, envir = e)
  learned = hc(d)

  add(field("kind", jstr("alpha.star")),
      field("dataset", jstr(dname)),
      field("modelstring", jstr(modelstring(learned))),
      field("value", sprintf("%.17g", alpha.star(learned, d))))

  for (arc in list(arcs(learned)[1, ])) {
    other = drop.arc(learned, arc[1], arc[2])
    for (sc in c("bde", "bic")) {
      add(field("kind", jstr("bf")),
          field("dataset", jstr(dname)),
          field("modelstring", jstr(modelstring(learned))),
          field("dropped", jarr(arc)),
          field("score", jstr(sc)),
          field("log", jbool(TRUE)),
          field("value", sprintf("%.17g", BF(learned, other, d, score = sc))))
    }
  }
}

# identifiable() and singular(), including a network built to be singular.
for (dname in c("learning.test", "asia")) {
  d = get(dname, envir = e)
  f = bn.fit(hc(d), d)
  add(field("kind", jstr("fitprops")),
      field("dataset", jstr(dname)),
      field("modelstring", jstr(modelstring(bn.net(f)))),
      field("identifiable", jbool(identifiable(f))),
      field("singular", jbool(singular(f))))
}

deterministic = custom.fit(model2network("[A][B|A]"), dist = list(
  A = array(c(0.5, 0.5), dim = 2, dimnames = list(A = c("a", "b"))),
  B = array(c(1, 0, 0, 1), dim = c(2, 2),
        dimnames = list(B = c("x", "y"), A = c("a", "b")))))
add(field("kind", jstr("fitprops")),
    field("dataset", jstr("deterministic")),
    field("modelstring", jstr("[A][B|A]")),
    field("identifiable", jbool(identifiable(deterministic))),
    field("singular", jbool(singular(deterministic))))

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "analysis.json"))
cat("wrote", length(records), "records to",
    file.path(outdir, "analysis.json"), "\n")
