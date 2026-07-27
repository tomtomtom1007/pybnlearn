#!/usr/bin/env Rscript
# pybnlearn: reference results for the adjacency-matrix accessor.
#
# amat() and `amat<-` are the only place bnlearn describes a whole graph at
# once rather than an arc at a time, so they are the way in and out of
# anything that already speaks adjacency matrices.
#
# The matrix is lossy in one specific way and the fixtures have to cover it:
# an undirected arc is stored as both directions, so it is a symmetric pair
# of ones, and a matrix cannot tell it from two opposed directed arcs.  Both
# a partially directed graph and a fully directed one are here, and the
# round trip through the matrix is recorded rather than assumed.
#
# The element order matters too.  amat2arcs() decides what order the arcs
# come back in, and the arc order decides how a model string reads and how a
# conditional probability table is laid out, so the arcs are recorded in the
# order R produces them rather than sorted.
#
# Usage: Rscript tools/gen_r_amat_fixtures.R [outdir]
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
field = function(name, value) paste0(jstr(name), ":", value)
jarcs = function(a) {
  if (is.null(a) || length(a) == 0) return("[]")
  if (is.null(dim(a))) a = matrix(a, ncol = 2)
  if (nrow(a) == 0) return("[]")
  paste0("[", paste(apply(a, 1, function(r)
    paste0("[", jstr(r[1]), ",", jstr(r[2]), "]")), collapse = ","), "]")
}
# a matrix as a list of rows, so the row/column convention is pinned down
# rather than left to whichever way both sides happen to read it.
jmat = function(m) paste0("[", paste(apply(m, 1, function(r)
         paste0("[", paste(as.integer(r), collapse = ","), "]")),
         collapse = ","), "]")
# the dimnames matter as much as the entries: a matrix labelled with the
# same nodes in a different order describes a different graph from the one
# reading it positionally would give.
jlabels = function(m) if (is.null(rownames(m))) "null" else jarr(rownames(m))

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

e = new.env()
for (nm in c("learning.test", "asia", "gaussian.test"))
  data(list = nm, envir = e)

# ---------------------------------------------------------------------------
# reading the matrix
# ---------------------------------------------------------------------------

graphs = list(
  "chain" = model2network("[A][B|A][C|B][D|C]"),
  "collider" = model2network("[A][B][C|A:B]"),
  "learning.test" = model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
  "empty" = empty.graph(LETTERS[1:4]),
  "complete" = model2network("[A][B|A][C|A:B][D|A:B:C]"),
  "single" = empty.graph("A"))

# a partially directed graph: the CPDAG of a chain leaves every arc
# undirected, which is the case a matrix cannot represent faithfully.
graphs[["cpdag-chain"]] = cpdag(model2network("[A][B|A][C|B][D|C]"))
graphs[["cpdag-collider"]] = cpdag(model2network("[A][B][C|A:B]"))

# one learned from data, so the node order is whatever the data had rather
# than alphabetical.
graphs[["learned-asia"]] = hc(get("asia", envir = e))
graphs[["learned-gaussian"]] = hc(get("gaussian.test", envir = e))

for (name in names(graphs)) {
  g = graphs[[name]]
  m = amat(g)

  add(field("kind", jstr("amat")),
      field("name", jstr(name)),
      field("nodes", jarr(nodes(g))),
      field("arcs", jarcs(arcs(g))),
      field("amat", jmat(m)),
      field("rownames", jarr(rownames(m))),
      field("colnames", jarr(colnames(m))),
      # the arcs the matrix gives back, in the order it gives them
      field("roundtrip", jarcs(bnlearn:::amat2arcs(m, nodes(g)))))
}

# ---------------------------------------------------------------------------
# writing the matrix
# ---------------------------------------------------------------------------

# Each case replaces the arcs of `base` wholesale.  The interesting ones are
# the matrix that is not what amat() would have produced -- a graph built
# from the outside -- and the symmetric entries that come back undirected.
assignments = list(
  list(name = "same", base = "learning.test", from = "learning.test"),
  list(name = "empty", base = "learning.test", from = "empty4"),
  list(name = "reversed", base = "chain", from = "reverse-chain"),
  list(name = "undirected", base = "chain", from = "symmetric"),
  list(name = "complete-dag", base = "chain", from = "upper-triangular"))

matrices = function(nodes) {
  n = length(nodes)
  out = list()
  out[["empty4"]] = matrix(0L, n, n, dimnames = list(nodes, nodes))
  m = matrix(0L, n, n, dimnames = list(nodes, nodes))
  for (i in seq_len(n - 1)) m[i, i + 1] = 1L
  out[["chain-like"]] = m
  out[["reverse-chain"]] = t(m)
  out[["symmetric"]] = m + t(m)
  u = matrix(0L, n, n, dimnames = list(nodes, nodes))
  u[upper.tri(u)] = 1L
  out[["upper-triangular"]] = u
  out
}

bases = list("learning.test" = graphs[["learning.test"]],
             "chain" = graphs[["chain"]])

for (spec in assignments) {
  g = bases[[spec$base]]
  ns = nodes(g)
  m = if (spec$from == "learning.test") amat(g) else matrices(ns)[[spec$from]]

  updated = g
  amat(updated) = m

  add(field("kind", jstr("assign")),
      field("name", jstr(spec$name)),
      field("base", jstr(spec$base)),
      field("nodes", jarr(ns)),
      field("amat", jmat(m)),
      field("labels", jlabels(m)),
      field("arcs", jarcs(arcs(updated))),
      field("modelstring", if (directed(updated) && acyclic(updated))
                             jstr(modelstring(updated)) else "null"),
      field("parents", paste0("{", paste(sapply(ns, function(n)
        paste0(jstr(n), ":", jarr(parents(updated, n)))), collapse = ","), "}")))
}

# a matrix whose rows and columns name the same nodes in a different order
# is rearranged rather than read positionally -- R warns and reorders, and
# reading it positionally would build a different graph.
g = graphs[["learning.test"]]
ns = nodes(g)
shuffled = rev(ns)
m = amat(g)[shuffled, shuffled]

updated = g
suppressWarnings(amat(updated) <- m)

add(field("kind", jstr("assign")),
    field("name", jstr("shuffled-labels")),
    field("base", jstr("learning.test")),
    field("nodes", jarr(ns)),
    field("amat", jmat(m)),
    field("labels", jlabels(m)),
    field("arcs", jarcs(arcs(updated))),
    field("modelstring", jstr(modelstring(updated))),
    field("parents", paste0("{", paste(sapply(ns, function(n)
      paste0(jstr(n), ":", jarr(parents(updated, n)))), collapse = ","), "}")))

# ---------------------------------------------------------------------------
# what is rejected
# ---------------------------------------------------------------------------

g = graphs[["chain"]]
ns = nodes(g)
n = length(ns)

bad = list(
  list(name = "cyclic", cycles = TRUE,
       m = { m = matrix(0L, n, n, dimnames = list(ns, ns))
             for (i in seq_len(n - 1)) m[i, i + 1] = 1L
             m[n, 1] = 1L; m }),
  list(name = "diagonal", cycles = TRUE,
       m = { m = matrix(0L, n, n, dimnames = list(ns, ns)); diag(m) = 1L; m }),
  list(name = "not-binary", cycles = TRUE,
       m = { m = matrix(0L, n, n, dimnames = list(ns, ns)); m[1, 2] = 2L; m }),
  list(name = "wrong-size", cycles = TRUE,
       m = matrix(0L, n + 1, n + 1)),
  list(name = "unknown-labels", cycles = TRUE,
       m = { m = matrix(0L, n, n)
             dimnames(m) = list(c("Z", ns[-1]), c("Z", ns[-1])); m }))

for (spec in bad) {
  updated = g
  message = tryCatch({ amat(updated) = spec$m; NA_character_ },
              error = function(err) conditionMessage(err))

  add(field("kind", jstr("rejected")),
      field("name", jstr(spec$name)),
      field("nodes", jarr(ns)),
      field("amat", jmat(spec$m)),
      field("labels", jlabels(spec$m)),
      field("error", if (is.na(message)) "null" else jstr(message)))
}

# the cycle check can be turned off, and then the cyclic matrix goes in.
updated = g
m = matrix(0L, n, n, dimnames = list(ns, ns))
for (i in seq_len(n - 1)) m[i, i + 1] = 1L
m[n, 1] = 1L
amat(updated, check.cycles = FALSE) = m

add(field("kind", jstr("assign")),
    field("name", jstr("cyclic-unchecked")),
    field("base", jstr("chain")),
    field("nodes", jarr(ns)),
    field("amat", jmat(m)),
    field("labels", jlabels(m)),
    field("arcs", jarcs(arcs(updated))),
    field("modelstring", "null"),
    field("parents", paste0("{", paste(sapply(ns, function(x)
      paste0(jstr(x), ":", jarr(parents(updated, x)))), collapse = ","), "}")))

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "amat.json"))
cat("wrote", length(records), "records to", file.path(outdir, "amat.json"),
    "\n")
