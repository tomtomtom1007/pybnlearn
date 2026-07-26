#!/usr/bin/env Rscript
# pybnlearn: reference results for the node and arc utilities.
#
# These are small functions, and most of them would be hard to get wrong.
# What is worth recording is the *order* they return things in -- R sorts
# node sets by traversal depth and its sort is stable, so ties come back in
# the network's own node order -- and what the five arc operations do when
# the arc is already there in some form, which is where the surprises are:
# set.arc() on an undirected arc orients it rather than adding anything,
# drop.edge() leaves a directed arc alone, and reverse.arc() refuses an
# undirected one.
#
# Usage: Rscript tools/gen_r_nodes_fixtures.R [outdir]
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
jbool = function(b) if (isTRUE(b)) "true" else "false"
field = function(name, value) paste0(jstr(name), ":", value)

jarcs = function(a) {
  if (is.null(a) || nrow(a) == 0) return("[]")
  paste0("[", paste(apply(a, 1, function(r)
    paste0("[", jstr(r[1]), ",", jstr(r[2]), "]")), collapse = ","), "]")
}

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

# A mix of shapes: a DAG, a DAG with a collider chain, a partially directed
# graph, an undirected graph, a graph with an isolated node, and one whose
# undirected part is a four-cycle and so not chordal.
graphs = list(
  "dag" = list(nodes = LETTERS[1:6],
               arcs = rbind(c("A","B"), c("A","D"), c("C","D"), c("B","E"),
                            c("F","E"))),
  "chain" = list(nodes = LETTERS[1:5],
                 arcs = rbind(c("A","B"), c("B","C"), c("C","D"), c("D","E"))),
  "pdag" = list(nodes = LETTERS[1:5],
                arcs = rbind(c("A","B"), c("B","A"), c("B","C"), c("C","B"),
                             c("C","D"), c("E","D"))),
  "undirected" = list(nodes = LETTERS[1:4],
                      arcs = rbind(c("A","B"), c("B","A"), c("B","C"),
                                   c("C","B"), c("A","C"), c("C","A"))),
  "isolated" = list(nodes = LETTERS[1:4],
                    arcs = rbind(c("A","B"), c("B","C"))),
  "square" = list(nodes = LETTERS[1:4],
                  arcs = rbind(c("A","B"), c("B","A"), c("B","C"), c("C","B"),
                               c("C","D"), c("D","C"), c("D","A"), c("A","D"))),
  "empty" = list(nodes = LETTERS[1:3], arcs = NULL),
  "cyclic" = list(nodes = LETTERS[1:4],
                  arcs = rbind(c("A","B"), c("B","C"), c("C","A"),
                               c("C","D"))),
  "collider" = list(nodes = LETTERS[1:7],
                    arcs = rbind(c("A","C"), c("B","C"), c("C","D"),
                                 c("D","E"), c("D","F"), c("E","G"),
                                 c("F","G"))))

build = function(spec) {
  g = empty.graph(spec$nodes)
  if (!is.null(spec$arcs))
    arcs(g, check.cycles = FALSE, check.illegal = FALSE) = spec$arcs
  g
}

for (gname in names(graphs)) {
  g = build(graphs[[gname]])
  nodes = bnlearn::nodes(g)

  # a graph with a directed cycle is not a graph most of these functions
  # accept, so it only contributes the three predicates that can answer.
  if (gname == "cyclic") {

    add(field("kind", jstr("cyclic")),
        field("graph", jstr(gname)),
        field("nodes", jarr(nodes)),
        field("arcs", jarcs(bnlearn::arcs(g))),
        field("narcs", sprintf("%d", narcs(g))),
        field("acyclic", jbool(acyclic(g))),
        field("acyclic.directed", jbool(acyclic(g, directed = TRUE))),
        field("directed", jbool(directed(g))),
        field("valid.dag", jbool(valid.dag(g))),
        field("valid.ug", jbool(valid.ug(g))))
    next

  }#THEN

  add(field("kind", jstr("graph")),
      field("graph", jstr(gname)),
      field("nodes", jarr(nodes)),
      field("arcs", jarcs(bnlearn::arcs(g))),
      field("nnodes", sprintf("%d", nnodes(g))),
      field("narcs", sprintf("%d", narcs(g))),
      field("acyclic", jbool(acyclic(g))),
      field("acyclic.directed", jbool(acyclic(g, directed = TRUE))),
      field("directed", jbool(directed(g))),
      field("valid.dag", jbool(valid.dag(g))),
      field("valid.ug", jbool(valid.ug(g))),
      field("valid.cpdag", jbool(valid.cpdag(g))),
      field("root.nodes", jarr(root.nodes(g))),
      field("leaf.nodes", jarr(leaf.nodes(g))),
      field("isolated.nodes", jarr(isolated.nodes(g))),
      field("directed.arcs", jarcs(directed.arcs(g))),
      field("undirected.arcs", jarcs(undirected.arcs(g))),
      field("compelled.arcs", jarcs(compelled.arcs(g))),
      field("ordering", if (directed(g)) jarr(node.ordering(g)) else "null"))

  for (node in nodes)
    add(field("kind", jstr("node")),
        field("graph", jstr(gname)),
        field("node", jstr(node)),
        field("parents", jarr(parents(g, node))),
        field("children", jarr(children(g, node))),
        field("mb", jarr(mb(g, node))),
        field("nbr1", jarr(nbr(g, node))),
        field("nbr2", jarr(nbr(g, node, max.dist = 2))),
        field("nbr3", jarr(nbr(g, node, max.dist = 3))),
        field("spouses", jarr(spouses(g, node))),
        field("ancestors", jarr(ancestors(g, node))),
        field("descendants", jarr(descendants(g, node))),
        field("degree", sprintf("%d", degree(g, node))),
        field("in.degree", sprintf("%d", in.degree(g, node))),
        field("out.degree", sprintf("%d", out.degree(g, node))),
        field("incoming", jarcs(incoming.arcs(g, node))),
        field("outgoing", jarcs(outgoing.arcs(g, node))),
        field("incident", jarcs(incident.arcs(g, node))))

  # every ordered pair, for path.exists().
  for (a in nodes) for (b in nodes) {
    if (a == b) next
    add(field("kind", jstr("path")),
        field("graph", jstr(gname)),
        field("from", jstr(a)), field("to", jstr(b)),
        field("direct", jbool(path.exists(g, a, b))),
        field("indirect", jbool(path.exists(g, a, b, direct = FALSE))),
        field("underlying", jbool(path.exists(g, a, b,
                underlying.graph = TRUE))))
  }
}

# ---------------------------------------------------------------------------
# arc operations
# ---------------------------------------------------------------------------

ops = list(
  list(op = "set.arc", f = function(g, a, b) set.arc(g, a, b)),
  list(op = "drop.arc", f = function(g, a, b) drop.arc(g, a, b)),
  list(op = "reverse.arc", f = function(g, a, b) reverse.arc(g, a, b)),
  list(op = "set.edge", f = function(g, a, b) set.edge(g, a, b)),
  list(op = "drop.edge", f = function(g, a, b) drop.edge(g, a, b)))

for (gname in c("dag", "pdag", "undirected", "isolated", "empty")) {
  g = build(graphs[[gname]])
  nodes = bnlearn::nodes(g)

  for (spec in ops)
    for (a in nodes) for (b in nodes) {
      if (a == b) next
      result = try(spec$f(g, a, b), silent = TRUE)
      failed = is(result, "try-error")

      add(field("kind", jstr("op")),
          field("graph", jstr(gname)),
          field("op", jstr(spec$op)),
          field("from", jstr(a)), field("to", jstr(b)),
          field("error", jbool(failed)),
          field("arcs", if (failed) "null" else jarcs(bnlearn::arcs(result))))
    }
}

# ---------------------------------------------------------------------------
# node operations
# ---------------------------------------------------------------------------

for (gname in c("dag", "pdag", "isolated")) {
  g = build(graphs[[gname]])
  nodes = bnlearn::nodes(g)

  added = add.node(g, "ZZ")
  add(field("kind", jstr("addnode")),
      field("graph", jstr(gname)),
      field("nodes", jarr(bnlearn::nodes(added))),
      field("arcs", jarcs(bnlearn::arcs(added))))

  for (node in nodes) {
    removed = remove.node(g, node)
    add(field("kind", jstr("removenode")),
        field("graph", jstr(gname)),
        field("node", jstr(node)),
        field("nodes", jarr(bnlearn::nodes(removed))),
        field("arcs", jarcs(bnlearn::arcs(removed))))
  }

  labels = paste0("n", seq_along(nodes))
  renamed = rename.nodes(g, labels)
  add(field("kind", jstr("rename")),
      field("graph", jstr(gname)),
      field("labels", jarr(labels)),
      field("nodes", jarr(bnlearn::nodes(renamed))),
      field("arcs", jarcs(bnlearn::arcs(renamed))))
}

# ---------------------------------------------------------------------------
# blacklists from orderings
# ---------------------------------------------------------------------------

orderings = list(
  list(name = "abc", nodes = c("A", "B", "C")),
  list(name = "six", nodes = LETTERS[1:6]),
  list(name = "one", nodes = "A"))

for (o in orderings)
  add(field("kind", jstr("ordering2blacklist")),
      field("name", jstr(o$name)),
      field("nodes", jarr(o$nodes)),
      field("blacklist", jarcs(ordering2blacklist(o$nodes))))

# from a learned network, where the ordering is the topological one.
e = new.env(); data(learning.test, envir = e)
learned = hc(get("learning.test", envir = e))
add(field("kind", jstr("ordering2blacklist")),
    field("name", jstr("learned")),
    field("nodes", jarr(node.ordering(learned))),
    field("blacklist", jarcs(ordering2blacklist(learned))))

tiers = list(
  list(name = "pairs", tiers = list(c("A", "B"), c("C", "D"))),
  list(name = "mixed", tiers = list("A", c("B", "C"), c("D", "E", "F"))),
  list(name = "singles", tiers = list("A", "B", "C")))

for (t in tiers)
  add(field("kind", jstr("tiers2blacklist")),
      field("name", jstr(t$name)),
      field("tiers", paste0("[", paste(sapply(t$tiers, jarr),
                                       collapse = ","), "]")),
      field("blacklist", jarcs(tiers2blacklist(t$tiers))))

for (s in list(c("A", "B"), c("A", "B", "C"), "A"))
  add(field("kind", jstr("set2blacklist")),
      field("set", jarr(s)),
      field("blacklist", jarcs(set2blacklist(s))))

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "nodes.json"))
cat("wrote", length(records), "records to", file.path(outdir, "nodes.json"),
    "\n")
