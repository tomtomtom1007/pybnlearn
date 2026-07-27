#!/usr/bin/env Rscript
# pybnlearn: reference results for the causal layer.
#
# Interventions, twin networks and counterfactuals are all graph surgery,
# so what these fixtures pin down is the surgery: which arcs survive, which
# nodes are duplicated, which duplicates are merged away again.  None of it
# is numerical, which means a mistake produces a plausible graph rather than
# a wrong number -- so the node sets and arc sets are compared exactly,
# including the order R lists them in where that is meaningful.
#
# The parameterised half is the interesting one.  A twin network turns each
# node's residual variance into an explicit node feeding *both* copies, so
# the copies are deterministic given their parents and everything random
# about them is shared.  That is what makes them counterfactual rather than
# a second sample, and it shows up in the coefficients.
#
# Usage: Rscript tools/gen_r_causal_fixtures.R [outdir]
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
# from.scm.to.bn() subsets the arc matrix without drop = FALSE, so a result
# with exactly one arc comes back as a length-2 vector rather than a 1x2
# matrix.  That is bnlearn's, not ours; it is normalised here.
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

# A chain, a collider, a diamond and a wider network: the shapes differ in
# how far an intervention propagates and therefore in how many
# counterfactual copies survive the merging.
graphs = list(
  "chain" = "[A][B|A][C|B]",
  "collider" = "[A][B][C|A:B]",
  "diamond" = "[A][B|A][C|A][D|B:C]",
  "learning.test" = "[A][C][F][B|A][D|A:C][E|B:F]")

for (name in names(graphs)) {
  g = model2network(graphs[[name]])

  add(field("kind", jstr("twin")),
      field("graph", jstr(name)),
      field("modelstring", jstr(graphs[[name]])),
      field("nodes", jarr(nodes(twin(g)))),
      field("arcs", jarcs(arcs(twin(g)))))

  # an intervention on every node in turn.
  for (node in nodes(g)) {
    fixed = list("x")
    names(fixed) = node

    add(field("kind", jstr("intervention")),
        field("graph", jstr(name)),
        field("modelstring", jstr(graphs[[name]])),
        field("node", jstr(node)),
        field("nodes", jarr(nodes(intervention(g, fixed)))),
        field("arcs", jarcs(arcs(intervention(g, fixed)))))
  }

  # a counterfactual on every node in turn, with and without merging.
  for (node in nodes(g)) {
    fixed = list("x")
    names(fixed) = paste0(node, ".")

    for (merge in c(TRUE, FALSE)) {
      ctf = counterfactual(g, fixed, merging = merge)

      add(field("kind", jstr("counterfactual")),
          field("graph", jstr(name)),
          field("modelstring", jstr(graphs[[name]])),
          field("node", jstr(node)),
          field("merging", jbool(merge)),
          field("nodes", jarr(nodes(ctf))),
          field("arcs", jarcs(arcs(ctf))))
    }
  }
}

# ---------------------------------------------------------------------------
# the parameterised half
# ---------------------------------------------------------------------------

e = new.env()
data(gaussian.test, envir = e)
data(learning.test, envir = e)

gauss.ms = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"
gfitted = bn.fit(model2network(gauss.ms), get("gaussian.test", envir = e))

dump.fitted = function(fitted, ...) {

  for (node in nodes(fitted)) {
    entry = fitted[[node]]

    if (is(entry, "bn.fit.dnode"))
      add(...,
          field("node", jstr(node)),
          field("class", jstr("dnode")),
          field("parents", jarr(entry$parents)),
          field("children", jarr(entry$children)),
          field("levels", jarr(dimnames(entry$prob)[[1]])),
          field("dim", paste0("[", paste(dim(entry$prob), collapse = ","), "]")),
          field("prob", num(as.vector(entry$prob))))
    else
      add(...,
          field("node", jstr(node)),
          field("class", jstr("gnode")),
          field("parents", jarr(entry$parents)),
          field("children", jarr(entry$children)),
          field("coefnames", jarr(names(entry$coefficients))),
          field("coefficients", num(as.numeric(entry$coefficients))),
          field("sd", num(entry$sd)))
  }

}#DUMP.FITTED

# the twin of a fitted Gaussian network.
tw = twin(gfitted)
add(field("kind", jstr("twin.fitted.nodes")),
    field("modelstring", jstr(gauss.ms)),
    field("nodes", jarr(nodes(tw))))
dump.fitted(tw, field("kind", jstr("twin.fitted")),
            field("modelstring", jstr(gauss.ms)))

# interventions on fitted networks, both kinds.
for (node in c("A", "C", "F")) {
  fixed = list(3.5)
  names(fixed) = node
  m = intervention(gfitted, fixed)

  dump.fitted(m, field("kind", jstr("intervention.fitted")),
              field("modelstring", jstr(gauss.ms)),
              field("fixed", jstr(node)))
}

dfitted = bn.fit(model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
            get("learning.test", envir = e))
for (node in c("B", "D")) {
  fixed = list("b")
  names(fixed) = node
  m = intervention(dfitted, fixed)

  dump.fitted(m, field("kind", jstr("intervention.fitted.discrete")),
              field("modelstring", jstr("[A][C][F][B|A][D|A:C][E|B:F]")),
              field("fixed", jstr(node)))
}

# counterfactuals on a fitted Gaussian network.
for (node in c("C.", "F.")) {
  for (merge in c(TRUE, FALSE)) {
    fixed = list(2.0)
    names(fixed) = node
    ctf = counterfactual(gfitted, fixed, merging = merge)

    add(field("kind", jstr("counterfactual.fitted.nodes")),
        field("modelstring", jstr(gauss.ms)),
        field("fixed", jstr(node)),
        field("merging", jbool(merge)),
        field("nodes", jarr(nodes(ctf))))

    dump.fitted(ctf, field("kind", jstr("counterfactual.fitted")),
                field("modelstring", jstr(gauss.ms)),
                field("fixed", jstr(node)),
                field("merging", jbool(merge)))
  }
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "causal.json"))
cat("wrote", length(records), "records to", file.path(outdir, "causal.json"),
    "\n")
