#!/usr/bin/env Rscript
# pybnlearn: reference results for exact inference, from gRain.
#
# bnlearn does not implement exact inference; it delegates to gRain, and so
# does this fixture generator.  pybnlearn implements it directly, so unlike
# the rest of the suite these fixtures check an independent implementation
# rather than a port -- the numbers are compared, not the algorithm.
#
# Usage: Rscript tools/gen_r_exact_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages({library(bnlearn); library(gRain)})

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"

num = function(x) sprintf("%.17g", as.numeric(x))
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
jnums = function(v) if (length(v) == 0) "[]" else
                      paste0("[", paste(sapply(v, num), collapse = ","), "]")
jmap = function(m) if (length(m) == 0) "{}" else
  paste0("{", paste(sprintf("%s:%s", sapply(names(m), jstr),
                            sapply(m, jstr)), collapse = ","), "}")

# coronary is left out: gRain builds model formulae from the variable names,
# and cannot handle "M. Work" and "P. Work".  pybnlearn has no such
# restriction, so those names are covered by a separate test that checks the
# junction tree against brute-force enumeration instead.
datasets = list()
for (nm in c("learning.test", "asia", "lizards")) {
  e = new.env(); data(list = nm, envir = e); datasets[[nm]] = get(nm, envir = e)
}

structures = list(
  "learning.test" = "[A][C][F][B|A][D|A:C][E|B:F]",
  "asia" = "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
  "lizards" = "[Species][Diameter|Species][Height|Species]"
)

fits = list(); grains = list()
for (nm in names(structures)) {
  fits[[nm]] = bn.fit(model2network(structures[[nm]]), datasets[[nm]])
  grains[[nm]] = compile(as.grain(fits[[nm]]))
}

records = character(0)

emit.query = function(network, nodes, evidence) {
  g = grains[[network]]
  if (length(evidence) > 0)
    g = setEvidence(g, nodes = names(evidence),
                    states = unlist(evidence, use.names = FALSE))

  if (length(nodes) == 1) {
    p = querygrain(g, nodes = nodes)[[nodes]]
    levels = list(names(p))
    values = as.numeric(p)
  } else {
    p = querygrain(g, nodes = nodes, type = "joint")
    # gRain returns the joint with the nodes as dimensions in the order asked
    # for; as.vector() unrolls column-major, matching numpy's order="F".
    levels = dimnames(p)[nodes]
    values = as.vector(aperm(p, nodes))
  }

  records <<- c(records, sprintf(paste0(
      '{"kind":"query","network":%s,"nodes":%s,"evidence":%s,',
      '"levels":[%s],"values":%s}'),
    jstr(network), jarr(nodes), jmap(evidence),
    paste(sapply(levels, jarr), collapse = ","), jnums(values)))
}

# marginals
for (nm in names(structures))
  for (node in names(datasets[[nm]]))
    emit.query(nm, node, list())

# conditionals, with one, two and three pieces of evidence
emit.query("learning.test", "B", list(A = "a"))
emit.query("learning.test", "E", list(A = "b", F = "a"))
emit.query("learning.test", "D", list(A = "a", C = "b", F = "b"))
emit.query("learning.test", "A", list(E = "c"))
emit.query("asia", "D", list(S = "yes"))
emit.query("asia", "D", list(X = "yes", A = "yes"))
emit.query("asia", "T", list(X = "yes", D = "yes", S = "no"))
emit.query("asia", "L", list(X = "no"))
emit.query("lizards", "Species", list(Diameter = "narrow"))

# joints, including variables that do not share a clique
emit.query("learning.test", c("B", "E"), list())
emit.query("learning.test", c("A", "F"), list(E = "c"))
emit.query("learning.test", c("B", "D", "E"), list())
emit.query("asia", c("D", "X"), list(S = "yes"))
emit.query("asia", c("T", "L", "B"), list(D = "yes"))

# --- exact prediction -------------------------------------------------------

for (nm in c("learning.test", "asia", "lizards")) {
  d = head(datasets[[nm]], 150)
  for (node in names(datasets[[nm]])[1:3]) {
    set.seed(21)
    p = predict(fits[[nm]], node, d, method = "exact", prob = TRUE)
    pr = attr(p, "prob")
    rows = sapply(rownames(pr), function(lv)
      sprintf("%s:%s", jstr(lv), jnums(pr[lv, ])))
    records = c(records, sprintf(paste0(
        '{"kind":"exact.predict","network":%s,"node":%s,"n":%s,"seed":%s,',
        '"predicted":%s,"levels":%s,"probabilities":{%s}}'),
      jstr(nm), jstr(node), num(nrow(d)), num(21),
      jarr(as.character(p)), jarr(rownames(pr)), paste(rows, collapse = ",")))
  }
}

out = file.path(outdir, "exact.json")
writeLines(paste0("[\n  ", paste(records, collapse = ",\n  "), "\n]"), out)
cat(sprintf("wrote %s: %d records\n", out, length(records)))
cat(sprintf("gRain %s\n", as.character(packageVersion("gRain"))))
