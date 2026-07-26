#!/usr/bin/env Rscript
# pybnlearn: reference results for custom.fit().
#
# custom.fit() has no data to check against, so the thing that can go wrong
# is the bookkeeping: which axis of a table is the node and which are its
# parents, which order the parents come in, which configuration a column of
# conditional Gaussian coefficients belongs to.  Getting any of those wrong
# produces a network that looks fine and samples from a different
# distribution.
#
# So the fixtures below do not stop at the parameters.  Each hand-built
# network is also sampled from with a fixed seed, and the generated
# observations are compared row by row -- which only agree if every axis
# went where it was meant to.
#
# Usage: Rscript tools/gen_r_custom_fixtures.R [outdir]
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
field = function(name, value) paste0(jstr(name), ":", value)

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

# ---------------------------------------------------------------------------
# the networks, built by hand
# ---------------------------------------------------------------------------

# a discrete network: a root, a node with one parent, and a node with two,
# so that the axis ordering is actually exercised.
disc.ms = "[A][B|A][C|A:B]"
A = array(c(0.3, 0.7), dim = 2, dimnames = list(A = c("a1", "a2")))
B = array(c(0.2, 0.8, 0.6, 0.4), dim = c(2, 2),
      dimnames = list(B = c("b1", "b2"), A = c("a1", "a2")))
C = array(c(0.1, 0.4, 0.5,
            0.5, 0.3, 0.2,
            0.2, 0.2, 0.6,
            0.7, 0.2, 0.1), dim = c(3, 2, 2),
      dimnames = list(C = c("c1", "c2", "c3"), A = c("a1", "a2"),
                      B = c("b1", "b2")))
discrete = custom.fit(model2network(disc.ms), dist = list(A = A, B = B, C = C))

# a Gaussian network, including a deterministic root.
gauss.ms = "[X][W][Y|X:W][Z|Y]"
gaussian = custom.fit(model2network(gauss.ms), dist = list(
  X = list(coef = c("(Intercept)" = 4), sd = 0),
  W = list(coef = c("(Intercept)" = -1), sd = 2),
  Y = list(coef = c("(Intercept)" = 1, "X" = 2, "W" = 0.5), sd = 1.5),
  Z = list(coef = c("(Intercept)" = 0, "Y" = -1), sd = 0.5)))

# a conditional Gaussian network: G has one discrete and one continuous
# parent, H has two discrete parents and so four configurations.
cg.ms = "[D][E][F][G|D:F][H|D:E:G]"
D = array(c(0.4, 0.6), dim = 2, dimnames = list(D = c("d1", "d2")))
E = array(c(0.25, 0.35, 0.4), dim = 3,
      dimnames = list(E = c("e1", "e2", "e3")))
Fd = list(coef = c("(Intercept)" = 2), sd = 1)
G = list(coef = matrix(c(1, 0.5, 3, -0.5), nrow = 2, ncol = 2,
           dimnames = list(c("(Intercept)", "F"), NULL)),
         sd = c(0.5, 1.5))
H = list(coef = matrix(c(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6),
           nrow = 2, ncol = 6,
           dimnames = list(c("(Intercept)", "G"), NULL)),
         sd = c(0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
cg = custom.fit(model2network(cg.ms), dist = list(
  D = D, E = E, F = Fd, G = G, H = H))

networks = list(
  "discrete" = list(fit = discrete, ms = disc.ms),
  "gaussian" = list(fit = gaussian, ms = gauss.ms),
  "cg" = list(fit = cg, ms = cg.ms))

for (name in names(networks)) {
  f = networks[[name]]$fit

  for (node in nodes(f)) {
    entry = f[[node]]
    cls = class(entry)[1]

    common = c(field("kind", jstr("node")),
               field("network", jstr(name)),
               field("modelstring", jstr(networks[[name]]$ms)),
               field("node", jstr(node)),
               field("class", jstr(cls)),
               field("parents", jarr(entry$parents)),
               field("children", jarr(entry$children)))

    if (cls == "bn.fit.dnode")
      add(common,
          field("dim", paste0("[", paste(dim(entry$prob), collapse = ","), "]")),
          field("dimnames", paste0("[", paste(sapply(dimnames(entry$prob), jarr),
                                              collapse = ","), "]")),
          field("varnames", jarr(names(dimnames(entry$prob)))),
          # as.vector() unrolls column-major, which is numpy's order="F".
          field("prob", num(as.vector(entry$prob))))
    else if (cls == "bn.fit.gnode")
      add(common,
          field("coefnames", jarr(names(entry$coefficients))),
          field("coefficients", num(as.numeric(entry$coefficients))),
          field("sd", num(entry$sd)))
    else
      add(common,
          field("dparents", jarr(entry$parents[entry$dparents])),
          field("gparents", jarr(entry$parents[entry$gparents])),
          field("coefnames", jarr(rownames(entry$coefficients))),
          field("nconfig", sprintf("%d", ncol(entry$coefficients))),
          field("coefficients", num(as.vector(entry$coefficients))),
          field("sd", num(as.numeric(entry$sd))),
          field("dlevels", paste0("{", paste(sapply(names(entry$dlevels),
            function(p) paste0(jstr(p), ":", jarr(entry$dlevels[[p]]))),
            collapse = ","), "}")))
  }

  # bn.net() drops the parameters and leaves the structure.
  add(field("kind", jstr("bn.net")),
      field("network", jstr(name)),
      field("modelstring", jstr(modelstring(bn.net(f)))),
      field("nodes", jarr(nodes(f))))
}

# ---------------------------------------------------------------------------
# sampling from them, which is what actually pins the bookkeeping down
# ---------------------------------------------------------------------------

for (name in names(networks)) {
  f = networks[[name]]$fit

  for (seed in c(1, 42)) {
    set.seed(seed)
    generated = rbn(f, n = 20)

    columns = sapply(names(generated), function(v)
      paste0(jstr(v), ":",
             if (is.factor(generated[[v]]))
               jarr(as.character(generated[[v]]))
             else num(generated[[v]])))

    add(field("kind", jstr("rbn")),
        field("network", jstr(name)),
        field("seed", sprintf("%d", seed)),
        field("n", "20"),
        field("columns", paste0("{", paste(columns, collapse = ","), "}")))
  }
}

# exact inference on the hand-built networks, which reads the tables along a
# different path than sampling does.
for (q in list(list(n = "discrete", nodes = c("C"), evidence = NULL),
               list(n = "discrete", nodes = c("C"),
                    evidence = list(A = "a2")),
               list(n = "discrete", nodes = c("A", "B"),
                    evidence = list(C = "c3")))) {
  f = networks[[q$n]]$fit
  # brute force, to avoid depending on gRain for these.
  joint = as.table(array(0, dim = c(2, 2, 3),
            dimnames = list(A = c("a1", "a2"), B = c("b1", "b2"),
                            C = c("c1", "c2", "c3"))))
  for (a in c("a1", "a2")) for (b in c("b1", "b2")) for (cc in c("c1", "c2", "c3"))
    joint[a, b, cc] = f$A$prob[a] * f$B$prob[b, a] * f$C$prob[cc, a, b]

  keep = joint
  if (!is.null(q$evidence)) {
    idx = lapply(names(dimnames(joint)), function(v)
            if (v %in% names(q$evidence)) q$evidence[[v]]
            else dimnames(joint)[[v]])
    names(idx) = names(dimnames(joint))
    keep = do.call("[", c(list(joint), idx, list(drop = FALSE)))
  }

  marginal = apply(keep, q$nodes, sum)
  marginal = marginal / sum(marginal)

  add(field("kind", jstr("query")),
      field("network", jstr(q$n)),
      field("nodes", jarr(q$nodes)),
      field("evidence", if (is.null(q$evidence)) "{}" else
        paste0("{", paste(sapply(names(q$evidence), function(v)
          paste0(jstr(v), ":", jstr(q$evidence[[v]]))), collapse = ","), "}")),
      field("values", num(as.vector(marginal))))
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "custom.json"))
cat("wrote", length(records), "records to", file.path(outdir, "custom.json"),
    "\n")
