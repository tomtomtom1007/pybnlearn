#!/usr/bin/env Rscript
# pybnlearn: reference results for arc strengths and network averaging.
#
# The interesting fixture here is the inclusion threshold.  It is not a
# formula but the minimum of a piecewise-linear function found by R's
# optimize(), and the value comes straight back as a cutoff on arc strengths
# -- so two optimisers that agree to six digits can still disagree about
# whether an arc is in the network.  The last section therefore hammers it
# with strength vectors of many shapes, including the degenerate ones.
#
# Usage: Rscript tools/gen_r_strength_fixtures.R [outdir]
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

e = new.env()
for (nm in c("learning.test", "asia", "coronary", "lizards", "gaussian.test",
             "marks"))
  data(list = nm, envir = e)
sets = as.list(e)

# ---------------------------------------------------------------------------
# arc.strength
# ---------------------------------------------------------------------------

structures = list(
  "learning.test" = "[A][C][F][B|A][D|A:C][E|B:F]",
  "asia" = "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
  "coronary" = paste0("[Smoking][M. Work|Smoking][P. Work|Smoking]",
                      "[Pressure|Smoking][Proteins|Smoking:M. Work]",
                      "[Family|M. Work]"),
  "lizards" = "[Species][Diameter|Species][Height|Species]",
  "gaussian.test" = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
  "marks" = "[MECH][VECT|MECH][ALG|MECH:VECT][ANL|ALG][STAT|ALG:ANL]")

criteria = list(
  "learning.test" = list(list(c = "mi"), list(c = "x2"), list(c = "mi-sh"),
                         list(c = "x2-adf"), list(c = "mi-adf"),
                         list(c = "loglik"), list(c = "aic"), list(c = "bic"),
                         list(c = "bde"), list(c = "bde", a = list(iss = 5)),
                         list(c = "bds"), list(c = "k2"), list(c = "bic", a = list(k = 1))),
  "asia" = list(list(c = "mi"), list(c = "bic"), list(c = "bde")),
  "coronary" = list(list(c = "mi"), list(c = "bic")),
  "lizards" = list(list(c = "x2"), list(c = "aic")),
  "gaussian.test" = list(list(c = "cor"), list(c = "zf"), list(c = "mi-g"),
                         list(c = "mi-g-sh"), list(c = "loglik-g"),
                         list(c = "bic-g"), list(c = "aic-g"), list(c = "bge")),
  "marks" = list(list(c = "cor"), list(c = "bic-g")))

for (dname in names(structures)) {
  d = sets[[dname]]
  net = model2network(structures[[dname]])

  for (spec in criteria[[dname]]) {
    res = do.call(arc.strength,
            c(list(x = net, data = d, criterion = spec$c), spec$a))

    add(field("kind", jstr("arc.strength")),
        field("dataset", jstr(dname)),
        field("modelstring", jstr(structures[[dname]])),
        field("criterion", jstr(spec$c)),
        field("args", if (is.null(spec$a)) "{}" else
                paste0("{", paste(sapply(names(spec$a), function(k)
                  paste0(jstr(k), ":", sprintf("%.17g", spec$a[[k]]))),
                  collapse = ","), "}")),
        field("method", jstr(attr(res, "method"))),
        field("threshold", sprintf("%.17g", attr(res, "threshold"))),
        field("from", jarr(res$from)),
        field("to", jarr(res$to)),
        field("strength", num(res$strength)))
  }

  # the default criterion, which comes from the learning algorithm.
  for (algo in c("hc", "gs")) {
    learned = do.call(algo, list(x = d))
    if (!bnlearn:::is.completely.directed(learned))
      next
    res = arc.strength(learned, d)
    add(field("kind", jstr("arc.strength.default")),
        field("dataset", jstr(dname)),
        field("algorithm", jstr(algo)),
        field("modelstring", jstr(modelstring(learned))),
        field("method", jstr(attr(res, "method"))),
        field("from", jarr(res$from)),
        field("to", jarr(res$to)),
        field("strength", num(res$strength)))
  }
}

# ---------------------------------------------------------------------------
# boot.strength's threshold, and the network it averages to
# ---------------------------------------------------------------------------

for (dname in c("learning.test", "asia", "lizards", "gaussian.test")) {
  d = sets[[dname]]

  for (seed in c(1, 42)) {
    for (R in c(20, 50)) {
      set.seed(seed)
      s = boot.strength(d, R = R, algorithm = "hc")

      add(field("kind", jstr("boot")),
          field("dataset", jstr(dname)),
          field("seed", sprintf("%d", seed)),
          field("replicates", sprintf("%d", R)),
          field("from", jarr(s$from)),
          field("to", jarr(s$to)),
          field("strength", num(s$strength)),
          field("direction", num(s$direction)),
          field("threshold", sprintf("%.17g", attr(s, "threshold"))))

      for (th in list(NULL, 0.5, 0.85, 1)) {
        avg = if (is.null(th)) averaged.network(s) else
                averaged.network(s, threshold = th)
        add(field("kind", jstr("averaged")),
            field("dataset", jstr(dname)),
            field("seed", sprintf("%d", seed)),
            field("replicates", sprintf("%d", R)),
            field("threshold", if (is.null(th)) "null" else sprintf("%.17g", th)),
            field("arcs", if (nrow(avg$arcs) == 0) "[]" else
              paste0("[", paste(apply(avg$arcs, 1, function(a)
                paste0("[", jstr(a[1]), ",", jstr(a[2]), "]")),
                collapse = ","), "]")))
      }
    }
  }
}

# ---------------------------------------------------------------------------
# custom.strength
# ---------------------------------------------------------------------------

for (dname in c("learning.test", "asia")) {
  d = sets[[dname]]
  nodes = names(d)

  set.seed(7)
  nets = lapply(1:12, function(i)
           hc(d[sample(nrow(d), nrow(d) / 2, replace = TRUE), , drop = FALSE]))

  for (cp in c(TRUE, FALSE)) {
    for (wname in c("equal", "uneven")) {
      w = if (wname == "equal") NULL else seq_along(nets) / sum(seq_along(nets)) * length(nets)
      s = custom.strength(nets, nodes = nodes, weights = w, cpdag = cp)

      add(field("kind", jstr("custom")),
          field("dataset", jstr(dname)),
          field("cpdag", if (cp) "true" else "false"),
          field("weights", jstr(wname)),
          field("networks", paste0("[", paste(sapply(nets, function(n)
            jstr(modelstring(n))), collapse = ","), "]")),
          field("nodes", jarr(nodes)),
          field("from", jarr(s$from)),
          field("to", jarr(s$to)),
          field("strength", num(s$strength)),
          field("direction", num(s$direction)),
          field("threshold", sprintf("%.17g", attr(s, "threshold"))))
    }
  }
}

# ---------------------------------------------------------------------------
# the inclusion threshold on its own
# ---------------------------------------------------------------------------

set.seed(123)
vectors = list(
  "all-zero" = rep(0, 10),
  "all-one" = rep(1, 10),
  "single" = 0.5,
  "two" = c(0, 1),
  "uniform" = round(runif(30), 4),
  "bimodal" = c(rep(0.02, 15), rep(0.98, 12)),
  "bimodal-uneven" = c(rep(0, 40), rep(1, 3)),
  "ramp" = seq(0, 1, length.out = 21),
  "ties" = c(rep(0.25, 8), rep(0.75, 8), 1, 0),
  "near-half" = round(rbeta(50, 2, 2), 6),
  "spiky" = c(runif(20, 0, 0.1), runif(5, 0.9, 1)),
  "one-plus" = c(rep(1, 5), 0.3, 0.4),
  "descending" = rev(seq(0, 1, by = 0.05)))

for (nm in names(vectors)) {
  v = vectors[[nm]]
  th = bnlearn:::threshold(data.frame(strength = v))
  add(field("kind", jstr("threshold")),
      field("name", jstr(nm)),
      field("strength", num(v)),
      field("threshold", sprintf("%.17g", th)))
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "strength.json"))
cat("wrote", length(records), "records to",
    file.path(outdir, "strength.json"), "\n")
