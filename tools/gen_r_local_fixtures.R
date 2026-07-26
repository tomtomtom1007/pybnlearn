#!/usr/bin/env Rscript
# pybnlearn: reference results for fast.iamb, learn.mb/learn.nbr, and the
# entropy and Kullback-Leibler divergence of a fitted network.
#
# fast.iamb is the interesting one to check.  It reaches the same answer
# IAMB is looking for by a different route -- admitting every candidate that
# looks associated in one pass and letting the backward pass sort them out --
# and it stops speculating when the contingency table would get too sparse
# for an asymptotic test.  That last rule fires on some data sets and not
# others, so the fixtures below span both.
#
# H() and KL() need exact inference over the discrete networks, which R gets
# from gRain; pybnlearn uses its own junction tree, so these compare the two.
#
# Usage: Rscript tools/gen_r_local_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))
suppressMessages(library(gRain))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
field = function(name, value) paste0(jstr(name), ":", value)
jarcs = function(a) {
  if (is.null(a) || nrow(a) == 0) return("[]")
  paste0("[", paste(apply(a, 1, function(r)
    paste0("[", jstr(r[1]), ",", jstr(r[2]), "]")), collapse = ","), "]")
}

# JSON has no Inf or NaN, and both turn up here: the divergence to a network
# that gives an observed configuration probability zero is infinite.
jnum = function(v) {
  if (is.na(v)) return(jstr("NaN"))
  if (is.infinite(v)) return(jstr(if (v > 0) "Inf" else "-Inf"))
  sprintf("%.17g", v)
}

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

e = new.env()
for (nm in c("learning.test", "asia", "coronary", "lizards", "gaussian.test",
             "marks"))
  data(list = nm, envir = e)

# A deliberately sparse data set: few rows, many levels, so that the
# contingency tables get thin enough for fast.iamb's asymptotic guard to
# fire.  On bnlearn's own data sets it never does, and without this the
# fixtures would only be exercising the path fast.iamb shares with iamb.
set.seed(11)
n = 60
sparse = data.frame(
  V1 = factor(sample(letters[1:4], n, TRUE)),
  V2 = factor(sample(letters[1:4], n, TRUE)),
  V3 = factor(sample(letters[1:4], n, TRUE)),
  V4 = factor(sample(letters[1:4], n, TRUE)),
  V5 = factor(sample(letters[1:4], n, TRUE)))
sparse$V2 = factor(ifelse(runif(n) < 0.8, as.character(sparse$V1),
                          as.character(sparse$V2)))
sparse$V3 = factor(ifelse(runif(n) < 0.8, as.character(sparse$V2),
                          as.character(sparse$V3)))
write.csv(sparse, file.path(outdir, "sparse.csv"), row.names = FALSE,
          quote = FALSE)
assign("sparse", sparse, envir = e)


discrete = c("learning.test", "asia", "coronary", "lizards", "sparse")
continuous = c("gaussian.test", "marks")

# ---------------------------------------------------------------------------
# fast.iamb
# ---------------------------------------------------------------------------

for (dname in c(discrete, continuous)) {
  d = get(dname, envir = e)
  tests = if (dname %in% discrete) c("mi", "x2", "mi-sh") else
            c("cor", "zf", "mi-g")

  for (tst in tests)
    for (a in c(0.01, 0.05)) {
      learned = suppressWarnings(fast.iamb(d, test = tst, alpha = a))
      add(field("kind", jstr("fast.iamb")),
          field("dataset", jstr(dname)),
          field("test", jstr(tst)),
          field("alpha", sprintf("%.17g", a)),
          field("arcs", jarcs(arcs(learned))))
    }

  # with constraints, and undirected.
  ns = names(d)
  learned = suppressWarnings(fast.iamb(d, whitelist = matrix(ns[1:2], ncol = 2)))
  add(field("kind", jstr("fast.iamb")),
      field("dataset", jstr(dname)),
      field("test", "null"), field("alpha", "0.05"),
      field("whitelist", jarcs(matrix(ns[1:2], ncol = 2))),
      field("arcs", jarcs(arcs(learned))))

  learned = suppressWarnings(fast.iamb(d, undirected = TRUE))
  add(field("kind", jstr("fast.iamb")),
      field("dataset", jstr(dname)),
      field("test", "null"), field("alpha", "0.05"),
      field("undirected", "true"),
      field("arcs", jarcs(arcs(learned))))
}

# ---------------------------------------------------------------------------
# learn.mb and learn.nbr
# ---------------------------------------------------------------------------

for (dname in c("learning.test", "asia", "gaussian.test")) {
  d = get(dname, envir = e)

  for (node in names(d)) {
    for (m in c("gs", "iamb", "inter.iamb", "iamb.fdr", "fast.iamb"))
      add(field("kind", jstr("mb")),
          field("dataset", jstr(dname)),
          field("node", jstr(node)),
          field("method", jstr(m)),
          field("result", jarr(suppressWarnings(
            learn.mb(d, node, method = m)))))

    for (m in c("mmpc", "si.hiton.pc"))
      add(field("kind", jstr("nbr")),
          field("dataset", jstr(dname)),
          field("node", jstr(node)),
          field("method", jstr(m)),
          field("result", jarr(learn.nbr(d, node, method = m))))
  }
}

# ---------------------------------------------------------------------------
# entropy and Kullback-Leibler divergence
# ---------------------------------------------------------------------------

structures = list(
  "learning.test" = c("[A][C][F][B|A][D|A:C][E|B:F]",
                      "[A][B][C][D][E][F]",
                      "[A][B|A][C|B][D|C][E|D][F|E]"),
  "asia" = c("[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
             "[A][S][T][L][B][E][X][D]"),
  "lizards" = c("[Species][Diameter|Species][Height|Species]",
                "[Species][Diameter][Height]"),
  "gaussian.test" = c("[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
                      "[A][B][C][D][E][F][G]",
                      "[A][B|A][C|B][D|C][E|D][F|E][G|F]"),
  "marks" = c("[MECH][VECT|MECH][ALG|MECH:VECT][ANL|ALG][STAT|ALG:ANL]",
              "[MECH][VECT][ALG][ANL][STAT]"))

for (dname in names(structures)) {
  d = get(dname, envir = e)
  fits = lapply(structures[[dname]],
           function(ms) bn.fit(model2network(ms), d))

  for (i in seq_along(fits))
    add(field("kind", jstr("entropy")),
        field("dataset", jstr(dname)),
        field("modelstring", jstr(structures[[dname]][i])),
        field("value", jnum(H(fits[[i]]))))

  for (i in seq_along(fits)) for (j in seq_along(fits))
    add(field("kind", jstr("kl")),
        field("dataset", jstr(dname)),
        field("p", jstr(structures[[dname]][i])),
        field("q", jstr(structures[[dname]][j])),
        field("value", jnum(KL(fits[[i]], fits[[j]]))))
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "local.json"))
cat("wrote", length(records), "records to", file.path(outdir, "local.json"),
    "\n")
