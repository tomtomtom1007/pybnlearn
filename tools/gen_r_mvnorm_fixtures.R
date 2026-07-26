#!/usr/bin/env Rscript
# pybnlearn: reference results for Gaussian networks as multivariate normals.
#
# A Gaussian network is a multivariate normal written in a factorised form,
# so exact inference on one is linear algebra rather than message passing.
# bnlearn implements that itself -- unlike the discrete case, which it hands
# to gRain -- so these are genuine parity fixtures: R/mvnorm.R makes choices
# (a pseudoinverse rather than a solve, a patched diagonal rather than an
# error on singular covariances) that a from-scratch implementation would not
# have made, and the point is to reproduce them.
#
# The last section is there for those choices specifically: a network with a
# deterministic root node, whose covariance matrix is singular.
#
# Usage: Rscript tools/gen_r_mvnorm_fixtures.R [outdir]
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
data(gaussian.test, envir = e)
data(marks, envir = e)
sets = list("gaussian.test" = get("gaussian.test", envir = e),
            "marks" = get("marks", envir = e))

structures = list(
  "gaussian.test" = c("[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
                      "[A][B|A][C|B][D|C][E|D][F|E][G|F]",
                      "[A][B][C][D][E][F][G]"),
  "marks" = c("[MECH][VECT|MECH][ALG|MECH:VECT][ANL|ALG][STAT|ALG:ANL]",
              "[MECH][VECT][ALG][ANL][STAT]"))

# ---------------------------------------------------------------------------
# the global distribution, and the factorisation back out of it
# ---------------------------------------------------------------------------

for (dname in names(structures)) {
  d = sets[[dname]]

  for (ms in structures[[dname]]) {
    dag = model2network(ms)
    fitted = bn.fit(dag, d)
    mvn = gbn2mvnorm(fitted)

    add(field("kind", jstr("mvnorm")),
        field("dataset", jstr(dname)),
        field("modelstring", jstr(ms)),
        field("variables", jarr(names(mvn$mu))),
        field("mu", num(as.numeric(mvn$mu))),
        # as.vector() unrolls column-major, which is numpy's order="F".
        field("sigma", num(as.vector(mvn$sigma))))

    # and back again: the same network, read off the joint distribution.
    back = mvnorm2gbn(dag, mu = mvn$mu, sigma = mvn$sigma)
    for (node in nodes(back))
      add(field("kind", jstr("mvnorm2gbn")),
          field("dataset", jstr(dname)),
          field("modelstring", jstr(ms)),
          field("node", jstr(node)),
          field("parents", jarr(back[[node]]$parents)),
          field("coefnames", jarr(names(back[[node]]$coefficients))),
          field("coefficients", num(as.numeric(back[[node]]$coefficients))),
          field("sd", num(back[[node]]$sd)))
  }
}

# ---------------------------------------------------------------------------
# conditioning
# ---------------------------------------------------------------------------

queries = list(
  list(dataset = "gaussian.test",
       ms = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
       to = "F", from = c("A", "B"), value = c(1, 2)),
  list(dataset = "gaussian.test",
       ms = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
       to = c("C", "D"), from = c("B"), value = c(3)),
  list(dataset = "gaussian.test",
       ms = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
       to = c("A", "C", "F"), from = c("B", "D", "E", "G"),
       value = c(2, 9, 3.5, 5)),
  list(dataset = "gaussian.test",
       ms = "[A][B|A][C|B][D|C][E|D][F|E][G|F]",
       to = c("G"), from = c("A"), value = c(0)),
  list(dataset = "marks",
       ms = "[MECH][VECT|MECH][ALG|MECH:VECT][ANL|ALG][STAT|ALG:ANL]",
       to = c("STAT"), from = c("MECH", "VECT"), value = c(50, 60)),
  list(dataset = "marks",
       ms = "[MECH][VECT|MECH][ALG|MECH:VECT][ANL|ALG][STAT|ALG:ANL]",
       to = c("MECH", "VECT"), from = c("STAT"), value = c(40)))

for (q in queries) {
  mvn = gbn2mvnorm(bn.fit(model2network(q$ms), sets[[q$dataset]]))
  mean = bnlearn:::conditional.mvnorm(mu = mvn$mu, sigma = mvn$sigma,
           to = q$to, from = q$from, value = q$value)

  add(field("kind", jstr("conditional")),
      field("dataset", jstr(q$dataset)),
      field("modelstring", jstr(q$ms)),
      field("to", jarr(q$to)),
      field("from", jarr(q$from)),
      field("value", num(q$value)),
      field("mean", num(as.numeric(mean))))
}

# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

for (dname in names(structures)) {
  d = sets[[dname]]

  for (ms in structures[[dname]]) {
    fitted = bn.fit(model2network(ms), d)

    for (node in nodes(fitted)) {
      predicted = predict(fitted, node, head(d, 25), method = "exact")
      add(field("kind", jstr("predict")),
          field("dataset", jstr(dname)),
          field("modelstring", jstr(ms)),
          field("node", jstr(node)),
          field("n", "25"),
          field("values", num(as.numeric(predicted))))
    }

    # and one with an explicit, smaller set of predictors.
    others = setdiff(nodes(fitted), nodes(fitted)[1])
    from = head(others, 2)
    predicted = predict(fitted, nodes(fitted)[1], head(d, 25),
                  method = "exact", from = from)
    add(field("kind", jstr("predict")),
        field("dataset", jstr(dname)),
        field("modelstring", jstr(ms)),
        field("node", jstr(nodes(fitted)[1])),
        field("from", jarr(from)),
        field("n", "25"),
        field("values", num(as.numeric(predicted))))
  }
}

# ---------------------------------------------------------------------------
# a singular covariance matrix
# ---------------------------------------------------------------------------

# X is deterministic, so it has zero variance and zero covariance with
# everything: the joint covariance matrix is singular, and factorising it
# needs the diagonal patch in mvnorm2gbn().
deterministic = custom.fit(model2network("[X][Y|X][Z|Y]"), dist = list(
  X = list(coef = c("(Intercept)" = 4), sd = 0),
  Y = list(coef = c("(Intercept)" = 1, "X" = 2), sd = 1.5),
  Z = list(coef = c("(Intercept)" = 0, "Y" = -1), sd = 0.5)))

mvn = gbn2mvnorm(deterministic)
add(field("kind", jstr("singular")),
    field("modelstring", jstr("[X][Y|X][Z|Y]")),
    field("variables", jarr(names(mvn$mu))),
    field("mu", num(as.numeric(mvn$mu))),
    field("sigma", num(as.vector(mvn$sigma))))

back = mvnorm2gbn(model2network("[X][Y|X][Z|Y]"), mu = mvn$mu,
                  sigma = mvn$sigma)
for (node in nodes(back))
  add(field("kind", jstr("singular.gbn")),
      field("modelstring", jstr("[X][Y|X][Z|Y]")),
      field("node", jstr(node)),
      field("parents", jarr(back[[node]]$parents)),
      field("coefnames", jarr(names(back[[node]]$coefficients))),
      field("coefficients", num(as.numeric(back[[node]]$coefficients))),
      field("sd", num(back[[node]]$sd)))

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "mvnorm.json"))
cat("wrote", length(records), "records to", file.path(outdir, "mvnorm.json"),
    "\n")
