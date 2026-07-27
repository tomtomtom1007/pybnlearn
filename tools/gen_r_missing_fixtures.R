#!/usr/bin/env Rscript
# pybnlearn: reference results for incomplete data.
#
# Fitting parameters, imputing missing values, and structural EM.
#
# The three imputation methods differ in what they are allowed to look at,
# and the fixtures are built so that the difference shows.  "parents" uses
# only a node's parents, so it ignores everything else the observation says;
# "bayes-lw" and "exact" condition on the whole observed part of the row.  If
# the gaps were all in root nodes the three would agree and the fixtures
# would prove nothing, so the data below have gaps in nodes with parents,
# nodes with children, and several at once in the same row.
#
# "bayes-lw" draws from R's generator, so it is seeded and compared exactly.
#
# The last section has a latent variable -- never observed at all -- which is
# the case structural EM cannot start from an empty network on.
#
# Usage: Rscript tools/gen_r_missing_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))
suppressMessages(library(gRain))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

num = function(x) if (length(x) == 0) "[]" else
       paste0("[", paste(sapply(x, function(v)
         if (is.na(v)) "null" else sprintf("%.17g", v)), collapse = ","), "]")
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
  paste0("[", paste(sapply(v, function(x)
    if (is.na(x)) "null" else jstr(as.character(x))), collapse = ","), "]")
field = function(name, value) paste0(jstr(name), ":", value)
jarcs = function(a) {
  if (is.null(a) || nrow(a) == 0) return("[]")
  paste0("[", paste(apply(a, 1, function(r)
    paste0("[", jstr(r[1]), ",", jstr(r[2]), "]")), collapse = ","), "]")
}

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

e = new.env()
data(learning.test, envir = e)
data(gaussian.test, envir = e)

# ---------------------------------------------------------------------------
# the incomplete data sets
# ---------------------------------------------------------------------------

# Gaps are punched into nodes of every kind -- a root, a node with parents,
# a node with children -- and rows with several gaps at once are included,
# because that is when the partitioning in the exact method has anything to
# do.
punch = function(d, columns, fractions, seed) {

  set.seed(seed)
  for (i in seq_along(columns)) {
    rows = sample(nrow(d), round(nrow(d) * fractions[i]))
    d[rows, columns[i]] = NA
  }
  d

}#PUNCH

discrete = punch(head(get("learning.test", envir = e), 500),
            c("A", "B", "D", "E"), c(0.1, 0.2, 0.15, 0.1), 7)
continuous = punch(head(get("gaussian.test", envir = e), 400),
              c("A", "C", "F"), c(0.15, 0.2, 0.1), 9)

write.csv(discrete, file.path(outdir, "incomplete.discrete.csv"),
          row.names = FALSE, na = "")
exact.numeric = continuous
for (col in names(exact.numeric))
  exact.numeric[[col]] = ifelse(is.na(exact.numeric[[col]]), NA,
                                sprintf("%.17g", exact.numeric[[col]]))
write.csv(exact.numeric, file.path(outdir, "incomplete.continuous.csv"),
          row.names = FALSE, na = "")

sets = list(
  "discrete" = list(data = discrete,
                    ms = "[A][C][F][B|A][D|A:C][E|B:F]"),
  "continuous" = list(data = continuous,
                      ms = "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"))

# ---------------------------------------------------------------------------
# fitting parameters from incomplete data
# ---------------------------------------------------------------------------

for (name in names(sets)) {
  d = sets[[name]]$data
  fitted = bn.fit(model2network(sets[[name]]$ms), d)

  for (node in nodes(fitted)) {
    entry = fitted[[node]]

    if (is(entry, "bn.fit.dnode"))
      add(field("kind", jstr("fit")),
          field("dataset", jstr(name)),
          field("modelstring", jstr(sets[[name]]$ms)),
          field("node", jstr(node)),
          field("class", jstr("dnode")),
          field("dim", paste0("[", paste(dim(entry$prob), collapse = ","), "]")),
          field("prob", num(as.vector(entry$prob))))
    else
      add(field("kind", jstr("fit")),
          field("dataset", jstr(name)),
          field("modelstring", jstr(sets[[name]]$ms)),
          field("node", jstr(node)),
          field("class", jstr("gnode")),
          field("coefnames", jarr(names(entry$coefficients))),
          field("coefficients", num(as.numeric(entry$coefficients))),
          field("sd", num(entry$sd)))
  }
}

# ---------------------------------------------------------------------------
# imputation
# ---------------------------------------------------------------------------

for (name in names(sets)) {
  d = sets[[name]]$data
  fitted = bn.fit(model2network(sets[[name]]$ms), d)

  # Every method is seeded, not just the sampling one.  A conditional
  # distribution with two equally likely levels is decided by a draw from
  # R's generator, and the fixture data have such ties -- so "parents" and
  # "exact" are only reproducible with the seed fixed too.
  for (m in c("parents", "exact", "bayes-lw")) {
    for (seed in if (m == "bayes-lw") c(1, 42) else c(1)) {

      set.seed(seed)
      if (m == "bayes-lw")
        imputed = impute(fitted, d, method = m, n = 200)
      else
        imputed = impute(fitted, d, method = m)

      columns = sapply(names(imputed), function(v)
        paste0(jstr(v), ":",
               if (is.factor(imputed[[v]])) jarr(as.character(imputed[[v]]))
               else num(imputed[[v]])))

      add(field("kind", jstr("impute")),
          field("dataset", jstr(name)),
          field("modelstring", jstr(sets[[name]]$ms)),
          field("method", jstr(m)),
          field("seed", sprintf("%d", seed)),
          field("n", "200"),
          field("columns", paste0("{", paste(columns, collapse = ","), "}")))
    }
  }
}

# ---------------------------------------------------------------------------
# structural EM
# ---------------------------------------------------------------------------

for (name in names(sets)) {
  d = sets[[name]]$data

  for (m in c("parents", "exact")) {
    for (maximize in c("hc", "tabu")) {
      for (iter in c(1, 3)) {

        set.seed(1)
        learned = structural.em(d, maximize = maximize, impute = m,
                    max.iter = iter)

        add(field("kind", jstr("sem")),
            field("dataset", jstr(name)),
            field("impute", jstr(m)),
            field("maximize", jstr(maximize)),
            field("max.iter", sprintf("%d", iter)),
            field("modelstring", jstr(modelstring(learned))),
            field("arcs", jarcs(arcs(learned))))
      }
    }
  }

  # starting from a network rather than from the empty graph.
  start = model2network(sets[[name]]$ms)
  set.seed(1)
  learned = structural.em(d, impute = "parents", start = start, max.iter = 2)
  add(field("kind", jstr("sem")),
      field("dataset", jstr(name)),
      field("impute", jstr("parents")),
      field("maximize", jstr("hc")),
      field("max.iter", "2"),
      field("start", jstr(sets[[name]]$ms)),
      field("modelstring", jstr(modelstring(learned))),
      field("arcs", jarcs(arcs(learned))))
}

# ---------------------------------------------------------------------------
# a latent variable
# ---------------------------------------------------------------------------

# C is never observed, so its distribution cannot be estimated from the data
# and structural EM has to be given one.
latent = head(get("learning.test", envir = e), 500)
latent$C = factor(NA, levels = levels(latent$C))
write.csv(latent, file.path(outdir, "latent.csv"), row.names = FALSE, na = "")

seed.fit = bn.fit(model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
             head(get("learning.test", envir = e), 500))

set.seed(1)
learned = structural.em(latent, impute = "exact", start = seed.fit,
            max.iter = 2)
add(field("kind", jstr("latent")),
    field("modelstring", jstr(modelstring(learned))),
    field("arcs", jarcs(arcs(learned))))

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "missing.json"))
cat("wrote", length(records), "records to", file.path(outdir, "missing.json"),
    "\n")
