#!/usr/bin/env Rscript
# pybnlearn: reference results for conditional Gaussian networks.
#
# Two things are recorded here, both for data that mixes factors and numeric
# columns: what the -cg scores make the searches do, and what bn.fit's
# mle-cg backend estimates.
#
# The fitting side is the interesting one, because bn.fit dispatches three
# ways on the same call -- a factor node gets a contingency table however its
# parents look, a numeric node with only numeric parents gets one regression,
# and only a numeric node with at least one factor parent gets the regression
# per configuration that gives these networks their name.  All three appear
# below.
#
# Usage: Rscript tools/gen_r_cg_fixtures.R [outdir]
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

e = new.env()
data(clgaussian.test, envir = e)
cg = get("clgaussian.test", envir = e)

# a second, smaller mixed data set, so nothing depends on the shape of the
# one bnlearn ships.  gen_r_levels.R sources the same file, so the level
# orders it records are the ones written out here.
source(file.path(dirname(sub("^--file=", "", grep("^--file=",
  commandArgs(FALSE), value = TRUE)[1])), "cgsmall.R"))
small = cgsmall

sets = list("clgaussian.test" = cg, "cgsmall" = small)

# write.csv() gives numeric columns 15 significant digits, which is not
# enough to get the same doubles back.  That is harmless for a coefficient,
# but a residual is the difference of two numbers of much larger magnitude,
# so an input rounded in the sixteenth digit moves it in the fourteenth --
# far more than the agreement claimed here.  The numeric columns are written
# out at full precision instead.
exact = function(d) {
  for (col in names(d))
    if (is.numeric(d[[col]]))
      d[[col]] = sprintf("%.17g", d[[col]])
  d
}

write.csv(exact(small), file.path(outdir, "cgsmall.csv"), row.names = FALSE,
          quote = FALSE)
write.csv(exact(cg), file.path(outdir, "clgaussian.test.csv"),
          row.names = FALSE, quote = FALSE)

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

# ---------------------------------------------------------------------------
# structure learning with the conditional Gaussian scores
# ---------------------------------------------------------------------------

for (dname in names(sets)) {
  d = sets[[dname]]
  for (algo in c("hc", "tabu")) {
    for (sc in c("loglik-cg", "aic-cg", "bic-cg", "ebic-cg")) {
      fitted = do.call(algo, list(x = d, score = sc))
      add(paste0(jstr("kind"), ":", jstr("structure")),
          paste0(jstr("dataset"), ":", jstr(dname)),
          paste0(jstr("algorithm"), ":", jstr(algo)),
          paste0(jstr("score"), ":", jstr(sc)),
          paste0(jstr("modelstring"), ":", jstr(modelstring(fitted))),
          paste0(jstr("value"), ":", sprintf("%.17g", score(fitted, d, type = sc))))
    }
  }

  # and the default, which check.score() has to resolve to bic-cg
  fitted = hc(d)
  add(paste0(jstr("kind"), ":", jstr("structure")),
      paste0(jstr("dataset"), ":", jstr(dname)),
      paste0(jstr("algorithm"), ":", jstr("hc")),
      paste0(jstr("score"), ":", "null"),
      paste0(jstr("modelstring"), ":", jstr(modelstring(fitted))),
      paste0(jstr("value"), ":", sprintf("%.17g", score(fitted, d))))
}

# the score of a fixed network, so the comparison does not depend on the
# search finding the same one.
fixed = list(
  "clgaussian.test" = c("[A][B][C][H][D|A:H][F|B:C][E|B:D][G|A:D:E:F]",
                        "[A][B][C][H][D|A][E|B][F|C][G|D:E:F:H]",
                        "[A][B][C][D][E][F][G][H]"),
  "cgsmall" = c("[P][Q][R][S|P:R]", "[P][Q|P][R|Q][S|P:Q:R]"))

for (dname in names(fixed))
  for (ms in fixed[[dname]])
    for (sc in c("loglik-cg", "aic-cg", "bic-cg", "ebic-cg"))
      add(paste0(jstr("kind"), ":", jstr("score")),
          paste0(jstr("dataset"), ":", jstr(dname)),
          paste0(jstr("modelstring"), ":", jstr(ms)),
          paste0(jstr("score"), ":", jstr(sc)),
          paste0(jstr("value"), ":",
                 sprintf("%.17g", score(model2network(ms), sets[[dname]],
                                        type = sc))))

# ---------------------------------------------------------------------------
# parameter learning
# ---------------------------------------------------------------------------

for (dname in names(fixed)) {
  d = sets[[dname]]
  for (ms in fixed[[dname]]) {
    fit = bn.fit(model2network(ms), d, method = "mle-cg")

    for (node in nodes(fit)) {
      entry = fit[[node]]
      cls = class(entry)[1]

      common = c(paste0(jstr("kind"), ":", jstr("fit")),
                 paste0(jstr("dataset"), ":", jstr(dname)),
                 paste0(jstr("modelstring"), ":", jstr(ms)),
                 paste0(jstr("node"), ":", jstr(node)),
                 paste0(jstr("class"), ":", jstr(cls)),
                 paste0(jstr("parents"), ":", jarr(entry$parents)))

      if (cls == "bn.fit.dnode") {
        # as.vector() unrolls column-major, which is numpy's order="F".
        add(common,
            paste0(jstr("dim"), ":",
                   paste0("[", paste(dim(entry$prob), collapse = ","), "]")),
            paste0(jstr("prob"), ":", num(as.vector(entry$prob))))
      } else if (cls == "bn.fit.gnode") {
        add(common,
            paste0(jstr("coefnames"), ":", jarr(names(entry$coefficients))),
            paste0(jstr("coefficients"), ":", num(as.vector(entry$coefficients))),
            paste0(jstr("sd"), ":", num(entry$sd)))
      } else if (cls == "bn.fit.cgnode") {
        add(common,
            paste0(jstr("dparents"), ":", jarr(entry$parents[entry$dparents])),
            paste0(jstr("gparents"), ":", jarr(entry$parents[entry$gparents])),
            paste0(jstr("coefnames"), ":", jarr(rownames(entry$coefficients))),
            paste0(jstr("nconfig"), ":", ncol(entry$coefficients)),
            paste0(jstr("coefficients"), ":", num(as.vector(entry$coefficients))),
            paste0(jstr("sd"), ":", num(as.vector(entry$sd))),
            paste0(jstr("residuals"), ":", num(head(entry$residuals, 20))),
            paste0(jstr("fitted"), ":", num(head(entry$fitted.values, 20))))
      } else {
        stop("unexpected node class ", cls)
      }
    }
  }
}

# ---------------------------------------------------------------------------
# sampling from a fitted conditional Gaussian network
# ---------------------------------------------------------------------------

# The three node types are sampled by three different bits of C, and a
# cgnode's parameters reach it as indices into the node's parents rather
# than by name -- so this is the check that those indices are right.
for (dname in names(fixed)) {
  d = sets[[dname]]
  ms = fixed[[dname]][1]
  fit = bn.fit(model2network(ms), d, method = "mle-cg")

  for (seed in c(1, 42)) {
    set.seed(seed)
    generated = rbn(fit, n = 20)

    columns = sapply(names(generated), function(v)
      paste0(jstr(v), ":",
             if (is.factor(generated[[v]])) jarr(as.character(generated[[v]]))
             else num(generated[[v]])))

    add(paste0(jstr("kind"), ":", jstr("rbn")),
        paste0(jstr("dataset"), ":", jstr(dname)),
        paste0(jstr("modelstring"), ":", jstr(ms)),
        paste0(jstr("seed"), ":", sprintf("%d", seed)),
        paste0(jstr("n"), ":", "20"),
        paste0(jstr("columns"), ":",
               paste0("{", paste(columns, collapse = ","), "}")))
  }
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "cg.json"))
cat("wrote", length(records), "records to", file.path(outdir, "cg.json"), "\n")
