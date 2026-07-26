#!/usr/bin/env Rscript
# pybnlearn: record each fixture data set's factor levels, in R's order.
#
# write.csv() loses the level order, and pandas re-derives categories
# alphabetically when reading.  For most of the suite that is harmless --
# mutual information and the network scores do not care which level is first
# -- but a conditional probability table is indexed by level, so a data set
# whose levels are not alphabetical (bnlearn's lizards is one: Sagrei comes
# before Distichus) would silently transpose.
#
# Usage: Rscript tools/gen_r_levels.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"

jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) paste0("[", paste(sapply(v, jstr), collapse = ","), "]")

sets = c("learning.test", "asia", "coronary", "lizards", "insurance",
         "alarm", "gaussian.test", "marks")

out = character(0)
for (nm in sets) {
  e = new.env(); data(list = nm, envir = e); d = get(nm, envir = e)
  cols = character(0)
  for (col in names(d))
    if (is.factor(d[[col]]))
      cols = c(cols, sprintf("%s:%s", jstr(col), jarr(levels(d[[col]]))))
  out = c(out, sprintf("%s:{%s}", jstr(nm), paste(cols, collapse = ",")))
}

writeLines(paste0("{\n  ", paste(out, collapse = ",\n  "), "\n}"),
           file.path(outdir, "levels.json"))
cat("wrote", file.path(outdir, "levels.json"), "\n")

# report the data sets where R's order is not the alphabetical one pandas
# would infer.
for (nm in sets) {
  e = new.env(); data(list = nm, envir = e); d = get(nm, envir = e)
  for (col in names(d))
    if (is.factor(d[[col]]) && !identical(levels(d[[col]]), sort(levels(d[[col]]))))
      cat("  non-alphabetical:", nm, "$", col, ":",
          paste(levels(d[[col]]), collapse = ", "), "\n")
}
