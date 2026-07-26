#!/usr/bin/env Rscript
# pybnlearn: generate the reference values that the parity suite checks against.
#
# The point of the port is that results match R, not that they look plausible,
# so the test suite compares against numbers produced by R itself rather than
# against values hard-coded by hand.  Run this whenever the set of cases grows,
# and commit the JSON so that CI does not need R installed.
#
# Usage: Rscript tools/gen_r_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# A tiny JSON writer, so that the script needs nothing beyond bnlearn itself.
esc = function(s) gsub('"', '\\\\"', s)
num = function(x) {
  if (length(x) == 0 || is.null(x) || is.na(x)) "null"
  else sprintf("%.17g", as.numeric(x))
}

data(learning.test)
data(gaussian.test)

write.csv(learning.test, file.path(outdir, "learning.test.csv"),
          row.names = FALSE, quote = FALSE)
write.csv(gaussian.test, file.path(outdir, "gaussian.test.csv"),
          row.names = FALSE, quote = FALSE)

records = list()
emit = function(dataset, x, y, sx, test, r) {
  sprintf(paste0('{"dataset":"%s","x":"%s","y":"%s","sx":[%s],"test":"%s",',
                 '"statistic":%s,"parameter":%s,"p.value":%s}'),
    dataset, x, y,
    if (length(sx)) paste0('"', esc(sx), '"', collapse = ",") else "",
    test, num(r$statistic), num(r$parameter), num(r$p.value))
}

discrete.tests = c("mi", "mi-adf", "x2", "x2-adf", "mi-sh")
gaussian.tests = c("cor", "zf", "mi-g", "mi-g-sh")

nodes = names(learning.test)
for (test in discrete.tests) {
  for (i in seq_along(nodes)) {
    for (j in seq_along(nodes)) {
      if (i >= j) next
      r = ci.test(nodes[i], nodes[j], data = learning.test, test = test)
      records[[length(records) + 1]] =
        emit("learning.test", nodes[i], nodes[j], character(0), test, r)
      # one conditioning variable, chosen to be neither of the two tested
      k = setdiff(nodes, c(nodes[i], nodes[j]))[1]
      r = ci.test(nodes[i], nodes[j], k, data = learning.test, test = test)
      records[[length(records) + 1]] =
        emit("learning.test", nodes[i], nodes[j], k, test, r)
    }
  }
}

gnodes = names(gaussian.test)
for (test in gaussian.tests) {
  for (i in seq_along(gnodes)) {
    for (j in seq_along(gnodes)) {
      if (i >= j) next
      r = ci.test(gnodes[i], gnodes[j], data = gaussian.test, test = test)
      records[[length(records) + 1]] =
        emit("gaussian.test", gnodes[i], gnodes[j], character(0), test, r)
      k = setdiff(gnodes, c(gnodes[i], gnodes[j]))[1]
      r = ci.test(gnodes[i], gnodes[j], k, data = gaussian.test, test = test)
      records[[length(records) + 1]] =
        emit("gaussian.test", gnodes[i], gnodes[j], k, test, r)
    }
  }
}

out = file.path(outdir, "ci_test.json")
writeLines(paste0("[\n  ", paste(unlist(records), collapse = ",\n  "), "\n]"),
           out)

cat(sprintf("wrote %s: %d reference values\n", out, length(records)))
cat(sprintf("R %s, bnlearn %s\n", getRversion(),
            as.character(packageVersion("bnlearn"))))
