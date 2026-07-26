#!/usr/bin/env Rscript
# pybnlearn: reference results for hpc() and h2pc().
#
# HPC is three algorithms stacked on each other, and each stage can go wrong
# in a way the next one papers over: the cheap parents-and-children superset,
# the spouse superset it derives from the separating sets, and the expensive
# filtering inside both.  A mistake in the first stage usually shows up as a
# missing arc rather than an error, so the arc sets are compared including
# direction, over data sets with different shapes: a sparse chain, a
# collider-heavy network, one with a large in-degree, and continuous data.
#
# The last section is the reason hpc is worth having at all: h2pc, which
# feeds it into a hill-climbing phase.
#
# Usage: Rscript tools/gen_r_hpc_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
jbool = function(b) if (isTRUE(b)) "true" else "false"
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
for (nm in c("learning.test", "asia", "coronary", "lizards", "insurance",
             "alarm", "gaussian.test", "marks"))
  data(list = nm, envir = e)

discrete = c("learning.test", "asia", "coronary", "lizards", "alarm")
continuous = c("gaussian.test", "marks")

# ---------------------------------------------------------------------------
# hpc
# ---------------------------------------------------------------------------

for (dname in c(discrete, continuous)) {
  d = get(dname, envir = e)
  tests = if (dname %in% discrete) c("mi", "x2", "mi-sh") else
            c("cor", "zf", "mi-g")

  for (tst in tests)
    for (a in c(0.01, 0.05)) {
      learned = hpc(d, test = tst, alpha = a)
      add(field("kind", jstr("hpc")),
          field("dataset", jstr(dname)),
          field("test", jstr(tst)),
          field("alpha", sprintf("%.17g", a)),
          field("undirected", "true"),
          field("arcs", jarcs(arcs(learned))))
    }

  # directed, which runs the orientation phase on hpc's skeleton.
  learned = hpc(d, undirected = FALSE)
  add(field("kind", jstr("hpc")),
      field("dataset", jstr(dname)),
      field("test", "null"), field("alpha", "0.05"),
      field("undirected", "false"),
      field("arcs", jarcs(arcs(learned))))

  # with a limit on the conditioning set, which changes both supersets.
  for (msx in c(1, 2)) {
    learned = hpc(d, max.sx = msx)
    add(field("kind", jstr("hpc")),
        field("dataset", jstr(dname)),
        field("test", "null"), field("alpha", "0.05"),
        field("undirected", "true"),
        field("max.sx", sprintf("%d", msx)),
        field("arcs", jarcs(arcs(learned))))
  }

  # and with constraints.
  ns = names(d)
  wl = matrix(ns[1:2], ncol = 2)
  learned = hpc(d, whitelist = wl)
  add(field("kind", jstr("hpc")),
      field("dataset", jstr(dname)),
      field("test", "null"), field("alpha", "0.05"),
      field("undirected", "true"),
      field("whitelist", jarcs(wl)),
      field("arcs", jarcs(arcs(learned))))

  bl = matrix(ns[c(1, 3)], ncol = 2)
  learned = hpc(d, blacklist = bl)
  add(field("kind", jstr("hpc")),
      field("dataset", jstr(dname)),
      field("test", "null"), field("alpha", "0.05"),
      field("undirected", "true"),
      field("blacklist", jarcs(bl)),
      field("arcs", jarcs(arcs(learned))))
}

# ---------------------------------------------------------------------------
# learn.nbr with hpc
# ---------------------------------------------------------------------------

for (dname in c("learning.test", "asia", "gaussian.test")) {
  d = get(dname, envir = e)
  for (node in names(d))
    add(field("kind", jstr("nbr")),
        field("dataset", jstr(dname)),
        field("node", jstr(node)),
        field("result", jarr(learn.nbr(d, node, method = "hpc"))))
}

# ---------------------------------------------------------------------------
# h2pc
# ---------------------------------------------------------------------------

for (dname in c("learning.test", "asia", "coronary", "lizards",
                "gaussian.test", "marks")) {
  d = get(dname, envir = e)

  learned = h2pc(d)
  add(field("kind", jstr("h2pc")),
      field("dataset", jstr(dname)),
      field("args", "{}"),
      field("modelstring", jstr(modelstring(learned))),
      field("arcs", jarcs(arcs(learned))))

  # arguments passed through to each phase separately.
  learned = h2pc(d, restrict.args = list(alpha = 0.01),
             maximize.args = list(maxp = 2))
  add(field("kind", jstr("h2pc")),
      field("dataset", jstr(dname)),
      field("args", paste0("{", jstr("alpha"), ":0.01,", jstr("maxp"), ":2}")),
      field("modelstring", jstr(modelstring(learned))),
      field("arcs", jarcs(arcs(learned))))
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "hpc.json"))
cat("wrote", length(records), "records to", file.path(outdir, "hpc.json"),
    "\n")
