#!/usr/bin/env Rscript
# pybnlearn: reference files and parameters for BIF, DSC and NET.
#
# This writes the files with R and records what R's own bn.fit object holds,
# so that reading them back in Python is compared against the network they
# were written from rather than against another parser's opinion.
#
# The formats differ in how much they say about where each conditional
# distribution belongs.  BIF names the parent configuration; DSC gives
# numeric coordinates; NET gives nothing and relies on the order.  A reader
# that gets the order wrong still produces a valid-looking network, so the
# recorded tables below are compared cell by cell rather than checked for
# shape.
#
# Usage: Rscript tools/gen_r_foreign_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
files = file.path(outdir, "foreign")
dir.create(files, showWarnings = FALSE, recursive = TRUE)

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
for (nm in c("learning.test", "asia", "lizards", "insurance"))
  data(list = nm, envir = e)

structures = list(
  # a root, a node with one parent, and a node with two.
  "learning.test" = "[A][C][F][B|A][D|A:C][E|B:F]",
  # eight nodes, all binary, the standard toy network.
  "asia" = "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
  # levels R orders Sagrei, Distichus -- not alphabetically, which is what
  # makes this one worth having.
  "lizards" = "[Species][Diameter|Species][Height|Species]",
  # 27 nodes, up to four parents, and levels containing spaces and
  # punctuation that the writers have to sanitise.
  "insurance" = paste0("[Age][Mileage][SocioEcon|Age][GoodStudent|Age:SocioEcon]",
                       "[RiskAversion|Age:SocioEcon][VehicleYear|SocioEcon:RiskAversion]",
                       "[MakeModel|SocioEcon:RiskAversion][DrivingSkill|Age:SocioEcon]",
                       "[SeniorTrain|Age:RiskAversion][DrivQuality|RiskAversion:DrivingSkill]",
                       "[CarValue|MakeModel:VehicleYear][AntiTheft|SocioEcon:RiskAversion]",
                       "[HomeBase|SocioEcon:RiskAversion][Antilock|MakeModel:VehicleYear]",
                       "[Airbag|MakeModel:VehicleYear][Cushioning|MakeModel:VehicleYear]",
                       "[RuggedAuto|MakeModel:VehicleYear][DrivHist|RiskAversion:DrivingSkill]",
                       "[Theft|CarValue:AntiTheft:HomeBase][OtherCar|SocioEcon]",
                       "[Accident|DrivQuality:Mileage:Antilock]",
                       "[ThisCarDam|RuggedAuto:Accident][ILiCost|Accident]",
                       "[MedCost|Age:Accident:Cushioning][OtherCarCost|RuggedAuto:Accident]",
                       "[ThisCarCost|ThisCarDam:CarValue:Theft]",
                       "[PropCost|ThisCarCost:OtherCarCost]"))

for (name in names(structures)) {
  d = get(name, envir = e)
  fitted = bn.fit(model2network(structures[[name]]), d)

  for (fmt in c("bif", "dsc", "net")) {
    path = file.path(files, paste0(name, ".", fmt))
    do.call(paste0("write.", fmt), list(path, fitted))

    # and read it back with R, which is the reference the Python reader is
    # compared against.
    back = do.call(paste0("read.", fmt), list(path))

    for (node in nodes(back)) {
      entry = back[[node]]
      add(field("kind", jstr("read")),
          field("dataset", jstr(name)),
          field("format", jstr(fmt)),
          field("file", jstr(paste0(name, ".", fmt))),
          field("node", jstr(node)),
          field("parents", jarr(entry$parents)),
          field("dim", paste0("[", paste(dim(entry$prob), collapse = ","), "]")),
          field("dimnames", paste0("[", paste(sapply(dimnames(entry$prob),
            jarr), collapse = ","), "]")),
          # as.vector() unrolls column-major, which is numpy's order="F".
          field("prob", num(as.vector(entry$prob))))
    }

    add(field("kind", jstr("network")),
        field("dataset", jstr(name)),
        field("format", jstr(fmt)),
        field("file", jstr(paste0(name, ".", fmt))),
        field("nodes", jarr(nodes(back))),
        field("modelstring", jstr(modelstring(bn.net(back)))))
  }
}

# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------

graphs = list(
  "dag" = model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
  "cpdag" = cpdag(model2network("[A][C][F][B|A][D|A:C][E|B:F]")),
  "skeleton" = skeleton(model2network("[A][C][F][B|A][D|A:C][E|B:F]")),
  "empty" = empty.graph(LETTERS[1:4]))

for (name in names(graphs)) {
  path = file.path(files, paste0(name, ".dot"))
  write.dot(path, graphs[[name]])

  add(field("kind", jstr("dot")),
      field("graph", jstr(name)),
      field("file", jstr(paste0(name, ".dot"))),
      field("nodes", jarr(bnlearn::nodes(graphs[[name]]))),
      field("arcs", if (narcs(graphs[[name]]) == 0) "[]" else
        paste0("[", paste(apply(bnlearn::arcs(graphs[[name]]), 1, function(r)
          paste0("[", jstr(r[1]), ",", jstr(r[2]), "]")), collapse = ","), "]")))
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "foreign.json"))
cat("wrote", length(records), "records to",
    file.path(outdir, "foreign.json"), "\n")
