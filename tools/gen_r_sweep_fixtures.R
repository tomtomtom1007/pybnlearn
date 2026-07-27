#!/usr/bin/env Rscript
# pybnlearn: an exhaustive sweep of the learning surface against R.
#
# The other generators are collections of cases chosen by hand, and hand-
# chosen cases share the blind spots of the hand that chose them.  Three in
# particular:
#
#   * arguments are varied one at a time, so a pair that interacts -- a
#     whitelist together with a parent limit, say, where the whitelist wants
#     an arc the limit has no room for -- is never reached;
#   * the data are always one of bnlearn's own sets, which are well behaved:
#     plenty of rows, moderate numbers of levels, no near-determinism;
#   * a few exported functions are never called at all.
#
# So this sweeps instead of sampling.  Every score against every data set,
# every algorithm against every independence test, and the argument
# combinations crossed rather than listed.  The data sets below are chosen to
# be awkward on purpose: two rows per cell, a variable with eleven levels, a
# column that is nearly constant, two columns that are nearly collinear, more
# variables than the sample can support.
#
# Everything is seeded and the data are written out alongside the results, so
# the comparison is against the same numbers rather than the same recipe.
# Continuous data go out at %.17g -- write.csv() rounds to 15 significant
# digits, which is invisible in a coefficient and moves a residual in the
# fourteenth place.
#
# Usage: Rscript tools/gen_r_sweep_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

num = function(x) if (length(x) == 0) "[]" else
       paste0("[", paste(sapply(x, function(v)
         if (is.na(v)) "null" else if (is.infinite(v))
           (if (v > 0) "\"inf\"" else "\"-inf\"") else
           sprintf("%.17g", v)), collapse = ","), "]")
one = function(v) if (is.na(v)) "null" else if (is.infinite(v))
        (if (v > 0) "\"inf\"" else "\"-inf\"") else sprintf("%.17g", v)
jstr = function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jarr = function(v) if (length(v) == 0) "[]" else
                     paste0("[", paste(sapply(v, jstr), collapse = ","), "]")
jbool = function(b) if (isTRUE(b)) "true" else "false"
field = function(name, value) paste0(jstr(name), ":", value)
jarcs = function(a) {
  if (is.null(a) || length(a) == 0) return("[]")
  if (is.null(dim(a))) a = matrix(a, ncol = 2)
  if (nrow(a) == 0) return("[]")
  paste0("[", paste(apply(a, 1, function(r)
    paste0("[", jstr(r[1]), ",", jstr(r[2]), "]")), collapse = ","), "]")
}

records = character(0)
add = function(...) records <<- c(records,
        paste0("{", paste(c(...), collapse = ","), "}"))

# a call that may legitimately fail -- an argument combination with no
# solution, a score that cannot be computed.  The failure is part of the
# answer and is recorded rather than skipped.
attempt = function(expr)
  tryCatch(list(ok = TRUE, value = expr),
    error = function(e) list(ok = FALSE, message = conditionMessage(e)),
    warning = function(w) tryCatch(
      list(ok = TRUE, value = suppressWarnings(expr)),
      error = function(e) list(ok = FALSE, message = conditionMessage(e))))

write_data = function(d, name) {
  out = d
  for (col in names(out))
    if (is.numeric(out[[col]]))
      out[[col]] = sprintf("%.17g", out[[col]])
  write.csv(out, file.path(outdir, paste0("sweep.", name, ".csv")),
            row.names = FALSE, quote = FALSE)
}

# ---------------------------------------------------------------------------
# the data
#
# Awkward on purpose.  bnlearn's own data sets are all comfortable: several
# thousand rows, two to five levels, nothing degenerate.  A port can agree on
# those and still disagree everywhere the arithmetic is delicate.
# ---------------------------------------------------------------------------

set.seed(20260727)

sets = list()

# a contingency table with roughly two observations per cell: the Bayesian
# scores' imaginary sample size stops being negligible here.
sets[["sparse-cells"]] = as.data.frame(lapply(
  setNames(1:5, LETTERS[1:5]),
  function(i) factor(sample(c("a", "b", "c"), 80, replace = TRUE))))

# eleven levels on one variable, two on another: the degrees of freedom of a
# test between them are lopsided.
sets[["many-levels"]] = local({
  A = factor(sample(letters[1:11], 400, replace = TRUE))
  B = factor(ifelse(runif(400) < 0.75, as.integer(A) %% 2, sample(0:1, 400, TRUE)))
  C = factor(sample(c("x", "y"), 400, replace = TRUE))
  D = factor(sample(letters[1:7], 400, replace = TRUE))
  data.frame(A = A, B = B, C = C, D = D)
})

# one level takes 98% of the mass.  Cells that are empty in the sample but
# not in the model are where the discrete scores differ from each other.
sets[["unbalanced"]] = local({
  A = factor(sample(c("a", "b"), 500, replace = TRUE, prob = c(0.98, 0.02)))
  B = factor(ifelse(runif(500) < 0.9, as.character(A), "b"))
  C = factor(sample(c("p", "q", "r"), 500, replace = TRUE,
                    prob = c(0.9, 0.07, 0.03)))
  D = factor(sample(c("u", "v"), 500, replace = TRUE))
  data.frame(A = A, B = B, C = C, D = D)
})

# nearly deterministic: B is A with 1% noise.  The mutual information is
# close to its maximum and the log-likelihood close to zero.
sets[["near-deterministic"]] = local({
  A = factor(sample(c("a", "b", "c"), 300, replace = TRUE))
  B = factor(ifelse(runif(300) < 0.99, as.character(A),
                    sample(c("a", "b", "c"), 300, TRUE)))
  C = factor(sample(c("x", "y"), 300, replace = TRUE))
  data.frame(A = A, B = B, C = C)
})

# more variables than the sample comfortably supports: 12 columns, 40 rows.
sets[["wide"]] = as.data.frame(lapply(
  setNames(1:12, paste0("V", 1:12)),
  function(i) factor(sample(c("0", "1"), 40, replace = TRUE))))

# nearly collinear continuous columns: the Gaussian scores go through a QR
# decomposition, and this is where a rank decision has to agree.
sets[["collinear"]] = local({
  A = rnorm(200)
  B = A + rnorm(200, sd = 1e-6)
  C = 2 * A - 3 * B + rnorm(200, sd = 0.1)
  D = rnorm(200)
  data.frame(A = A, B = B, C = C, D = D)
})

# wildly different scales, and one variable that is nearly constant.
sets[["scaled"]] = local({
  A = rnorm(150, mean = 0, sd = 1e-4)
  B = rnorm(150, mean = 1e6, sd = 1e5)
  C = A * 1e4 + B * 1e-6 + rnorm(150)
  D = rep(1, 150) + rnorm(150, sd = 1e-9)
  data.frame(A = A, B = B, C = C, D = D)
})

# heavy tails: nothing here is Gaussian, which the Gaussian scores do not
# know and must still compute consistently.
sets[["heavy-tailed"]] = local({
  A = rt(180, df = 2)
  B = A * 0.8 + rt(180, df = 2)
  C = rcauchy(180, scale = 0.5)
  D = rt(180, df = 3)
  data.frame(A = A, B = B, C = C, D = D)
})

# a mixture, with the discrete parent of a continuous child having a level
# that only a handful of rows reach.
sets[["mixed-rare"]] = local({
  A = factor(sample(c("a", "b", "c"), 250, replace = TRUE,
                    prob = c(0.6, 0.36, 0.04)))
  B = factor(sample(c("x", "y"), 250, replace = TRUE))
  C = rnorm(250) + as.integer(A)
  D = rnorm(250) + as.integer(B) * 2
  data.frame(A = A, B = B, C = C, D = D)
})

# small: 30 rows, which is where degrees of freedom start to bite.
sets[["small-n"]] = local({
  A = factor(sample(c("a", "b"), 30, replace = TRUE))
  B = factor(sample(c("a", "b"), 30, replace = TRUE))
  C = factor(sample(c("p", "q", "r"), 30, replace = TRUE))
  data.frame(A = A, B = B, C = C)
})

for (name in names(sets))
  write_data(sets[[name]], name)

# Which columns are factors has to be recorded rather than inferred: the
# "wide" set's levels are "0" and "1", which read back as numbers and would
# send a discrete data set to a continuous score.
for (name in names(sets)) {
  d = sets[[name]]
  add(field("kind", jstr("dataset")),
      field("name", jstr(name)),
      field("nodes", jarr(names(d))),
      field("rows", sprintf("%d", nrow(d))),
      field("discrete", jarr(names(d)[sapply(d, is.factor)])),
      field("levels", paste0("{", paste(sapply(
        names(d)[sapply(d, is.factor)],
        function(n) paste0(jstr(n), ":", jarr(levels(d[[n]])))),
        collapse = ","), "}")))
}

kind = function(d) {
  f = sapply(d, is.factor)
  if (all(f)) "discrete" else if (!any(f)) "continuous" else "mixed"
}

scores_for = list(
  discrete = c("loglik", "aic", "bic", "ebic", "bde", "bds", "bdj", "k2",
               "fnml", "qnml"),
  continuous = c("loglik-g", "aic-g", "bic-g", "ebic-g", "bge"),
  mixed = c("loglik-cg", "aic-cg", "bic-cg", "ebic-cg"))

tests_for = list(
  discrete = c("mi", "mi-adf", "x2", "x2-adf", "mi-sh"),
  continuous = c("cor", "zf", "mi-g", "mi-g-sh"),
  mixed = character(0))

# ---------------------------------------------------------------------------
# every score against every data set, on three structures each
# ---------------------------------------------------------------------------

structures = function(d) {
  ns = names(d)
  out = list(empty = empty.graph(ns))
  # a chain through the variables in order
  chain = empty.graph(ns)
  if (length(ns) > 1)
    arcs(chain) = matrix(c(ns[-length(ns)], ns[-1]), ncol = 2,
                    dimnames = list(NULL, c("from", "to")))
  out$chain = chain
  # everything pointing at the last variable
  star = empty.graph(ns)
  if (length(ns) > 1)
    arcs(star) = matrix(c(ns[-length(ns)], rep(ns[length(ns)],
      length(ns) - 1)), ncol = 2, dimnames = list(NULL, c("from", "to")))
  out$star = star
  out
}

for (dname in names(sets)) {
  d = sets[[dname]]
  k = kind(d)

  for (sname in names(structures(d))) {
    net = structures(d)[[sname]]

    for (sc in scores_for[[k]]) {
      r = attempt(score(net, d, type = sc))
      # the per-node breakdown as well as the total: a total can be right
      # while the parts that make it up are wrong in ways that cancel.
      parts = attempt(score(net, d, type = sc, by.node = TRUE))

      add(field("kind", jstr("score")),
          field("dataset", jstr(dname)),
          field("structure", jstr(sname)),
          field("score", jstr(sc)),
          field("ok", jbool(r$ok)),
          field("value", if (r$ok) one(r$value) else "null"),
          field("nodes", jarr(names(d))),
          field("by.node", if (parts$ok) num(parts$value[names(d)]) else "null"),
          field("error", if (r$ok) "null" else jstr(r$message)))
    }
  }
}

# ---------------------------------------------------------------------------
# every conditional independence test, over every ordered pair and a
# conditioning set of each size the data allow
# ---------------------------------------------------------------------------

for (dname in names(sets)) {
  d = sets[[dname]]
  k = kind(d)
  if (length(tests_for[[k]]) == 0) next
  ns = names(d)
  if (length(ns) > 6) ns = ns[1:6]     # keep the sweep finite on the wide set

  for (test in tests_for[[k]])
    for (x in ns) for (y in ns) {
      if (x >= y) next
      rest = setdiff(ns, c(x, y))
      # unconditional, then one variable, then two
      for (sx in list(character(0), rest[1],
                      if (length(rest) >= 2) rest[1:2] else NULL)) {
        if (is.null(sx)) next
        # z has to be *absent* for an unconditional test, not NULL: passing
        # NULL positionally is an argument of the wrong type, not a missing
        # one, and ci.test() rejects it.
        r = attempt(if (length(sx))
                      ci.test(x, y, sx, data = d, test = test)
                    else
                      ci.test(x, y, data = d, test = test))

        add(field("kind", jstr("citest")),
            field("dataset", jstr(dname)),
            field("test", jstr(test)),
            field("x", jstr(x)), field("y", jstr(y)),
            field("sx", jarr(sx)),
            field("ok", jbool(r$ok)),
            field("statistic", if (r$ok) one(unname(r$value$statistic)) else "null"),
            field("df", if (r$ok && !is.null(r$value$parameter))
                          one(unname(r$value$parameter)) else "null"),
            field("p.value", if (r$ok) one(r$value$p.value) else "null"),
            field("error", if (r$ok) "null" else jstr(r$message)))
      }
    }
}

# ---------------------------------------------------------------------------
# score-based learning: arguments crossed rather than listed
# ---------------------------------------------------------------------------

for (dname in names(sets)) {
  d = sets[[dname]]
  k = kind(d)
  ns = names(d)

  # a whitelist and a blacklist that overlap the same pair of variables, and
  # a parent limit tight enough to fight the whitelist.
  wl = matrix(c(ns[1], ns[2]), ncol = 2,
              dimnames = list(NULL, c("from", "to")))
  bl = matrix(c(ns[2], ns[3], ns[3], ns[1]), ncol = 2, byrow = TRUE,
              dimnames = list(NULL, c("from", "to")))

  for (algo in c("hc", "tabu"))
    for (sc in scores_for[[k]])
      for (constraint in list(
             list(name = "none"),
             list(name = "whitelist", whitelist = wl),
             list(name = "blacklist", blacklist = bl),
             list(name = "both", whitelist = wl, blacklist = bl)))
        for (limit in list(NULL, 1, 2)) {

          call = list(x = d, score = sc)
          if (!is.null(constraint$whitelist)) call$whitelist = constraint$whitelist
          if (!is.null(constraint$blacklist)) call$blacklist = constraint$blacklist
          if (!is.null(limit)) call$maxp = limit

          r = attempt(do.call(algo, call))

          add(field("kind", jstr("learn")),
              field("dataset", jstr(dname)),
              field("algorithm", jstr(algo)),
              field("score", jstr(sc)),
              field("constraint", jstr(constraint$name)),
              field("maxp", if (is.null(limit)) "null" else sprintf("%d", limit)),
              field("ok", jbool(r$ok)),
              field("arcs", if (r$ok) jarcs(arcs(r$value)) else "[]"),
              field("error", if (r$ok) "null" else jstr(r$message)))
        }
}

# ---------------------------------------------------------------------------
# combinations with no solution
#
# Without these the failure paths on the Python side are dead code: the sweep
# above is all satisfiable, so nothing would ever check that this refuses
# what R refuses.  Each of these is impossible for a different reason.
# ---------------------------------------------------------------------------

for (dname in names(sets)) {
  d = sets[[dname]]
  k = kind(d)
  ns = names(d)

  impossible = list(
    # a whitelist that is itself a cycle
    list(name = "cyclic-whitelist", algorithm = "hc",
         args = list(whitelist = matrix(c(ns[1], ns[2], ns[2], ns[1]),
                       ncol = 2, byrow = TRUE,
                       dimnames = list(NULL, c("from", "to"))))),
    # the same arc whitelisted and blacklisted
    list(name = "contradiction", algorithm = "hc",
         args = list(
           whitelist = matrix(c(ns[1], ns[2]), ncol = 2,
                        dimnames = list(NULL, c("from", "to"))),
           blacklist = matrix(c(ns[1], ns[2]), ncol = 2,
                        dimnames = list(NULL, c("from", "to"))))),
    # a score for the wrong kind of data
    list(name = "wrong-score", algorithm = "hc",
         args = list(score = if (k == "discrete") "bge" else "bde")),
    # a test for the wrong kind of data
    list(name = "wrong-test", algorithm = "gs",
         args = list(test = if (k == "discrete") "cor" else "mi")),
    # a significance level that is not a probability
    list(name = "alpha-too-big", algorithm = "gs",
         args = list(alpha = 1.5)),
    # a node that is not in the data
    list(name = "unknown-node", algorithm = "hc",
         args = list(whitelist = matrix(c("nonesuch", ns[1]), ncol = 2,
                       dimnames = list(NULL, c("from", "to"))))))

  for (spec in impossible) {
    r = attempt(do.call(spec$algorithm, c(list(x = d), spec$args)))

    add(field("kind", jstr("impossible")),
        field("dataset", jstr(dname)),
        field("name", jstr(spec$name)),
        field("algorithm", jstr(spec$algorithm)),
        field("ok", jbool(r$ok)),
        field("arcs", if (r$ok) jarcs(arcs(r$value)) else "[]"),
        field("error", if (r$ok) "null" else jstr(r$message)))
  }
}

# ---------------------------------------------------------------------------
# constraint-based learning: every algorithm against every test, at two
# significance levels, directed and undirected
# ---------------------------------------------------------------------------

for (dname in names(sets)) {
  d = sets[[dname]]
  k = kind(d)
  if (length(tests_for[[k]]) == 0) next

  for (algo in c("pc.stable", "gs", "iamb", "inter.iamb", "iamb.fdr",
                 "fast.iamb", "mmpc", "si.hiton.pc", "hpc"))
    for (test in tests_for[[k]])
      for (alpha in c(0.01, 0.1))
        for (und in c(FALSE, TRUE)) {

          r = attempt(do.call(algo, list(x = d, test = test, alpha = alpha,
                                         undirected = und)))

          add(field("kind", jstr("constraint")),
              field("dataset", jstr(dname)),
              field("algorithm", jstr(algo)),
              field("test", jstr(test)),
              field("alpha", one(alpha)),
              field("undirected", jbool(und)),
              field("ok", jbool(r$ok)),
              field("arcs", if (r$ok) jarcs(arcs(r$value)) else "[]"),
              field("error", if (r$ok) "null" else jstr(r$message)))
        }
}

# ---------------------------------------------------------------------------
# parameter learning and the quantities derived from it
# ---------------------------------------------------------------------------

for (dname in names(sets)) {
  d = sets[[dname]]
  k = kind(d)
  net = structures(d)$star

  methods = if (k == "discrete") c("mle", "bayes") else
            if (k == "continuous") "mle-g" else "mle-cg"

  for (method in methods)
    for (iss in if (method == "bayes") c(1, 10) else NA) {

      call = list(x = net, data = d, method = method)
      if (!is.na(iss)) call$iss = iss
      r = attempt(do.call(bn.fit, call))
      if (!r$ok) {
        add(field("kind", jstr("fit")), field("dataset", jstr(dname)),
            field("method", jstr(method)),
            field("iss", if (is.na(iss)) "null" else one(iss)),
            field("ok", "false"), field("error", jstr(r$message)),
            field("nodes", jarr(names(d))), field("params", "{}"))
        next
      }
      fitted = r$value

      # every number in every node, flattened in R's own storage order.
      params = paste0("{", paste(sapply(names(d), function(n) {
        node = fitted[[n]]
        v = if (!is.null(node$prob)) as.numeric(node$prob)
            else c(as.numeric(node$coefficients), as.numeric(node$sd))
        paste0(jstr(n), ":", num(v))
      }), collapse = ","), "}")

      add(field("kind", jstr("fit")),
          field("dataset", jstr(dname)),
          field("method", jstr(method)),
          field("iss", if (is.na(iss)) "null" else one(iss)),
          field("ok", "true"),
          field("nodes", jarr(names(d))),
          field("nparams", sprintf("%d", nparams(fitted, d))),
          field("logLik", one(logLik(fitted, d))),
          field("params", params),
          field("error", "null"))
    }
}

# ---------------------------------------------------------------------------
# what a learned network remembers about how it was learned
#
# whitelist(), blacklist() and ntests() read the learning metadata back.  The
# blacklist is the interesting one: it is not what was passed in, because
# whitelisting an arc implicitly blacklists its reverse, and that implied
# entry is what pins the direction down during orientation.  ntests() is the
# work counter, which only agrees if the search took the same path.
# ---------------------------------------------------------------------------

for (dname in names(sets)) {
  d = sets[[dname]]
  k = kind(d)
  ns = names(d)

  wl = matrix(c(ns[1], ns[2]), ncol = 2,
              dimnames = list(NULL, c("from", "to")))
  bl = matrix(c(ns[3], ns[1]), ncol = 2,
              dimnames = list(NULL, c("from", "to")))

  # every algorithm, not just two: ntests() is the sharpest check there is on
  # whether a search took the same path, and it caught mmpc doing more work
  # than R for the same answer.
  specs = list(
    list(name = "hc-constrained", algorithm = "hc",
         args = list(whitelist = wl, blacklist = bl)),
    list(name = "gs-constrained", algorithm = "gs",
         args = list(whitelist = wl, blacklist = bl)))
  for (a in c("hc", "tabu", "gs", "pc.stable", "iamb", "inter.iamb",
              "iamb.fdr", "fast.iamb", "mmpc", "si.hiton.pc", "hpc"))
    specs[[length(specs) + 1]] = list(name = paste0(a, "-plain"),
                                      algorithm = a, args = list())

  for (spec in specs) {

    if (!(spec$algorithm %in% c("hc", "tabu"))
        && length(tests_for[[k]]) == 0) next
    r = attempt(do.call(spec$algorithm, c(list(x = d), spec$args)))
    if (!r$ok) next
    g = r$value

    add(field("kind", jstr("metadata")),
        field("dataset", jstr(dname)),
        field("name", jstr(spec$name)),
        field("algorithm", jstr(spec$algorithm)),
        field("constrained", jbool(length(spec$args) > 0)),
        field("whitelist", jarcs(whitelist(g))),
        field("blacklist", jarcs(blacklist(g))),
        field("ntests", sprintf("%d", ntests(g))))
  }
}

# ---------------------------------------------------------------------------
# the graph utilities, over the structures every data set produces
#
# These are the functions the runtime coverage count found unreached:
# subgraph(), connected.components(), reversible.arcs(), and the accessors.
# ---------------------------------------------------------------------------

graphs = list(
  "chain" = model2network("[A][B|A][C|B][D|C][E|D]"),
  "collider" = model2network("[A][B][C|A:B]"),
  "diamond" = model2network("[A][B|A][C|A][D|B:C]"),
  "disconnected" = model2network("[A][B|A][C][D|C][E]"),
  "star-in" = model2network("[A][B][C][D|A:B:C]"),
  "star-out" = model2network("[A][B|A][C|A][D|A]"),
  "single" = empty.graph("A"),
  "empty5" = empty.graph(LETTERS[1:5]))

for (gname in names(graphs)) {
  g = graphs[[gname]]
  ns = nodes(g)
  eq = cpdag(g)

  add(field("kind", jstr("graph")),
      field("name", jstr(gname)),
      field("nodes", jarr(ns)),
      field("arcs", jarcs(arcs(g))),
      field("narcs", sprintf("%d", narcs(g))),
      field("nnodes", sprintf("%d", nnodes(g))),
      field("directed", jbool(directed(g))),
      field("acyclic", jbool(acyclic(g))),
      field("cpdag", jarcs(arcs(eq))),
      field("moral", jarcs(arcs(moral(g)))),
      field("skeleton", jarcs(arcs(skeleton(g)))),
      field("compelled", jarcs(compelled.arcs(eq))),
      field("reversible", jarcs(reversible.arcs(eq))),
      # connected.components() returns list(components =, chordal =), so it
      # is the inner list that has to be counted -- length() of the whole
      # thing is 2 for every graph, which is a fixture that agrees with
      # nothing.
      field("components", sprintf("%d", length(
        bnlearn:::connected.components(skeleton(g))$components))),
      field("component.nodes", paste0("[", paste(sapply(
        bnlearn:::connected.components(skeleton(g))$components, jarr),
        collapse = ","), "]")),
      field("ordering", jarr(node.ordering(g))))

  # every proper subgraph on a prefix of the nodes
  for (i in seq_along(ns)) {
    keep = ns[1:i]
    sub = attempt(subgraph(g, keep))
    add(field("kind", jstr("subgraph")),
        field("name", jstr(gname)),
        field("keep", jarr(keep)),
        field("ok", jbool(sub$ok)),
        field("arcs", if (sub$ok) jarcs(arcs(sub$value)) else "[]"),
        field("error", if (sub$ok) "null" else jstr(sub$message)))
  }
}

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "sweep.json"))
cat("wrote", length(records), "records to", file.path(outdir, "sweep.json"),
    "\n")
