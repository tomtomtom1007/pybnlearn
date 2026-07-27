#!/usr/bin/env Rscript
# pybnlearn: reference results for the inputs R *refuses*.
#
# Every other generator here records what R computes from a valid input, so
# every other fixture file is a table of right answers.  That is the wrong
# shape for one whole class of bug.  If pybnlearn accepts something R rejects
# there is no right answer to compare against, and the parity suite --
# assembled entirely from inputs R was happy with -- cannot see it at all.
#
# The bugs that hide there are the expensive ones.  A malformed argument that
# crashes is self-announcing; a malformed argument that comes back as an
# empty network, or a network learned over half the variables, looks exactly
# like a finding.  Five had already been found by hand before this file
# existed: non-finite data (every Gaussian score goes NaN, no move is ever
# accepted, the empty network is returned), maxp <= 0 (same empty network),
# cyclic model strings, and a start= network over the wrong variables
# (silently learned over the subset).  All five returned a result.
#
# So this generator inverts the usual arrangement.  It walks a grid of
# deliberately malformed calls -- one per line, near enough -- and records
# only whether R stopped and what it said.  tests/parity/test_rejections.py
# replays the grid against pybnlearn and requires it to raise wherever R
# does.
#
# Two things are worth knowing about reading the output.
#
# `errors: false` rows are not padding.  R's boundaries are not always where
# you would guess -- is.probability() admits both 0 and 1, so alpha = 0 is
# legal; check.restart() short-circuits on zero, so restart = 0 is legal
# while perturb = 0 is not; a whitelisted self-loop is quietly dropped rather
# than refused.  Recording those keeps the Python side from over-rejecting,
# which would be its own parity bug and a much more annoying one to find.
#
# The messages are recorded but are not the contract.  They are here to say
# *which* check fired, so that a Python-side rejection for an unrelated
# reason is visible as such; the test matches on R's wording only loosely.
#
# Usage: Rscript tools/gen_r_rejection_fixtures.R [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

# R translates its own error messages, and the fixtures are read by a test
# that greps them; pin the language so the file does not change with the
# locale of whoever regenerates it.
Sys.setenv(LANGUAGE = "en")

suppressMessages(library(bnlearn))

args = commandArgs(trailingOnly = TRUE)
outdir = if (length(args) > 0) args[1] else "tests/parity/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

jstr = function(s) {
  s = paste(s, collapse = " ")
  s = gsub("\\", "\\\\", s, fixed = TRUE)
  s = gsub('"', '\\"', s, fixed = TRUE)
  s = gsub("\n", "\\n", s, fixed = TRUE)
  s = gsub("\t", "\\t", s, fixed = TRUE)
  paste0('"', s, '"')
}
jbool = function(b) if (isTRUE(b)) "true" else "false"
field = function(name, value) paste0(jstr(name), ":", value)

records = character(0)
group = ""

# Run one malformed call and record whether R stopped.
#
# `expr` arrives as an unevaluated promise and is forced inside the handler,
# so a call that errors is recorded rather than taking the generator down
# with it.  Warnings are suppressed rather than caught: R warns about plenty
# of things it goes on to accept (a maxp above the node count, an unused
# argument), and treating those as rejections would be wrong.
probe = function(id, expr) {
  outcome = tryCatch({
      suppressWarnings(force(expr))
      list(errors = FALSE, message = "")
    }, error = function(e) list(errors = TRUE, message = conditionMessage(e)))

  records <<- c(records, paste0("{", paste(c(
    field("id", jstr(id)),
    field("group", jstr(group)),
    field("errors", jbool(outcome$errors)),
    field("message", jstr(outcome$message))), collapse = ","), "}"))
}

e = new.env()
data(learning.test, envir = e)
data(gaussian.test, envir = e)

lt = get("learning.test", envir = e)          # discrete,   5000 x 6, A..F
gt = get("gaussian.test", envir = e)          # continuous, 5000 x 7, A..G

# a short discrete frame, so that the cross-validation cases can ask for
# more folds than there are rows without waiting for 5000 of them.
short = lt[1:20, ]

dag = model2network("[A][C][F][B|A][D|A:C][E|B:F]")
fitted = bn.fit(dag, lt)
gdag = model2network("[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")
gfitted = bn.fit(gdag, gt)

# a network over variables the data have never heard of.  This is the shape
# that used to be learned over the subset instead of refused.
foreign.dag = empty.graph(c("X", "Y", "Z"))
# and one over a strict subset of them, which is the subtler half of the
# same mistake.
subset.dag = empty.graph(c("A", "B", "C"))

# a triangle, for the arc operations that have to notice a cycle.  The six-
# node DAG above is too sparse to make one by touching a single arc, and
# set.arc() drops the opposing arc before it adds its own, so the obvious
# two-node "cycle" is not one.
tri = model2network("[A][B|A][C|A:B]")

wl = function(from, to) data.frame(from = from, to = to)

# ---------------------------------------------------------------------------
# alpha
#
# check.alpha() is is.probability(), which is a closed interval: 0 and 1 are
# both legal, and a test at alpha = 0 that accepts nothing is a defensible
# thing to ask for.  What it rejects is everything that is not a single
# finite number in that interval.  Swept exhaustively on gs and then once,
# with the same bad value, across every function that takes an alpha -- the
# question there is whether the check is wired up at all, not what it does.
# ---------------------------------------------------------------------------

group = "alpha"

probe("alpha/gs/zero", gs(lt, alpha = 0))
probe("alpha/gs/one", gs(lt, alpha = 1))
probe("alpha/gs/negative", gs(lt, alpha = -0.1))
probe("alpha/gs/above-one", gs(lt, alpha = 1.5))
probe("alpha/gs/nan", gs(lt, alpha = NaN))
probe("alpha/gs/infinite", gs(lt, alpha = Inf))
probe("alpha/gs/missing", gs(lt, alpha = NA))
probe("alpha/gs/string", gs(lt, alpha = "0.05"))
probe("alpha/gs/vector", gs(lt, alpha = c(0.01, 0.05)))

probe("alpha/iamb/negative", iamb(lt, alpha = -0.1))
probe("alpha/iamb.fdr/negative", iamb.fdr(lt, alpha = -0.1))
probe("alpha/inter.iamb/negative", inter.iamb(lt, alpha = -0.1))
probe("alpha/fast.iamb/negative", fast.iamb(lt, alpha = -0.1))
probe("alpha/mmpc/negative", mmpc(lt, alpha = -0.1))
probe("alpha/pc.stable/negative", pc.stable(lt, alpha = -0.1))
probe("alpha/si.hiton.pc/negative", si.hiton.pc(lt, alpha = -0.1))
probe("alpha/hpc/negative", hpc(lt, alpha = -0.1))
probe("alpha/learn.mb/negative",
      learn.mb(lt, node = "A", method = "gs", alpha = -0.1))
probe("alpha/learn.nbr/negative",
      learn.nbr(lt, node = "A", method = "mmpc", alpha = -0.1))
probe("alpha/arc.strength/negative",
      arc.strength(dag, lt, criterion = "mi", alpha = -0.1))
probe("alpha/arc.strength/above-one",
      arc.strength(dag, lt, criterion = "mi", alpha = 1.5))
probe("alpha/mmhc/negative",
      mmhc(lt, restrict.args = list(alpha = -0.1)))
probe("alpha/rsmax2/negative",
      rsmax2(lt, restrict.args = list(alpha = -0.1)))
probe("alpha/h2pc/negative",
      h2pc(lt, restrict.args = list(alpha = -0.1)))

# ---------------------------------------------------------------------------
# max.sx
#
# The bound on the size of the conditioning set.  Zero would mean marginal
# independence only, which sounds like a coherent request and is not one R
# will take; a limit above the number of nodes only warns.
# ---------------------------------------------------------------------------

group = "max.sx"

probe("max.sx/gs/zero", gs(lt, max.sx = 0))
probe("max.sx/gs/negative", gs(lt, max.sx = -1))
probe("max.sx/gs/fractional", gs(lt, max.sx = 1.5))
probe("max.sx/gs/infinite", gs(lt, max.sx = Inf))
probe("max.sx/gs/above-nnodes", gs(lt, max.sx = 100))
probe("max.sx/pc.stable/zero", pc.stable(lt, max.sx = 0))
probe("max.sx/si.hiton.pc/zero", si.hiton.pc(lt, max.sx = 0))
probe("max.sx/learn.nbr/zero",
      learn.nbr(lt, node = "A", method = "si.hiton.pc", max.sx = 0))

# ---------------------------------------------------------------------------
# the counters that steer hill climbing and tabu search
#
# Each of these is a count of something the search does, and each of them has
# a degenerate value that reads as "do none of it" -- zero iterations, zero
# parents, a tabu list with no slots.  R refuses all of them rather than
# returning the network it started from, which is the whole point: a search
# that was never allowed to move returns the empty network, and an empty
# network is a publishable-looking result.
#
# The exception is restart, which short-circuits on zero before the check
# runs -- no restarts is the default, not a degenerate request.  perturb = 0
# is *not* excused the same way, so the two differ by one line of R and by
# one row here.
# ---------------------------------------------------------------------------

group = "search-counters"

probe("maxp/hc/zero", hc(lt, maxp = 0))
probe("maxp/hc/negative", hc(lt, maxp = -1))
probe("maxp/hc/fractional", hc(lt, maxp = 1.5))
probe("maxp/hc/infinite", hc(lt, maxp = Inf))
probe("maxp/hc/above-nnodes", hc(lt, maxp = 100))
probe("maxp/tabu/zero", tabu(lt, maxp = 0))
probe("maxp/tabu/negative", tabu(lt, maxp = -1))
probe("maxp/tabu/fractional", tabu(lt, maxp = 1.5))

probe("max.iter/hc/zero", hc(lt, max.iter = 0))
probe("max.iter/hc/negative", hc(lt, max.iter = -1))
probe("max.iter/hc/fractional", hc(lt, max.iter = 1.5))
probe("max.iter/hc/infinite", hc(lt, max.iter = Inf))
probe("max.iter/tabu/zero", tabu(lt, max.iter = 0))
probe("max.iter/tabu/negative", tabu(lt, max.iter = -1))

probe("restart/hc/zero", hc(lt, restart = 0))
probe("restart/hc/negative", hc(lt, restart = -1))
probe("restart/hc/fractional", hc(lt, restart = 1.5))

probe("perturb/hc/zero", hc(lt, restart = 2, perturb = 0))
probe("perturb/hc/negative", hc(lt, restart = 2, perturb = -1))
probe("perturb/hc/fractional", hc(lt, restart = 2, perturb = 1.5))

probe("tabu/tabu/zero", tabu(lt, tabu = 0))
probe("tabu/tabu/negative", tabu(lt, tabu = -1))
probe("tabu/tabu/fractional", tabu(lt, tabu = 1.5))

# ---------------------------------------------------------------------------
# whitelists and blacklists
#
# The interesting rows here are the two that R accepts.  A self-loop is
# dropped rather than refused -- arcs.unique() removes it on the way in --
# so a Python side that raised would be wrong.  Everything else is refused,
# and refused *early*: a whitelist that cannot be part of any acyclic graph
# has no valid answer, so returning one would be a fabrication.
# ---------------------------------------------------------------------------

group = "arc-lists"

probe("whitelist/hc/self-loop", hc(lt, whitelist = wl("A", "A")))
probe("blacklist/hc/self-loop", hc(lt, blacklist = wl("A", "A")))
probe("whitelist/gs/self-loop", gs(lt, whitelist = wl("A", "A")))
probe("blacklist/gs/self-loop", gs(lt, blacklist = wl("A", "A")))

probe("whitelist/hc/unknown-node", hc(lt, whitelist = wl("Z", "A")))
probe("blacklist/hc/unknown-node", hc(lt, blacklist = wl("A", "Z")))
probe("whitelist/gs/unknown-node", gs(lt, whitelist = wl("Z", "A")))
probe("blacklist/gs/unknown-node", gs(lt, blacklist = wl("A", "Z")))
probe("whitelist/tabu/unknown-node", tabu(lt, whitelist = wl("Z", "A")))
probe("whitelist/pc.stable/unknown-node",
      pc.stable(lt, whitelist = wl("Z", "A")))
probe("whitelist/mmhc/unknown-node", mmhc(lt, whitelist = wl("Z", "A")))
probe("whitelist/chow.liu/unknown-node", chow.liu(lt, whitelist = wl("Z", "A")))
probe("whitelist/aracne/unknown-node", aracne(lt, whitelist = wl("Z", "A")))

probe("whitelist/hc/cycle",
      hc(lt, whitelist = wl(c("A", "B", "C"), c("B", "C", "A"))))
probe("whitelist/tabu/cycle",
      tabu(lt, whitelist = wl(c("A", "B", "C"), c("B", "C", "A"))))
probe("whitelist/gs/cycle",
      gs(lt, whitelist = wl(c("A", "B", "C"), c("B", "C", "A"))))
probe("whitelist/hc/both-directions",
      hc(lt, whitelist = wl(c("A", "B"), c("B", "A"))))
probe("whitelist/tabu/both-directions",
      tabu(lt, whitelist = wl(c("A", "B"), c("B", "A"))))

# ---------------------------------------------------------------------------
# start= networks that do not match the data
#
# One of the five found by hand.  A start network over other variables was
# silently learned over whichever ones happened to overlap, so asking hc()
# to start from the wrong network returned a network over the wrong node
# set -- with no indication that the answer was not about the data passed in.
# ---------------------------------------------------------------------------

group = "start"

probe("start/hc/foreign-nodes", hc(lt, start = foreign.dag))
probe("start/hc/subset-nodes", hc(lt, start = subset.dag))
probe("start/tabu/foreign-nodes", tabu(lt, start = foreign.dag))
probe("start/tabu/subset-nodes", tabu(lt, start = subset.dag))
probe("start/hc/matching-nodes", hc(lt, start = dag))

# ---------------------------------------------------------------------------
# the data themselves
#
# check.data() is the gate everything else stands behind, and the two rows
# that matter most are the empty frame -- which has no answer -- and the
# single-level factor, whose contingency tables have no degrees of freedom
# left.  Both used to be able to reach the C code.
# ---------------------------------------------------------------------------

group = "data"

probe("data/hc/no-rows", hc(lt[0, ]))
probe("data/gs/no-rows", gs(lt[0, ]))
probe("data/hc/single-column", hc(lt[, "A", drop = FALSE]))
probe("data/hc/single-level",
      hc(data.frame(A = factor(rep("a", 100)), B = lt$B[1:100])))
probe("data/hc/missing-values",
      hc(within(lt[1:100, ], A[1] <- NA)))
probe("data/hc/infinite",
      hc(within(gt[1:100, ], A[1] <- Inf)))
probe("data/hc/negative-infinite",
      hc(within(gt[1:100, ], A[1] <- -Inf)))
probe("data/hc/nan", hc(within(gt[1:100, ], A[1] <- NaN)))
probe("data/hc/large-but-finite",
      hc(within(gt[1:100, ], A[1] <- 1e300)))
probe("data/score/infinite",
      score(gdag, within(gt[1:100, ], A[1] <- Inf), type = "bic-g"))
probe("data/bn.fit/infinite",
      bn.fit(gdag, within(gt[1:100, ], A[1] <- Inf)))

# ---------------------------------------------------------------------------
# score and test labels, and the data types they apply to
#
# A score used on the wrong kind of data is the one mismatch that would
# otherwise produce a number rather than a complaint.
# ---------------------------------------------------------------------------

group = "labels"

probe("score/hc/unknown", hc(lt, score = "nosuch"))
probe("score/hc/discrete-on-continuous", hc(gt, score = "bde"))
probe("score/hc/continuous-on-discrete", hc(lt, score = "bge"))
probe("score/score/discrete-on-continuous", score(gdag, gt, type = "bde"))
probe("score/score/continuous-on-discrete", score(dag, lt, type = "bic-g"))
probe("test/gs/unknown", gs(lt, test = "nosuch"))
probe("test/gs/continuous-on-discrete", gs(lt, test = "cor"))
probe("test/gs/discrete-on-continuous", gs(gt, test = "mi"))
probe("criterion/arc.strength/unknown",
      arc.strength(dag, lt, criterion = "nosuch"))
probe("method/learn.mb/unknown", learn.mb(lt, node = "A", method = "nosuch"))
probe("node/learn.mb/unknown", learn.mb(lt, node = "Z", method = "gs"))
probe("method/bn.fit/unknown", bn.fit(dag, lt, method = "nosuch"))
probe("method/discretize/unknown", discretize(gt, method = "nosuch"))
probe("method/predict/unknown",
      predict(fitted, node = "A", data = lt, method = "nosuch"))

# ---------------------------------------------------------------------------
# score hyperparameters
# ---------------------------------------------------------------------------

group = "score-args"

probe("iss/score/zero", score(dag, lt, type = "bde", iss = 0))
probe("iss/score/negative", score(dag, lt, type = "bde", iss = -1))
probe("iss/score/fractional", score(dag, lt, type = "bde", iss = 0.5))
probe("iss/bn.fit/zero", bn.fit(dag, lt, method = "bayes", iss = 0))
probe("iss/bn.fit/negative", bn.fit(dag, lt, method = "bayes", iss = -1))
probe("iss/hc/negative", hc(lt, score = "bde", iss = -1))
probe("k/score/negative", score(dag, lt, type = "aic", k = -1))
# R does reject an unknown prior, but never gets as far as saying so:
# check.graph.prior() passes the *function* score to check.label()'s see=
# argument, and building the message dies on the closure.  The rejection is
# the contract; the wording here is upstream's accident.
probe("prior/score/unknown", score(dag, lt, type = "bde", prior = "nosuch"))
probe("beta/score/negative",
      score(dag, lt, type = "bde", prior = "vsp", beta = -1))
probe("beta/score/above-one",
      score(dag, lt, type = "bde", prior = "vsp", beta = 2))

# ---------------------------------------------------------------------------
# networks that do not match the data
#
# fit(), score() and predict() each take a network and a data set and have
# to agree that they are about the same variables.  When they do not, the
# honest answers are all errors; the dangerous answers are a fit over the
# overlap and a prediction from parents that were never observed.
# ---------------------------------------------------------------------------

group = "mismatch"

probe("mismatch/bn.fit/foreign-nodes", bn.fit(foreign.dag, lt))
probe("mismatch/bn.fit/subset-nodes", bn.fit(subset.dag, lt))
probe("mismatch/bn.fit/extra-node", bn.fit(add.node(dag, "Z"), lt))
probe("mismatch/score/foreign-nodes", score(foreign.dag, lt))
probe("mismatch/score/subset-nodes", score(subset.dag, lt))
probe("mismatch/score/extra-node", score(add.node(dag, "Z"), lt))
probe("mismatch/predict/unknown-node",
      predict(fitted, node = "Z", data = lt))
probe("mismatch/predict/missing-parents",
      predict(fitted, node = "E", data = lt[, c("A", "E"), drop = FALSE]))
probe("mismatch/arc.strength/foreign-nodes", arc.strength(foreign.dag, lt))
probe("mismatch/nparams/foreign-nodes", nparams(foreign.dag, lt))
probe("mismatch/bn.cv/foreign-nodes", bn.cv(short, foreign.dag, k = 2))
probe("mismatch/impute/foreign-data",
      impute(fitted, data.frame(X = factor(c("a", "b")))))

# ---------------------------------------------------------------------------
# cross-validation
#
# k has three separate boundaries: it must be a positive integer, it must be
# at least 2, and it must not exceed the number of rows.  The last one is the
# one a caller trips over accidentally -- ten-fold cross-validation on eight
# observations -- and folds that do not partition anything still produce
# numbers.
# ---------------------------------------------------------------------------

group = "cross-validation"

probe("k/bn.cv/zero", bn.cv(short, dag, k = 0))
probe("k/bn.cv/one", bn.cv(short, dag, k = 1))
probe("k/bn.cv/negative", bn.cv(short, dag, k = -1))
probe("k/bn.cv/fractional", bn.cv(short, dag, k = 2.5))
probe("k/bn.cv/above-nrow", bn.cv(short, dag, k = nrow(short) + 1))
probe("k/bn.cv/equals-nrow", bn.cv(short, dag, k = nrow(short)))
probe("runs/bn.cv/zero", bn.cv(short, dag, k = 2, runs = 0))
probe("runs/bn.cv/negative", bn.cv(short, dag, k = 2, runs = -1))
probe("runs/bn.cv/fractional", bn.cv(short, dag, k = 2, runs = 1.5))
probe("m/bn.cv/zero", bn.cv(short, dag, method = "hold-out", m = 0))
probe("m/bn.cv/negative", bn.cv(short, dag, method = "hold-out", m = -1))
probe("m/bn.cv/equals-nrow",
      bn.cv(short, dag, method = "hold-out", m = nrow(short)))
probe("m/bn.cv/above-nrow",
      bn.cv(short, dag, method = "hold-out", m = nrow(short) + 1))
probe("method/bn.cv/unknown", bn.cv(short, dag, method = "nosuch"))
probe("loss/bn.cv/unknown", bn.cv(short, dag, k = 2, loss = "nosuch"))
probe("loss/bn.cv/continuous-on-discrete", bn.cv(short, dag, k = 2,
        loss = "cor", loss.args = list(target = "A")))
probe("loss/bn.cv/no-target", bn.cv(short, dag, k = 2, loss = "pred"))
probe("loss/bn.cv/unknown-target", bn.cv(short, dag, k = 2, loss = "pred",
        loss.args = list(target = "Z")))
probe("folds/bn.cv/single-fold", bn.cv(short, dag, method = "custom-folds",
        folds = list(seq_len(nrow(short)))))
probe("folds/bn.cv/incomplete", bn.cv(short, dag, method = "custom-folds",
        folds = list(1:5, 6:10)))
probe("folds/bn.cv/overlapping", bn.cv(short, dag, method = "custom-folds",
        folds = list(1:10, 5:20)))
probe("folds/bn.cv/out-of-range", bn.cv(short, dag, method = "custom-folds",
        folds = list(1:10, 11:21)))

# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------

group = "resampling"

probe("R/boot.strength/zero", boot.strength(short, algorithm = "hc", R = 0))
probe("R/boot.strength/negative",
      boot.strength(short, algorithm = "hc", R = -5))
probe("R/boot.strength/fractional",
      boot.strength(short, algorithm = "hc", R = 2.5))
probe("m/boot.strength/zero",
      boot.strength(short, algorithm = "hc", R = 2, m = 0))
probe("m/boot.strength/negative",
      boot.strength(short, algorithm = "hc", R = 2, m = -1))
probe("algorithm/boot.strength/unknown",
      boot.strength(short, algorithm = "nosuch", R = 2))
probe("R/bn.boot/zero",
      bn.boot(short, statistic = narcs, algorithm = "hc", R = 0))
probe("R/bn.boot/negative",
      bn.boot(short, statistic = narcs, algorithm = "hc", R = -5))
probe("R/bn.boot/fractional",
      bn.boot(short, statistic = narcs, algorithm = "hc", R = 2.5))

# ---------------------------------------------------------------------------
# arc strength objects
# ---------------------------------------------------------------------------

group = "strength"

str.boot = boot.strength(short, algorithm = "hc", R = 5)

probe("threshold/averaged.network/negative",
      averaged.network(str.boot, threshold = -0.1))
probe("threshold/averaged.network/above-one",
      averaged.network(str.boot, threshold = 1.5))
probe("threshold/averaged.network/zero",
      averaged.network(str.boot, threshold = 0))
probe("threshold/averaged.network/one",
      averaged.network(str.boot, threshold = 1))
probe("custom.strength/mismatched-nodes",
      custom.strength(list(dag, foreign.dag), nodes = names(lt)))
probe("custom.strength/subset-nodes",
      custom.strength(list(dag, subset.dag), nodes = names(lt)))
probe("custom.strength/unknown-node-set",
      custom.strength(list(dag, dag), nodes = c(names(lt), "Z")))
probe("custom.strength/weights/negative",
      custom.strength(list(dag, dag), nodes = names(lt), weights = c(-1, 1)))
probe("custom.strength/weights/wrong-length",
      custom.strength(list(dag, dag), nodes = names(lt), weights = c(1, 1, 1)))

# ---------------------------------------------------------------------------
# graph constructors and accessors
#
# The empty and single-node cases are here because they are the ones a
# caller reaches by accident, filtering a data frame down to nothing.  R
# will build a one-node graph quite happily; it will not build a zero-node
# one, and it will not accept a node list with a repeat in it.
# ---------------------------------------------------------------------------

group = "graphs"

probe("empty.graph/no-nodes", empty.graph(character(0)))
probe("empty.graph/single-node", empty.graph("A"))
probe("empty.graph/duplicate-nodes", empty.graph(c("A", "A")))
probe("empty.graph/empty-label", empty.graph(c("A", "")))
probe("complete.graph/no-nodes", complete.graph(character(0)))
probe("complete.graph/single-node", complete.graph("A"))
probe("complete.graph/duplicate-nodes", complete.graph(c("A", "A")))

probe("model2network/reciprocal", model2network("[A|B][B|A]"))
probe("model2network/cycle", model2network("[A|C][B|A][C|B]"))
probe("model2network/self-loop", model2network("[A|A]"))
probe("model2network/duplicate-node", model2network("[A][A]"))
probe("model2network/undeclared-parent", model2network("[A|Z]"))
probe("model2network/single-node", model2network("[A]"))
probe("model2network/empty", model2network(""))

probe("random.graph/no-nodes", random.graph(character(0)))
probe("random.graph/single-node", random.graph("A"))
probe("random.graph/duplicate-nodes", random.graph(c("A", "A")))
probe("random.graph/num-zero", random.graph(c("A", "B", "C"), num = 0))
probe("random.graph/num-negative", random.graph(c("A", "B", "C"), num = -1))
probe("random.graph/method-unknown",
      random.graph(c("A", "B", "C"), method = "nosuch"))
probe("random.graph/prob-negative",
      random.graph(c("A", "B", "C"), method = "ordered", prob = -0.1))
probe("random.graph/prob-above-one",
      random.graph(c("A", "B", "C"), method = "ordered", prob = 1.5))
probe("random.graph/every-zero",
      random.graph(c("A", "B", "C"), method = "ic-dag", every = 0))
probe("random.graph/burn.in-negative",
      random.graph(c("A", "B", "C"), method = "ic-dag", burn.in = -1))
probe("random.graph/max.degree-zero",
      random.graph(c("A", "B", "C"), method = "ic-dag", max.degree = 0))

probe("subgraph/unknown-node", subgraph(dag, c("A", "Z")))
probe("subgraph/no-nodes", subgraph(dag, character(0)))
probe("subgraph/single-node", subgraph(dag, "A"))
probe("subgraph/duplicate-nodes", subgraph(dag, c("A", "A")))

probe("compare/mismatched-nodes", compare(dag, foreign.dag))
probe("shd/mismatched-nodes", shd(dag, foreign.dag))
probe("hamming/mismatched-nodes", hamming(dag, foreign.dag))
probe("compare/subset-nodes", compare(dag, subset.dag))

probe("count.graphs/nodes-negative", count.graphs("all-dags", nodes = -1))
probe("count.graphs/nodes-zero", count.graphs("all-dags", nodes = 0))
probe("count.graphs/nodes-fractional", count.graphs("all-dags", nodes = 2.5))
probe("count.graphs/type-unknown", count.graphs("nosuch", nodes = 3))
probe("count.graphs/k-above-nodes",
      count.graphs("dags-with-k-roots", nodes = 3, k = 5))
probe("count.graphs/k-zero",
      count.graphs("dags-with-k-roots", nodes = 3, k = 0))
probe("count.graphs/r-negative",
      count.graphs("dags-with-r-arcs", nodes = 3, r = -1))

# ---------------------------------------------------------------------------
# node and arc accessors
#
# These are the smallest functions in the package and the ones most likely
# to be called in a loop with a computed node name.  An unknown label has to
# be an error and not an empty answer, because an empty answer is what a
# genuinely childless node returns.
# ---------------------------------------------------------------------------

group = "accessors"

probe("parents/unknown-node", parents(dag, "Z"))
probe("children/unknown-node", children(dag, "Z"))
probe("mb/unknown-node", mb(dag, "Z"))
probe("nbr/unknown-node", nbr(dag, "Z"))
probe("ancestors/unknown-node", bnlearn::ancestors(dag, "Z"))
probe("descendants/unknown-node", bnlearn::descendants(dag, "Z"))
probe("in.degree/unknown-node", in.degree(dag, "Z"))
probe("incident.arcs/unknown-node", incident.arcs(dag, "Z"))
probe("parents/several-nodes", parents(dag, c("A", "B")))

probe("set.arc/unknown-node", set.arc(dag, "Z", "A"))
probe("set.arc/self-loop", set.arc(dag, "A", "A"))
probe("set.arc/reversal", set.arc(dag, "B", "A"))
probe("set.arc/cycle", set.arc(tri, "C", "A"))
probe("drop.arc/unknown-node", drop.arc(dag, "Z", "A"))
probe("drop.arc/absent-arc", drop.arc(dag, "A", "F"))
probe("reverse.arc/absent-arc", reverse.arc(dag, "A", "F"))
probe("reverse.arc/cycle", reverse.arc(tri, "A", "C"))
probe("set.edge/self-loop", set.edge(dag, "A", "A"))

probe("add.node/existing-node", add.node(dag, "A"))
probe("remove.node/unknown-node", remove.node(dag, "Z"))
probe("rename.nodes/wrong-length", rename.nodes(dag, c("a", "b")))
probe("rename.nodes/duplicates",
      rename.nodes(dag, c("a", "a", "c", "d", "e", "f")))

probe("amat/set/wrong-size",
      { g = dag; m = amat(g)[1:3, 1:3]; amat(g) = m; g })
probe("amat/set/cyclic",
      { g = tri; m = amat(g); m["A", "C"] = 0L; m["C", "A"] = 1L
        amat(g) = m; g })
probe("amat/set/undirected",
      { g = tri; m = amat(g); m["C", "A"] = 1L; amat(g) = m; g })

# ---------------------------------------------------------------------------
# simulation and inference
# ---------------------------------------------------------------------------

group = "inference"

probe("n/rbn/zero", rbn(fitted, n = 0))
probe("n/rbn/negative", rbn(fitted, n = -1))
probe("n/rbn/fractional", rbn(fitted, n = 1.5))
probe("n/cpdist/zero", cpdist(fitted, nodes = "A", evidence = TRUE, n = 0))
probe("n/cpdist/negative", cpdist(fitted, nodes = "A", evidence = TRUE, n = -1))
probe("nodes/cpdist/unknown", cpdist(fitted, nodes = "Z", evidence = TRUE))
probe("n/predict/zero",
      predict(fitted, node = "A", data = lt, method = "bayes-lw", n = 0))
probe("n/predict/negative",
      predict(fitted, node = "A", data = lt, method = "bayes-lw", n = -1))
probe("n/impute/zero", impute(fitted, lt, method = "bayes-lw", n = 0))
probe("n/impute/negative", impute(fitted, lt, method = "bayes-lw", n = -1))

# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

group = "preprocessing"

probe("breaks/discretize/zero", discretize(gt, method = "quantile", breaks = 0))
probe("breaks/discretize/one", discretize(gt, method = "quantile", breaks = 1))
probe("breaks/discretize/negative",
      discretize(gt, method = "quantile", breaks = -1))
probe("breaks/discretize/fractional",
      discretize(gt, method = "quantile", breaks = 2.5))
probe("ibreaks/discretize/below-breaks",
      discretize(gt, method = "hartemink", breaks = 3, ibreaks = 2))
probe("discretize/discrete-data", discretize(lt, method = "quantile"))
probe("threshold/dedup/negative", dedup(gt, threshold = -0.1))
probe("threshold/dedup/above-one", dedup(gt, threshold = 1.5))
# dedup(lt) with no method= is deliberately absent: check.deduplication.method()
# only rejects "cor" on discrete data when the label was passed explicitly, so
# the default path walks straight into dedup.backend() and segfaults R.  That
# is an upstream bug rather than a boundary to reproduce, and running it here
# would take the generator down with it.
probe("dedup/discrete-data", dedup(lt, method = "cor"))

# ---------------------------------------------------------------------------
# classifiers
# ---------------------------------------------------------------------------

group = "classifiers"

probe("training/naive.bayes/unknown", naive.bayes(lt, training = "Z"))
probe("training/naive.bayes/continuous", naive.bayes(gt, training = "A"))
probe("explanatory/naive.bayes/unknown",
      naive.bayes(lt, training = "A", explanatory = "Z"))
probe("explanatory/naive.bayes/includes-training",
      naive.bayes(lt, training = "A", explanatory = c("A", "B")))
probe("training/tree.bayes/unknown", tree.bayes(lt, training = "Z"))
probe("root/tree.bayes/unknown", tree.bayes(lt, training = "A", root = "Z"))
probe("root/tree.bayes/is-training",
      tree.bayes(lt, training = "A", root = "A"))

writeLines(paste0("[\n", paste(records, collapse = ",\n"), "\n]"),
           file.path(outdir, "rejections.json"))
cat("wrote", length(records), "records to",
    file.path(outdir, "rejections.json"), "\n")
cat("  R rejects",
    sum(grepl('"errors":true', records, fixed = TRUE)), "of them\n")
