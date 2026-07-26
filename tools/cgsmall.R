# pybnlearn: a small mixed data set, defined once and sourced where needed.
#
# bnlearn ships exactly one data set that mixes factors with numeric columns,
# and it has a shape the conditional Gaussian code suits rather well: every
# continuous node has at least one discrete parent.  This one is deliberately
# different -- R has no discrete parent at all, so the fitting backend has to
# choose between three estimators rather than reach for the same one twice --
# and its first factor's levels are not in alphabetical order, which is what
# the recorded level orders exist to preserve.
#
# Defining it here rather than in either generator keeps the data and its
# recorded level orders from drifting apart.
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

cgsmall = local({
  set.seed(42)
  n = 800
  d = data.frame(
    P = factor(sample(c("low", "high"), n, TRUE), levels = c("low", "high")),
    Q = factor(sample(c("x", "y", "z"), n, TRUE)),
    R = rnorm(n),
    S = rnorm(n))
  d$S = d$S + 3 * as.numeric(d$P) + 0.5 * d$R
  d
})
