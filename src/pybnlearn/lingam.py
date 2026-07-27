"""Direct LiNGAM: causal ordering from non-Gaussianity.

Every other structure learning algorithm here stops at the equivalence
class, because that is as far as conditional independence can see.  LiNGAM
gets past it by assuming something the others do not: that the noise is
*not* Gaussian.  Under that assumption the direction of an arc is
identifiable, because regressing the effect on the cause leaves a residual
independent of the cause, while regressing the cause on the effect does not.

The ordering below is that argument turned into an algorithm.  Take the
variable that is least dependent on the others once each is regressed out,
call it first, remove its influence from the rest, and repeat.  This mirrors
bnlearn's R/lingam.R.

The second half of R's direct.lingam() -- turning the ordering into arcs
with an adaptive lasso -- is *not* ported, because it is glmnet's
coordinate-descent solver and its lambda path, not bnlearn's.  What is here
instead is a score-based search restricted to the ordering, which is a
different estimator and says so: see `direct_lingam`.

bnlearn's other estimator, mi = "gkernel", is refused rather than ported.
Not because it is hard: because its answer is not reproducible.  See
`_gkernel_is_not_reproducible` for the measurements.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["direct_lingam", "lingam_ordering"]

_ESTIMATORS = ("pwling", "gkernel")


def _sd(x):
    """cgsd(): the sample standard deviation, with n - 1 in the denominator."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.sqrt(np.sum((x - x.mean()) ** 2) / (len(x) - 1)))


def _scale(x):
    """.scale(): centre and divide by the standard deviation, unless the
    variable is constant, in which case dividing would be by zero."""
    x = np.asarray(x, dtype=float)
    centred = x - np.nanmean(x)
    sd = _sd(x)
    return centred if sd == 0 else centred / sd


def _remove_effect(xi, xj):
    """The residuals of xi regressed on xj, intercept included.

    Removing one variable's influence from another is the step that makes
    the ordering greedy: once a variable is placed, what it explains is
    taken out of everything still to be placed.
    """
    xi = np.asarray(xi, dtype=float)
    xj = np.asarray(xj, dtype=float)

    design = np.column_stack([np.ones(len(xj)), xj])
    coefficients, *_ = np.linalg.lstsq(design, xi, rcond=None)
    return xi - design @ coefficients


def _entropy(u):
    """The maximum-entropy approximation of Hyvarinen's, with the constants
    from the paper.  Not the entropy of anything in closed form -- a
    correction to the Gaussian entropy that vanishes when u is Gaussian,
    which is exactly what LiNGAM needs to measure."""
    u = np.asarray(u, dtype=float)

    k1, k2, gamma = 79.047, 7.4129, 0.37457

    first = (1 + np.log(2 * np.pi)) / 2
    second = k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2
    third = k2 * (np.mean(u * np.exp(-(u ** 2) / 2))) ** 2

    return first - second - third


def _pairwise_mutual_information(xi, xj):
    """approx.mutual.information(): which way round the pair sits.

    The difference of two likelihood ratios -- xj causing xi against xi
    causing xj.  It is signed, and the sign is the answer; the caller keeps
    only the negative part, because a positive value says this variable is
    the effect rather than the cause.
    """
    residual_ij = _remove_effect(xi, xj)
    residual_ji = _remove_effect(xj, xi)

    sd_ij, sd_ji = _sd(residual_ij), _sd(residual_ji)
    if sd_ij == 0 or sd_ji == 0:
        return 0.0

    return ((_entropy(xj) + _entropy(residual_ij / sd_ij))
            - (_entropy(xi) + _entropy(residual_ji / sd_ji)))


def _gkernel_is_not_reproducible():
    """Why mi = "gkernel" raises instead of returning a number.

    bnlearn's Gaussian-kernel estimator builds its Gram matrix as

        exp(-(X^2 + t(X)^2 - 2 * tcrossprod(X)) / (2 * sigma^2))

    where every row of X is the data vector.  `tcrossprod` is a matrix
    product, so that third term is the constant sum(x^2) rather than the
    outer product -- the entry for a pair of observations does not depend on
    the distance between them, which is the one thing a kernel is for.  The
    line commented out immediately above it, in NumPy notation, uses the
    elementwise product and is correct; the one that runs does not.

    Reproducing a bug would be acceptable -- agreeing with R is the point of
    this package.  What is not acceptable is that the bug makes the answer
    unreproducible.  With the sign of that term flipped the exponent comes
    out positive, and on 80 rows of gaussian.test the Gram matrix runs from
    3.6e31 to 2.0e34 with a condition number around 1e20.  Double precision
    carries about 1e16.  The small singular values are therefore rounding
    noise, and `sum(log(sigma))` is dominated by them: R and this package,
    given identical inputs and identical formulas, get -52.2 and -35.7.  The
    difference is which order LAPACK happened to accumulate in.

    So there is no answer to agree with.  mi = "pwling" is the default, is
    correct, and is checked against R.
    """
    raise NotImplementedError(
        'mi="gkernel" is not available: bnlearn 5.2.1 computes its Gram '
        "matrix with a matrix product where it means an elementwise one, "
        "which leaves the matrix with a condition number around 1e20 -- so "
        "its result is floating-point noise and cannot be reproduced. Use "
        'mi="pwling", which is the default.')


def _precedence(nodes, whitelist, blacklist):
    """topological.precedence(): what the constraints already fix.

    A blacklist does not merely forbid an arc, it can force an ordering: if
    every route from A to B is blacklisted but the route back is not, then B
    cannot precede A.  Working that out before the search starts is what
    keeps the greedy step from choosing something the constraints forbid.
    """
    from .graph import path_exists

    forbidden = set(blacklist)
    viable = [(a, b) for a in nodes for b in nodes
              if a != b and (a, b) not in forbidden]

    reachable = {(a, b): path_exists(
        _bare_graph(nodes, viable), a, b, direct=True)
        for a in nodes for b in nodes if a != b}

    preceding = {node: [] for node in nodes}
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            if reachable[a, b] and not reachable[b, a]:
                preceding[b].append(a)

    roots = [n for n in nodes
             if not any(reachable[m, n] for m in nodes if m != n)]
    leaves = [n for n in nodes
              if not any(reachable[n, m] for m in nodes if m != n)]
    # an isolated node is both; R lists it only among the roots.
    leaves = [n for n in leaves if n not in roots]

    # a whitelisted arc is a parent-child relationship, and a parent always
    # precedes its child.
    present = set(whitelist)
    for a, b in whitelist:
        if (b, a) in present:
            continue  # undirected: it says nothing about the ordering
        if a not in preceding[b]:
            preceding[b].append(a)

    return preceding, roots, leaves


def _bare_graph(nodes, arcs):
    from .structure import BayesianNetwork

    return BayesianNetwork(nodes, arcs)


def lingam_ordering(data, mi="pwling", whitelist=None, blacklist=None):
    """The causal ordering LiNGAM infers from non-Gaussianity.

    Returns the variables in an order consistent with some DAG: every
    variable comes after everything that causes it.  This is more than an
    equivalence class gives you, and it is bought with the assumption that
    the noise is not Gaussian -- on genuinely Gaussian data the ordering is
    arbitrary and this will still return one.

    Parameters
    ----------
    data : pandas.DataFrame
        All columns continuous.
    mi : {"pwling"}
        How to measure dependence: Hyvarinen and Smith's pairwise likelihood
        ratio.  bnlearn's other option, "gkernel", raises -- its result in
        R is numerically meaningless, and `_gkernel_is_not_reproducible`
        gives the measurements.
    whitelist, blacklist : sequence of (from, to), optional
        Constraints, which can fix part of the ordering before the search
        starts.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if mi not in _ESTIMATORS:
        raise ValueError("mi must be 'pwling' or 'gkernel'")

    from .structure import _data_type, build_blacklist

    if _data_type(data) != "continuous":
        raise ValueError("direct LiNGAM needs continuous data")
    if data.isna().any().any():
        raise ValueError("direct LiNGAM does not accept missing values")

    nodes = [str(c) for c in data.columns]
    whitelist = [(str(a), str(b)) for a, b in (whitelist or ())]
    blacklist = build_blacklist(
        [(str(a), str(b)) for a, b in (blacklist or ())], whitelist)

    for a, b in whitelist + blacklist:
        if a not in nodes or b not in nodes:
            raise ValueError(f"unknown node in arc ({a}, {b})")

    preceding, roots, leaves = _precedence(nodes, whitelist, blacklist)

    working = {node: np.asarray(data[node], dtype=float) for node in nodes}

    ordering = list(roots)
    candidates = [n for n in nodes if n not in ordering and n not in leaves]

    if mi == "gkernel":
        _gkernel_is_not_reproducible()

    measure = _pairwise_mutual_information

    for _ in range(len(nodes)):
        if not candidates:
            break

        # a candidate can only go next if everything that must precede it is
        # already placed.
        viable = [n for n in candidates
                  if all(p in ordering for p in preceding[n])]

        found = _least_dependent(working, viable, candidates, measure)

        # take the chosen variable's influence out of the ones still to come.
        for node in candidates:
            if node != found:
                working[node] = _remove_effect(working[node], working[found])

        ordering.append(found)
        candidates = [n for n in candidates if n != found]

    return ordering + list(leaves)


def _least_dependent(working, candidates, to_test, measure):
    """search.exogenous.variable(): the variable least explained by the rest.

    For the pairwise measure only the negative part counts, squared: a
    positive value means this variable looks like the effect, and an effect
    is exactly what we are not looking for at this step.
    """
    totals = {}

    for candidate in candidates:
        xi = _scale(working[candidate])
        total = 0.0

        for other in to_test:
            if other == candidate:
                continue
            xj = _scale(working[other])

            # only the negative part counts, squared: a positive value says
            # this variable looks like the effect, and an effect is what we
            # are not looking for at this step.
            total += min(0.0, measure(xi, xj)) ** 2

        totals[candidate] = total

    return min(totals, key=totals.get)


def direct_lingam(data, mi="pwling", whitelist=None, blacklist=None,
                  maximize="hc", maximize_args=None):
    """A DAG from the LiNGAM causal ordering.

    The ordering is R's, exactly.  The arcs are **not**: bnlearn selects
    them with an adaptive lasso from glmnet, whose coordinate-descent solver
    and lambda path are not vendored here, so reproducing its parent sets is
    not possible.  What happens instead is a score-based search constrained
    to the ordering -- a different estimator, which will often but not
    always agree.

    Use `lingam_ordering` when the ordering is what you want; it is the part
    of LiNGAM that is checked against R.

    Parameters
    ----------
    data : pandas.DataFrame
    mi : {"pwling"}
    whitelist, blacklist : sequence of (from, to), optional
    maximize : {"hc", "tabu"}
        The search used to pick the arcs within the ordering.
    maximize_args : dict, optional
    """
    from . import structure
    from .graph import ordering2blacklist

    searches = {"hc": structure.hc, "tabu": structure.tabu}
    if maximize not in searches:
        raise ValueError("maximize must be 'hc' or 'tabu'")

    ordering = lingam_ordering(data, mi=mi, whitelist=whitelist,
                               blacklist=blacklist)

    # the ordering is the constraint: an arc may only run forwards along it.
    forbidden = ordering2blacklist(ordering)
    forbidden += [(str(a), str(b)) for a, b in (blacklist or ())
                  if (str(a), str(b)) not in forbidden]

    learned = searches[maximize](data, whitelist=whitelist,
                                 blacklist=forbidden,
                                 **(maximize_args or {}))

    learned.learning = dict(learned.learning)
    learned.learning.update(algo="direct.lingam", ordering=ordering,
                            maximize=maximize)
    return learned
