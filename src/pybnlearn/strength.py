"""How strongly the data support each arc, and the network that follows.

Three ways of measuring the same thing, mirroring bnlearn's
R/arc.strength.R and R/averaged.network.R:

* as a p-value -- how hard it is to reject the hypothesis that the arc is
  unnecessary, so *small* means strong;
* as a score difference -- how much worse the network gets without the arc,
  so *negative* means strong;
* as a bootstrap frequency -- how often the arc appears when the data are
  resampled, so *large* means strong.

`averaged_network` turns the third kind back into a network.  The threshold
it uses by default is not a round number chosen for convenience: it is the
value that best separates the arcs that keep reappearing from those that do
not, in the sense of Scutari and Nagarajan (2013), and computing it needs
R's own one-dimensional optimiser -- reproduced in `_brent_fmin` -- because
a different optimiser would pick a slightly different threshold and so a
different network.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._core import Search, Tester, averaged_arcs, which_arcs_undirected
from .structure import (BayesianNetwork, _check_complete, _check_score,
                        _check_score_args, _data_type)

__all__ = ["arc_strength", "averaged_network", "bf_strength",
           "custom_strength", "inclusion_threshold"]


def arc_strength(network, data, criterion=None, alpha=0.05, **extra_args):
    """How much each arc matters, one arc at a time.

    Every arc is removed in turn and the damage measured, either by testing
    the two nodes for independence given the rest of the target's parents or
    by rescoring the network without it.  Which of the two happens depends on
    whether `criterion` names a conditional independence test or a score.

    Parameters
    ----------
    network : BayesianNetwork
        Completely directed; strength is undefined for an undirected arc.
    data : pandas.DataFrame
    criterion : str, optional
        A test or a score.  Defaults to whatever the network was learned
        with, and to the default test for the data if it was not learned.
    alpha : float
        The significance threshold, recorded for `averaged_network`; it does
        not change the p-values.
    **extra_args
        Passed to the test or the score.

    Returns
    -------
    pandas.DataFrame with columns from, to, strength.  `.attrs` carries the
    method ("test" or "score"), the threshold and the node labels.
    """
    if not isinstance(network, BayesianNetwork):
        raise TypeError("arc_strength() needs a BayesianNetwork")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    _check_complete(data)

    nodes = [str(c) for c in data.columns]
    if set(nodes) != set(network.nodes):
        raise ValueError("the network and the data have different variables")

    undirected = _undirected(network)
    if undirected:
        raise ValueError(
            "the graph is only partially directed; "
            + ", ".join(f"{a}-{b}" for a, b in undirected[:3])
            + " has no direction, and the strength of an arc is the strength "
              "of removing it, which needs one")

    # R reuses whatever the network was learned with, which may be a test or
    # a score depending on the algorithm, and falls back to the default test.
    if criterion is None:
        criterion = network.learning.get("test")
    if criterion is None:
        from .constraint import _check_test

        criterion = _check_test(None, data)

    if criterion in _TESTS:
        from .constraint import _check_test

        criterion = _check_test(criterion, data)
        strength = _strength_by_test(network, data, criterion, extra_args)
        method, threshold = "test", float(alpha)
    else:
        criterion = _check_score(criterion, data)
        strength = _strength_by_score(network, data, criterion, extra_args)
        method, threshold = "score", 0.0

    result = pd.DataFrame(
        {"from": [a for a, _ in network.arcs],
         "to": [b for _, b in network.arcs],
         "strength": strength})
    result.attrs.update(nodes=nodes, method=method, threshold=threshold,
                        criterion=criterion)
    return result


def _undirected(network):
    """The arcs that appear in both directions."""
    arcs = set(network.arcs)
    return sorted({(a, b) for a, b in arcs if (b, a) in arcs})


def _strength_by_test(network, data, test, extra_args):
    """arc.strength.test(): the p-value of dropping the arc.

    Conditioning on the target's *other* parents is what makes this a test of
    the arc rather than of the pair: it asks whether the parent still says
    anything once its co-parents have spoken.
    """
    if extra_args:
        raise ValueError(
            f"the test {test!r} takes no extra arguments; "
            + ", ".join(sorted(extra_args)) + " given")

    values = []
    with Tester(data, test) as tester:
        for parent, child in network.arcs:
            others = [p for p in network.parents(child) if p != parent]
            values.append(tester.pvalue(parent, child, others))

    return values


def _strength_by_score(network, data, score, extra_args):
    """arc.strength.score(): the change in score from dropping the arc.

    Every score wired up here is decomposable -- only the Castelo & Siebes
    priors are not, and those are not supported -- so removing an arc can
    only change the score of the node it points at, and the difference is
    that one node's contribution before and after.
    """
    extra = _check_score_args(score, data, extra_args)
    nodes = [str(c) for c in data.columns]

    zeros = np.zeros((len(nodes), len(nodes)), dtype=np.int32)
    values = []
    with Search(data, nodes, score, extra, zeros, zeros) as search:
        reference = dict(zip(nodes, search.node_scores(network.arcs, nodes)))

        for parent, child in network.arcs:
            without = [arc for arc in network.arcs if arc != (parent, child)]
            updated = search.node_scores(without, [child])
            values.append(float(updated[0]) - float(reference[child]))

    return values


def custom_strength(networks, nodes=None, weights=None, cpdag=True):
    """How often an arc appears across a collection of networks.

    The same counting `boot_strength` does, over networks you supply rather
    than ones learned from resampled data -- from several algorithms, say, or
    from several people's prior beliefs.

    Parameters
    ----------
    networks : sequence of BayesianNetwork, FittedNetwork, or arc lists
    nodes : sequence of str, optional
        The node labels; taken from the first network if not given.
    weights : sequence of float, optional
        How much each network counts; equal by default.
    cpdag : bool
        Reduce each network to its CPDAG first, so that an arc whose
        direction the data cannot identify is not counted as if it could.

    Returns
    -------
    pandas.DataFrame with columns from, to, strength, direction.
    """
    from .bootstrap import _count_networks

    networks = list(networks)
    if not networks:
        raise ValueError("at least one network is needed")

    if nodes is None:
        nodes = _nodes_of(networks[0])
    nodes = [str(v) for v in nodes]

    if weights is None:
        weights = np.ones(len(networks))
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(networks):
            raise ValueError(
                f"{len(weights)} weights for {len(networks)} networks")
        if (weights < 0).any() or not weights.sum() > 0:
            raise ValueError("the weights must be non-negative and not all zero")

    # a graph keeps whatever it was learned under, which cpdag() needs; a
    # fitted network or a bare arc list has nothing to keep.
    prepared = [n if isinstance(n, BayesianNetwork)
                else BayesianNetwork(nodes, _arcs_of(n, nodes))
                for n in networks]

    result = _count_networks(prepared, nodes, weights=weights, cpdag=cpdag)
    result.attrs.update(nodes=nodes, method="bootstrap",
                        threshold=inclusion_threshold(result))
    return result


def _nodes_of(network):
    if hasattr(network, "nodes"):
        return list(network.nodes)
    raise TypeError(
        "the node labels cannot be taken from a bare arc list; pass nodes=")


def _arcs_of(network, nodes):
    """The arcs of a network given as a graph, a fitted network or a list."""
    if isinstance(network, BayesianNetwork):
        return list(network.arcs)

    if hasattr(network, "nodes") and hasattr(network, "__getitem__"):
        # a FittedNetwork: read the structure back off the parameters.
        try:
            return [(parent, node) for node in network.nodes
                    for parent in network[node].parents]
        except (AttributeError, KeyError, TypeError):
            pass

    arcs = [(str(a), str(b)) for a, b in network]
    unknown = {n for arc in arcs for n in arc} - set(nodes)
    if unknown:
        raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))
    return arcs


# ---------------------------------------------------------------------------
# from strengths back to a network
# ---------------------------------------------------------------------------

def averaged_network(strength, threshold=None):
    """The consensus network of a set of arc strengths.

    Arcs are taken strongest first and an arc is kept unless it would close a
    cycle with the arcs already taken, so the result is a DAG even though
    nothing about the strengths guarantees one.

    Parameters
    ----------
    strength : pandas.DataFrame
        From `boot_strength` or `custom_strength`.
    threshold : float, optional
        Keep arcs stronger than this; defaults to `inclusion_threshold`,
        which is what the strengths were annotated with.

    Returns
    -------
    BayesianNetwork
    """
    _check_strength(strength)

    if threshold is None:
        threshold = strength.attrs.get("threshold")
        if threshold is None:
            threshold = inclusion_threshold(strength)
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("the threshold must be between 0 and 1")

    nodes = list(strength.attrs.get("nodes")
                 or sorted(set(strength["from"]) | set(strength["to"])))

    values = strength["strength"].to_numpy(dtype=float)
    # an arc that appeared in every single network is kept whatever the
    # threshold says, which matters when the threshold itself lands at 1.
    significant = (values > threshold) | (values == 1)
    if "direction" in strength.columns:
        significant &= strength["direction"].to_numpy(dtype=float) >= 0.5

    if not significant.any():
        return BayesianNetwork(nodes, [], {"algo": "averaged",
                                           "args": {"threshold": threshold}})

    candidates = [(str(a), str(b)) for a, b in
                  zip(strength.loc[significant, "from"],
                      strength.loc[significant, "to"])]

    if which_arcs_undirected(candidates, nodes).all():
        arcs = candidates
    else:
        arcs = averaged_arcs(candidates, nodes, values[significant])

    return BayesianNetwork(nodes, arcs,
                           {"algo": "averaged",
                            "args": {"threshold": threshold}})


def inclusion_threshold(strength):
    """The threshold that best separates the arcs that belong from the rest.

    Scutari and Nagarajan's estimator: the strengths are treated as a mixture
    of arcs that are really there and arcs that are not, and the threshold is
    the point at which the observed distribution of strengths is closest, in
    L1, to the ideal one that would arise if the split were clean.
    """
    _check_strength(strength)

    values = np.sort(np.asarray(strength["strength"], dtype=float))
    if len(values) == 0:
        return 0.0

    knots = _unique(values)

    def ecdf(x):
        return np.searchsorted(values, x, side="right") / len(values)

    # R builds these two vectors once and reuses them for every candidate.
    widths = np.diff(_unique(np.concatenate([[0.0], knots, [1.0]])))
    observed = ecdf(_unique(np.concatenate([[0.0], knots[knots < 1]])))

    def norm(p):
        return float(np.sum(widths * np.abs(observed - p)))

    best = _brent_fmin(0.0, 1.0, norm, np.finfo(float).eps ** 0.25)

    # optimize() never evaluates the endpoints, and they are legal answers.
    if norm(1.0) < norm(best):
        best = 1.0
    if norm(0.0) < norm(best):
        best = 0.0

    return _quantile_type1(values, best)


def _check_strength(strength):
    if not isinstance(strength, pd.DataFrame):
        raise TypeError("strength must be a DataFrame from boot_strength() "
                        "or custom_strength()")
    missing = {"from", "to", "strength"} - set(strength.columns)
    if missing:
        raise ValueError(
            "the strengths are missing the column(s) "
            + ", ".join(sorted(missing)))
    if strength.attrs.get("method") not in (None, "bootstrap"):
        raise ValueError(
            f"these strengths are {strength.attrs['method']} statistics, not "
            "bootstrap frequencies; averaging needs frequencies, which come "
            "from boot_strength() or custom_strength()")


def _unique(values):
    """R's unique(): first appearance wins, and the order is not sorted."""
    values = np.asarray(values, dtype=float)
    _, index = np.unique(values, return_index=True)
    return values[np.sort(index)]


def _quantile_type1(sorted_values, p):
    """quantile(type = 1): the inverse of the empirical distribution.

    R pads the sorted sample at both ends rather than clamping the index, and
    the padding is what makes p = 0 and p = 1 come out right.
    """
    n = len(sorted_values)
    padded = np.concatenate([sorted_values[:1], sorted_values[:1],
                             sorted_values,
                             sorted_values[-1:], sorted_values[-1:]])

    nppm = n * p
    j = math.floor(nppm)
    high = nppm > j

    return float(padded[j + 2] if high else padded[j + 1])


def _brent_fmin(ax, bx, f, tol):
    """R's optimize(), which is Brent's fmin.

    Ported rather than replaced with scipy or a coarse search because the
    answer is fed straight into a threshold: two optimisers that agree to
    six digits can still put an arc on opposite sides of it and so return
    different networks.
    """
    c = (3.0 - math.sqrt(5.0)) * 0.5

    eps = np.finfo(float).eps
    tol1 = eps + 1.0
    eps = math.sqrt(eps)

    a, b = ax, bx
    v = a + c * (b - a)
    w = x = v

    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0

    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0

        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break

        p = q = r = 0.0
        if abs(e) > tol1:  # fit a parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d

        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            # a golden-section step
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            # a parabolic-interpolation step
            d = p / q
            u = x + d
            # f must not be evaluated too close to the ends of the interval
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d

        # nor too close to the best point so far
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1

        fu = f(u)

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

    return x


# the criteria arc_strength() treats as tests rather than scores.
_TESTS = frozenset({
    "mi", "mi-adf", "mi-sh", "x2", "x2-adf", "mc-mi", "smc-mi",
    "cor", "zf", "mi-g", "mi-g-sh",
})


def _bayes_factors(network, data, nodes, type, extra):
    """One Bayes factor per pair of nodes, in the installed decimal context.

    For each pair: the marginal likelihood with the arc one way, the other
    way, and not at all.  Normalising the three gives a probability that the
    arc is there and, given that, which way it points.
    """
    import decimal
    import itertools

    from .graph import path_exists

    one = decimal.Decimal(1)
    zeros = np.zeros((len(nodes), len(nodes)), dtype=np.int32)
    strength, direction = {}, {}

    with Search(data, nodes, type, extra, zeros, zeros) as search:
        for first, second in itertools.combinations(network.nodes, 2):
            without = [arc for arc in network.arcs
                       if arc not in ((first, second), (second, first))]
            base = dict(zip([first, second],
                            search.node_scores(without, [first, second])))

            def weight(frm, to):
                # An arc that would close a cycle is not a candidate at all.
                # The check is against the *original* network, not the one
                # with this pair's arc removed: R does it that way, and it
                # is the network the strengths are relative to.
                if path_exists(network, to, frm, direct=False):
                    return decimal.Decimal(0)

                added = without + [(frm, to)]
                updated = float(search.node_scores(added, [to])[0])

                # a numpy scalar's repr() carries its type name, which
                # Decimal will not parse.
                delta = float(updated - base[to])
                if not np.isfinite(delta):
                    return decimal.Decimal(0)

                return decimal.Decimal(repr(delta)).exp()

            forward = weight(first, second)
            backward = weight(second, first)

            total = one + forward + backward
            if not total.is_finite():
                # every finite weight is discarded and the mass spread over
                # the infinite ones, as R does.
                infinite = [w for w in (backward, one, forward)
                            if not w.is_finite()]
                share = one / len(infinite)
                forward = backward = share
                total = one

            strength[first, second] = float((forward + backward) / total)
            direction[first, second] = (
                0.0 if (one / total) == 1
                else float(forward / (forward + backward)))

    return strength, direction


def bf_strength(network, data, type=None, **extra_args):
    """Arc strengths from Bayes factors, without resampling.

    `boot_strength` learns hundreds of networks to find out how often an arc
    appears.  This asks the same question of a single network by comparing,
    for each pair of nodes, the marginal likelihood with the arc one way,
    the other way, and not at all -- and normalising the three.

    The arithmetic is done in extended precision, as R's is.  A Bayes factor
    between two networks routinely runs past 1e308, so the three unnormalised
    weights overflow a double long before their ratio does.

    Returns a DataFrame with one row per ordered pair, carrying both the
    strength and the direction, so `averaged_network` accepts it directly.
    """
    import decimal

    if not isinstance(network, BayesianNetwork):
        raise TypeError("bf_strength() needs a BayesianNetwork")

    _check_complete(data)

    nodes = [str(c) for c in data.columns]
    if set(nodes) != set(network.nodes):
        raise ValueError("the network and the data have different variables")
    if _undirected(network):
        raise ValueError("the graph is only partially directed")

    if type is None:
        type = {"discrete": "bde", "continuous": "bge",
                "mixed-cg": "bic-cg"}[_data_type(data)]
    type = _check_score(type, data)
    extra = _check_score_args(type, data, extra_args)

    # R works at 200 bits, which is about 60 decimal digits; the extra few
    # here cost nothing and keep the rounding well below what a double sees.
    #
    # The context has to be *installed*, not merely used to build the
    # numbers: Decimal's operators read the thread's context, which defaults
    # to 28 digits.  At 28 digits, 1 + 4e-30 rounds to exactly 1, and the
    # direction of every improbable arc comes out as a tie rather than as
    # the near-certainty it is.
    with decimal.localcontext() as context:
        context.prec = 70
        context.Emax = decimal.MAX_EMAX
        context.Emin = decimal.MIN_EMIN

        strength, direction = _bayes_factors(network, data, nodes, type,
                                             extra)

    rows = []
    for frm in network.nodes:
        for to in network.nodes:
            if frm == to:
                continue
            key = (frm, to) if (frm, to) in strength else (to, frm)
            forward = (direction[key] if key == (frm, to)
                       else 1 - direction[key])
            rows.append((frm, to, strength[key], forward))

    result = pd.DataFrame(rows, columns=["from", "to", "strength",
                                         "direction"])
    result.attrs.update(nodes=nodes, method="bootstrap", criterion=type)
    result.attrs["threshold"] = inclusion_threshold(result)
    return result
