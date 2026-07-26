"""How much a network says, and how far two networks are apart.

Entropy and Kullback-Leibler divergence, mirroring bnlearn's
R/kullback.leibler.R.  Both are properties of the distribution a network
encodes rather than of its graph, so both need inference: the discrete case
needs the joint distribution of every node's parents, which is what the
junction tree in exact.py provides (bnlearn asks gRain for the same thing).

The Gaussian case is closed-form, and is done on the Cholesky factors rather
than on the covariance matrices -- which is not just an optimisation.  The
factors are what `gbn2mvnorm` already builds, and a determinant read off a
triangular factor is exact where one computed from the covariance matrix
would lose the very smallness that tells a singular network apart from a
nearly-singular one.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import math

import numpy as np

from .fit import DiscreteNode, FittedNetwork, GaussianNode

__all__ = ["H", "KL"]


def H(P):
    """Shannon's entropy of the distribution a network encodes.

    How much uncertainty is left once the network is believed -- in nats,
    since the logarithms are natural, as R's are.
    """
    _check_fitted(P, "H")

    if _is_discrete(P):
        return _discrete_entropy(P)
    if _is_gaussian(P):
        return _gaussian_entropy(P)

    raise NotImplementedError(
        "the entropy of a conditional Gaussian network is not implemented "
        "yet; the network must be all discrete or all Gaussian")


def KL(P, Q):
    """The Kullback-Leibler divergence from Q to P.

    How much is lost by using Q when P is the truth.  Not a distance: it is
    not symmetric, and KL(P, Q) is generally not KL(Q, P).
    """
    _check_fitted(P, "KL")
    _check_fitted(Q, "KL")

    if set(P.nodes) != set(Q.nodes):
        raise ValueError("the two networks have different node sets")

    if _is_discrete(P) and _is_discrete(Q):
        return _discrete_kl(P, Q)
    if _is_gaussian(P) and _is_gaussian(Q):
        return _gaussian_kl(P, Q)

    raise NotImplementedError(
        "both networks must be all discrete or all Gaussian")


def _check_fitted(x, name):
    if not isinstance(x, FittedNetwork):
        raise TypeError(f"{name}() needs a fitted network; call fit() first")


def _is_discrete(x):
    return all(isinstance(x[n], DiscreteNode) for n in x.nodes)


def _is_gaussian(x):
    return all(isinstance(x[n], GaussianNode) for n in x.nodes)


# ---------------------------------------------------------------------------
# discrete networks
# ---------------------------------------------------------------------------

def _discrete_entropy(P):
    """The entropy of a discrete network, node by node.

    A root node contributes the textbook formula.  A node with parents
    contributes the entropy of each of its conditional distributions,
    weighted by how likely that configuration of the parents is -- and *that*
    is what needs inference, because the parents' joint distribution is not
    in the tables.
    """
    from .exact import _JunctionTree

    for node in P.nodes:
        if not np.isfinite(P[node].probabilities).all():
            return float("nan")

    tree = _JunctionTree(P) if any(P[n].parents for n in P.nodes) else None

    total = 0.0
    for node in P.nodes:
        entry = P[node]
        table = entry.probabilities

        if not entry.parents:
            positive = table[table != 0]
            total += -float(np.sum(positive * np.log(positive)))
            continue

        weights = tree.marginal(list(entry.parents))

        contributions = np.where(table == 0, 0.0,
                                 table * np.log(np.where(table == 0, 1.0,
                                                         table)))
        conditional = -contributions.sum(axis=0)

        total += float(np.sum(weights.values * conditional))

    return total


def _discrete_kl(P, Q):
    """KL for discrete networks, as the difference of two log-likelihoods.

    The cross term is the awkward one: it needs the joint distribution *P*
    assigns to each node's family *in Q*, which is a different family from
    the one P's own tables are indexed by.
    """
    from .exact import _JunctionTree

    for network in (P, Q):
        for node in network.nodes:
            if not np.isfinite(network[node].probabilities).all():
                return float("nan")

    tree = _JunctionTree(P)

    cross = 0.0
    for node in Q.nodes:
        entry = Q[node]
        family = [node] + list(entry.parents)

        joint = tree.marginal(family)

        # the marginal comes back in whatever order the tree found
        # convenient; the table is indexed by the family.
        aligned = joint.reorder(family)

        # A zero in Q's table is a -Inf here, and it is left alone rather
        # than swept to zero: if P gives that configuration any weight at
        # all, Q is infinitely wrong about it, and the divergence should say
        # so.  Where P gives it no weight either, 0 * -Inf is NaN, and R
        # propagates that too -- a network with an unidentifiable parameter
        # has no divergence rather than a divergence of zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log(entry.probabilities)
            cross += float(np.sum(aligned.values * logs))

    own = -_discrete_entropy(P)

    # the two terms are the same number when the networks agree, and
    # subtracting them then leaves rounding rather than zero.
    if math.isclose(own, cross, rel_tol=1.5e-8, abs_tol=1.5e-8):
        return 0.0

    return own - cross


# ---------------------------------------------------------------------------
# Gaussian networks
# ---------------------------------------------------------------------------

def _gaussian_entropy(P):
    """Each node contributes the entropy of its own residual noise, and they
    add up: the joint entropy of a Gaussian network is the sum."""
    total = 0.0
    for node in P.nodes:
        sd = float(P[node].sd)
        total += 0.5 * math.log(2 * math.pi * sd ** 2) + 0.5
    return total


def _cholesky_factor(fitted):
    """The mean and the lower Cholesky factor of the joint distribution.

    The same accumulation gbn2mvnorm() does, kept as the factor rather than
    squared up into a covariance matrix.
    """
    from .mvnorm import _ordering

    order = _ordering(fitted)
    position = {node: i for i, node in enumerate(order)}

    mean = {node: 0.0 for node in fitted.nodes}
    factor = np.zeros((len(order), len(order)))

    for node in order:
        entry = fitted[node]
        coefficients = entry.coefficients

        mean[node] = float(coefficients["(Intercept)"]) + sum(
            mean[p] * float(coefficients[p]) for p in entry.parents)

        row = position[node]
        factor[row, row] = entry.sd
        for parent in entry.parents:
            factor[row] += float(coefficients[parent]) * factor[position[parent]]

    return order, np.array([mean[node] for node in order]), factor


def _gaussian_kl(P, Q):
    """The closed form, on the Cholesky factors.

    A singular network has no density, so the divergence is infinite in one
    direction and minus infinite in the other; R reports both rather than
    failing, and so does this.
    """
    order_p, mean_p, factor_p = _cholesky_factor(P)
    order_q, mean_q, factor_q = _cholesky_factor(Q)

    # both are in their own topological order; put Q into P's.
    index = [order_q.index(node) for node in order_p]
    mean_q = mean_q[index]
    factor_q = factor_q[np.ix_(index, index)]

    if (np.allclose(mean_p, mean_q, rtol=1.5e-8, atol=1.5e-8)
            and np.allclose(factor_p, factor_q, rtol=1.5e-8, atol=1.5e-8)):
        return 0.0

    if not (np.isfinite(mean_p).all() and np.isfinite(factor_p).all()
            and np.isfinite(mean_q).all() and np.isfinite(factor_q).all()):
        return float("nan")

    determinant_p = float(np.prod(np.diag(factor_p))) ** 2
    determinant_q = float(np.prod(np.diag(factor_q))) ** 2

    if determinant_p == 0:
        return float("inf")
    if determinant_q == 0:
        return float("-inf")

    inverse = np.linalg.inv(factor_q)
    trace = float(np.sum(_zapsmall(inverse @ factor_p) ** 2))

    offset = _zapsmall(inverse) @ (mean_q - mean_p)
    quadratic = float(offset @ offset)

    return 0.5 * (trace + quadratic - len(order_p)
                  + math.log(determinant_q / determinant_p))


def _zapsmall(x, digits=7):
    """R's zapsmall(): round away what is small next to the largest entry.

    Reproduced rather than skipped because it decides whether a
    near-singular matrix contributes a huge number or nothing at all, and
    the answer changes with it.
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    largest = np.max(np.abs(finite)) if finite.size else 0.0

    if largest > 0:
        places = max(0.0, digits - math.log10(largest))
    else:
        places = float(digits)

    # R's round() truncates the number of digits towards the nearest integer.
    return np.round(x, int(math.floor(places + 0.5)))
