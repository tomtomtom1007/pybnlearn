"""Gaussian networks as multivariate normals, and back.

A Gaussian network is a multivariate normal written in a particular
factorised form, so exact inference on one needs no junction tree: build the
global mean and covariance, then condition.  This mirrors bnlearn's
R/mvnorm.R.

The two directions are not symmetric.  Going to the multivariate normal is
one pass in topological order, accumulating the Cholesky factor row by row.
Coming back needs a Cholesky decomposition, and that is where the awkward
part is: a root node with zero variance, or a leaf node with no residual
variance, makes the covariance matrix singular.  bnlearn patches the
diagonal in both cases -- the patch provably does not change any other local
distribution -- and undoes it afterwards.  Those patches are reproduced here
exactly, because dropping them would turn networks R handles into errors.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import topological_order
from .fit import FittedNetwork, GaussianNode

__all__ = ["MultivariateNormal", "gbn2mvnorm", "mvnorm2gbn"]


class MultivariateNormal:
    """A normal distribution over named variables."""

    __slots__ = ("variables", "mean", "cov")

    def __init__(self, variables, mean, cov):
        self.variables = [str(v) for v in variables]
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)

    @property
    def sd(self):
        """The marginal standard deviations."""
        return np.sqrt(np.diag(self.cov))

    def marginal(self, variables):
        """The distribution of a subset, which for a normal is just the
        corresponding entries of the mean and the covariance."""
        index = self._index(variables)
        return MultivariateNormal([self.variables[i] for i in index],
                                  self.mean[index],
                                  self.cov[np.ix_(index, index)])

    def condition(self, evidence):
        """The distribution given observed values of some of the variables.

        The mean is what bnlearn's conditional.mvnorm() computes, down to the
        pseudoinverse: a plain solve would fail on the singular covariance
        matrices that networks with deterministic nodes produce.
        """
        evidence = {str(k): float(v) for k, v in (evidence or {}).items()}
        unknown = set(evidence) - set(self.variables)
        if unknown:
            raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))

        if not evidence:
            return MultivariateNormal(self.variables, self.mean, self.cov)

        from_ = self._index(evidence)
        to = [i for i in range(len(self.variables))
              if self.variables[i] not in evidence]

        value = np.array([evidence[self.variables[i]] for i in from_])
        ginv = _pseudoinverse(self.cov[np.ix_(from_, from_)])
        cross = self.cov[np.ix_(to, from_)]

        # bnlearn stops at the mean; the covariance is the other half of the
        # same conditioning, and callers that want a distribution rather than
        # a point prediction need it.
        mean = self.mean[to] + cross @ ginv @ (value - self.mean[from_])
        cov = self.cov[np.ix_(to, to)] - cross @ ginv @ cross.T

        return MultivariateNormal([self.variables[i] for i in to], mean, cov)

    def to_frame(self):
        """The mean and standard deviation of each variable, as a table."""
        return pd.DataFrame({"mean": self.mean, "sd": self.sd},
                            index=pd.Index(self.variables, name="node"))

    def _index(self, variables):
        if isinstance(variables, str):
            variables = [variables]
        position = {v: i for i, v in enumerate(self.variables)}
        missing = [str(v) for v in variables if str(v) not in position]
        if missing:
            raise ValueError("unknown node(s): " + ", ".join(sorted(missing)))
        return [position[str(v)] for v in variables]

    def __repr__(self):
        return (f"MultivariateNormal({len(self.variables)} variables: "
                + ", ".join(self.variables[:6])
                + (", ..." if len(self.variables) > 6 else "") + ")")


def _pseudoinverse(matrix):
    """The Moore-Penrose inverse, cut off where bnlearn cuts it off.

    R's conditional.mvnorm() keeps the singular values above
    .Machine$double.eps times the largest one, which is not what
    numpy.linalg.pinv defaults to, so the SVD is done by hand.
    """
    u, d, vh = np.linalg.svd(np.atleast_2d(matrix))

    positive = d > max(np.finfo(float).eps * d[0], 0.0) if d.size else d > 0
    if not positive.any():
        return np.zeros_like(np.atleast_2d(matrix))

    return vh[positive].T @ ((1.0 / d[positive])[:, None] * u[:, positive].T)


def _ordering(fitted):
    """The topological ordering of a fitted network, R's way."""
    arcs = [(parent, node) for node in fitted.nodes
            for parent in fitted[node].parents]
    return topological_order(list(fitted.nodes), arcs)


def gbn2mvnorm(fitted):
    """The global distribution of a Gaussian network.

    Each node is its regression on its parents plus independent noise, so
    substituting the parents' definitions recursively gives one multivariate
    normal.  Doing that in topological order builds the Cholesky factor of
    the covariance a row at a time, which is what the loop below is.

    Parameters
    ----------
    fitted : FittedNetwork
        A network whose nodes are all Gaussian.

    Returns
    -------
    MultivariateNormal
        Its variables are in the network's node order, as R's are.
    """
    if not isinstance(fitted, FittedNetwork):
        raise TypeError("gbn2mvnorm() needs a fitted network; call fit() first")

    names = list(fitted.nodes)
    for node in names:
        if not isinstance(fitted[node], GaussianNode):
            raise ValueError(
                f"{node!r} is not a Gaussian node; gbn2mvnorm() needs a "
                "network whose nodes are all continuous with continuous "
                "parents")

    order = _ordering(fitted)
    position = {node: i for i, node in enumerate(order)}

    mean = {node: 0.0 for node in names}
    # lower-triangular in topological order, so that a node's row is finished
    # by the time any of its children needs it.
    factor = np.zeros((len(order), len(order)))

    for node in order:
        entry = fitted[node]
        parents = list(entry.parents)
        coefficients = entry.coefficients

        mean[node] = float(coefficients["(Intercept)"]) + sum(
            mean[parent] * float(coefficients[parent]) for parent in parents)

        row = position[node]
        factor[row, row] = entry.sd
        for parent in parents:
            factor[row] += float(coefficients[parent]) * factor[position[parent]]

    covariance = factor @ factor.T

    # back from topological order into the network's own node order.
    index = [position[node] for node in names]
    return MultivariateNormal(names,
                              np.array([mean[node] for node in names]),
                              covariance[np.ix_(index, index)])


def mvnorm2gbn(dag, mean, cov, nodes=None):
    """Factorise a multivariate normal into a Gaussian network.

    The inverse of `gbn2mvnorm`, for a given structure: the arcs say which
    regressions to read off the joint distribution.

    Parameters
    ----------
    dag : BayesianNetwork
        A completely directed acyclic graph.
    mean : sequence of float or dict
        The mean vector, in `nodes` order if given, else in the graph's.
    cov : array_like or pandas.DataFrame
        The covariance matrix, ordered to match `mean`.
    nodes : sequence of str, optional
        The order `mean` and `cov` are in, when it is not the graph's.

    Returns
    -------
    FittedNetwork
    """
    from .structure import BayesianNetwork

    if not isinstance(dag, BayesianNetwork):
        raise TypeError("mvnorm2gbn() needs a BayesianNetwork")

    names = list(dag.nodes)

    if isinstance(mean, dict):
        nodes = list(mean) if nodes is None else nodes
        mean = [mean[node] for node in nodes]
    if isinstance(cov, pd.DataFrame):
        nodes = [str(c) for c in cov.columns] if nodes is None else nodes
        cov = cov.to_numpy()
    if nodes is None:
        nodes = names

    nodes = [str(n) for n in nodes]
    if sorted(nodes) != sorted(names):
        raise ValueError(
            "the variables of the distribution do not match the nodes of the "
            "graph")

    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (len(names), len(names)):
        raise ValueError(
            f"the covariance matrix must be {len(names)}x{len(names)}")

    parents = {node: dag.parents(node) for node in names}
    children = {node: dag.children(node) for node in names}
    roots = [node for node in names if not parents[node]]
    leaves = [node for node in names if not children[node]]

    order = dag.topological_order()
    given = {node: i for i, node in enumerate(nodes)}
    index = [given[node] for node in order]

    mean = mean[index]
    cov = cov[np.ix_(index, index)].copy()
    at = {node: i for i, node in enumerate(order)}

    # A root node with zero variance has zero covariance with everything, so
    # giving it a variance of one leaves every other local distribution
    # alone -- and makes the matrix invertible.
    deterministic = [node for node in roots if cov[at[node], at[node]] == 0]
    for node in deterministic:
        cov[at[node], at[node]] = 1.0

    # Same trick for a leaf with no residual variance, except that a leaf can
    # legitimately have any variance, so this one is added unconditionally
    # and subtracted back off at the end.
    for node in leaves:
        cov[at[node], at[node]] += 1.0

    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        nudge = np.sqrt(np.finfo(float).eps)
        cov[np.diag_indices_from(cov)] += nudge
        factor = np.linalg.cholesky(cov)

    diagonal = np.diag(factor).copy()
    # rounding can leave a patched leaf just under one, which would make the
    # square root below the square root of a negative number.
    for node in leaves:
        if diagonal[at[node]] < 1.0:
            diagonal[at[node]] = 1.0

    if len(names) == 1:
        rho = np.zeros((1, 1))
    else:
        rho = np.eye(len(names)) - np.diag(np.diag(factor)) @ np.linalg.inv(
            factor)

    fitted = {}
    for node in names:
        row = at[node]
        coefficients = {parent: float(rho[row, at[parent]])
                        for parent in parents[node]}
        intercept = mean[row] - sum(
            mean[at[parent]] * coefficients[parent]
            for parent in parents[node])

        if node in deterministic:
            sd = 0.0
        elif node in leaves:
            sd = float(np.sqrt(diagonal[row] ** 2 - 1.0))
        else:
            sd = float(diagonal[row])

        fitted[node] = GaussianNode(
            node, parents[node], children[node],
            {"coefficients": {"(Intercept)": float(intercept), **coefficients},
             "sd": sd})

    return FittedNetwork(fitted, "mle-g", dag.learning)
