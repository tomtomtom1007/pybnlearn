"""Parameter learning.

A port of the parts of bnlearn's R/fit.R that estimate the parameters of a
network whose structure is already known: conditional probability tables for
discrete networks, and a linear regression per node for Gaussian ones.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import (discrete_parameters, gaussian_parameters,
                    predict_bayes_lw, predict_parents)
from .structure import _check_complete, _data_type

__all__ = ["DiscreteNode", "FittedNetwork", "GaussianNode", "fit",
           "predict"]


class DiscreteNode:
    """One node of a fitted discrete network: its conditional probability
    table, indexed by the node's own level and then by its parents'."""

    def __init__(self, node, parents, children, table):
        self.node = node
        self.parents = list(parents)
        self.children = list(children)
        self.probabilities = table["values"]
        self.variables = table["variables"]
        self.levels = table["levels"]

    def as_frame(self):
        """The table as a long-format DataFrame, one row per cell."""
        index = pd.MultiIndex.from_product(self.levels, names=self.variables)
        return pd.DataFrame(
            {"probability": self.probabilities.reshape(-1, order="F")},
            index=index).reset_index()

    def __repr__(self):
        shape = "x".join(str(n) for n in self.probabilities.shape)
        return (f"DiscreteNode({self.node!r}, parents={self.parents}, "
                f"table={shape})")


class GaussianNode:
    """One node of a fitted Gaussian network: the regression of the node on
    its parents."""

    def __init__(self, node, parents, children, fitted):
        self.node = node
        self.parents = list(parents)
        self.children = list(children)
        self.coefficients = fitted["coefficients"]
        self.sd = float(fitted["sd"])
        self.residuals = fitted.get("residuals")
        self.fitted_values = fitted.get("fitted.values")

    def __repr__(self):
        return (f"GaussianNode({self.node!r}, parents={self.parents}, "
                f"sd={self.sd:.6g})")


class FittedNetwork:
    """A network with parameters attached."""

    def __init__(self, nodes, method, learning=None):
        self._nodes = dict(nodes)
        self.method = method
        # carried over from the structure, so that a fitted classifier still
        # knows which node is the class.
        self.learning = dict(learning or {})

    @property
    def nodes(self):
        return list(self._nodes)

    def __getitem__(self, node):
        return self._nodes[node]

    def __iter__(self):
        return iter(self._nodes.values())

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, node):
        return node in self._nodes

    def __repr__(self):
        return (f"FittedNetwork({len(self._nodes)} nodes, "
                f"method={self.method!r})")


def _check_method(method, data):
    """check.fitting.method(): the default depends on the data type, and the
    Bayesian estimator only applies to discrete data."""
    kind = _data_type(data)

    if method is None:
        return {"discrete": "mle", "continuous": "mle-g",
                "mixed-cg": "mle-cg"}[kind]

    if method in ("mle", "bayes") and kind != "discrete":
        raise ValueError(
            f"method {method!r} may only be used with discrete data")
    if method == "mle-g" and kind != "continuous":
        raise ValueError(
            f"method {method!r} may only be used with continuous data")
    if method not in ("mle", "bayes", "mle-g"):
        raise ValueError(
            f"method {method!r} is not implemented yet; available methods are "
            "mle and bayes for discrete data, mle-g for continuous data")

    return method


def fit(network, data, method=None, iss=1, keep_fitted=True,
        replace_unidentifiable=False):
    """Estimate the parameters of a network, as bnlearn's bn.fit() does.

    Parameters
    ----------
    network : BayesianNetwork
        Must be completely directed and acyclic.
    data : pandas.DataFrame
    method : str, optional
        "mle" or "bayes" for discrete data, "mle-g" for continuous data.
        Defaults to maximum likelihood for the data type.
    iss : float
        The imaginary sample size for `method="bayes"`.
    keep_fitted : bool
        Keep the residuals and fitted values of the Gaussian regressions.
    replace_unidentifiable : bool
        Replace parameters that the data cannot identify -- a conditional
        distribution with no observations, or a regression coefficient on a
        collinear parent -- with zeros instead of leaving them NaN.

    Returns
    -------
    FittedNetwork
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    _check_complete(data)

    nodes = [str(c) for c in data.columns]
    if set(nodes) != set(network.nodes):
        raise ValueError("the network and the data have different variables")

    # bn.fit() needs a DAG: an undirected or partially directed graph does not
    # say which variable each node is conditioned on.
    undirected = [(a, b) for a, b in network.arcs if (b, a) in network.arcs]
    if undirected:
        raise ValueError(
            "the network is only partially directed; "
            f"{undirected[0][0]} - {undirected[0][1]} has no direction. "
            "Use pdag2dag() to pick a consistent extension first.")

    method = _check_method(method, data)

    parents = {node: [] for node in nodes}
    children = {node: [] for node in nodes}
    for a, b in network.arcs:
        parents[b].append(a)
        children[a].append(b)

    # Order the parents by the *network's* node order, not the data's column
    # order: that is what cache.structure() does, and the order decides the
    # axes of the conditional probability table, so getting it wrong permutes
    # the table rather than merely relabelling it.
    rank = {node: i for i, node in enumerate(network.nodes)}
    for node in nodes:
        parents[node].sort(key=rank.__getitem__)
        children[node].sort(key=rank.__getitem__)

    # Iterate in the network's node order, not the data's: that order becomes
    # the order of the fitted network, and rbn() samples the nodes in it, so
    # using the data's order would draw the random numbers in a different
    # sequence from R for any network whose nodes are ordered differently --
    # which model2network() guarantees, since it sorts them.
    fitted = {}
    for node in network.nodes:
        if method in ("mle", "bayes"):
            table = discrete_parameters(
                data, node, parents[node],
                iss=(float(iss) if method == "bayes" else None),
                replace_unidentifiable=replace_unidentifiable)
            fitted[node] = DiscreteNode(node, parents[node], children[node],
                                        table)
        else:
            result = gaussian_parameters(
                data, node, parents[node], keep_fitted=keep_fitted,
                replace_unidentifiable=replace_unidentifiable)
            fitted[node] = GaussianNode(node, parents[node], children[node],
                                        result)

    return FittedNetwork(fitted, method, network.learning)


def predict(fitted, node, data, method="parents", predictors=None, n=500,
            prob=False):
    """Predict one variable from the others, as bnlearn's predict() does.

    Parameters
    ----------
    fitted : FittedNetwork
    node : str
        The variable to predict.
    data : pandas.DataFrame
        The observed values.  The predicted node may be present; it is ignored.
    method : {"parents", "bayes-lw"}
        "parents" conditions only on the node's parents and is exact given
        them.  "bayes-lw" conditions on every other observed variable, by
        likelihood weighting, so it is a Monte Carlo estimate -- seed with
        `set_seed()` to make it reproducible.
    predictors : sequence of str, optional
        For "bayes-lw", which variables to condition on; defaults to every
        other variable present in both the data and the network.
    n : int
        Particles per observation for "bayes-lw".
    prob : bool
        Also return the probability of each level.  Discrete nodes only.

    Returns
    -------
    The predicted values, or `(values, probabilities)` when `prob` is set,
    where `probabilities` is a DataFrame with one column per level.
    """
    if not isinstance(fitted, FittedNetwork):
        raise TypeError("predict() needs a fitted network; call fit() first")
    if node not in fitted:
        raise ValueError(f"unknown node {node!r}")
    if method not in ("parents", "bayes-lw"):
        if method == "exact":
            raise NotImplementedError(
                "exact prediction goes through the gRain package in bnlearn "
                "and is not ported; use method='bayes-lw' for an approximation")
        raise ValueError("method must be 'parents' or 'bayes-lw'")

    if prob and not isinstance(fitted[node], DiscreteNode):
        raise ValueError(
            "prediction probabilities are only available for discrete nodes")

    if method == "parents":
        needed = list(fitted[node].parents)
        missing = [p for p in needed if p not in data.columns]
        if missing:
            raise ValueError(
                f"predicting {node!r} from its parents needs "
                + ", ".join(missing) + ", which the data do not have")
        values, probabilities = predict_parents(fitted, node, data, prob=prob)
    else:
        if predictors is None:
            predictors = [str(c) for c in data.columns
                          if str(c) != node and str(c) in fitted.nodes]
        else:
            predictors = [str(p) for p in predictors]
            if node in predictors:
                raise ValueError(
                    f"{node!r} is both a predictor and the node being "
                    "predicted")
        values, probabilities = predict_bayes_lw(
            fitted, node, data, predictors, n=int(n), prob=prob)

    if not prob:
        return values

    table, levels = probabilities
    return values, pd.DataFrame(table, columns=levels, index=data.index)
