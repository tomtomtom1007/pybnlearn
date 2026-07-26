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

from ._core import (conditional_gaussian_parameters,
                    discrete_parameters, gaussian_parameters,
                    parent_configurations, predict_bayes_lw,
                    predict_parents)
from .structure import _check_complete, _data_type, is_discrete_column

__all__ = ["ConditionalGaussianNode", "DiscreteNode", "FittedNetwork",
           "GaussianNode", "bn_net", "custom_fit", "fit", "predict"]


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


class ConditionalGaussianNode:
    """A continuous node with at least one discrete parent: one linear
    regression on the continuous parents per configuration of the discrete
    ones, each with its own residual standard deviation."""

    def __init__(self, node, parents, children, discrete_parents,
                 continuous_parents, levels, fitted):
        self.node = node
        self.parents = list(parents)
        self.children = list(children)
        self.discrete_parents = list(discrete_parents)
        self.continuous_parents = list(continuous_parents)
        self.discrete_levels = {k: list(v) for k, v in levels.items()}
        # one column of coefficients per configuration of the discrete parents
        self.coefficients = fitted["coefficients"]
        self.coefficient_names = fitted.get("coefnames")
        self.sd = np.atleast_1d(np.asarray(fitted["sd"], dtype=float))
        self.configurations = fitted.get("configs")
        self.residuals = fitted.get("residuals")
        self.fitted_values = fitted.get("fitted.values")

    @property
    def nconfigurations(self):
        return self.coefficients.shape[1]

    def __repr__(self):
        return (f"ConditionalGaussianNode({self.node!r}, "
                f"discrete={self.discrete_parents}, "
                f"continuous={self.continuous_parents}, "
                f"{self.nconfigurations} regressions)")


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
    if method == "mle-cg" and kind != "mixed-cg":
        raise ValueError(
            f"method {method!r} may only be used with a mixture of discrete "
            "and continuous data")
    if method not in ("mle", "bayes", "mle-g", "mle-cg"):
        raise ValueError(
            f"method {method!r} is not implemented yet; available methods are "
            "mle and bayes for discrete data, mle-g for continuous data, and "
            "mle-cg for a mixture of the two")

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
    def _is_discrete(column):
        return is_discrete_column(data[column])

    fitted = {}
    for node in network.nodes:
        if method == "mle-cg":
            fitted[node] = _fit_mixed(data, node, parents[node],
                                      children[node], _is_discrete,
                                      keep_fitted, replace_unidentifiable)
        elif method in ("mle", "bayes"):
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


def _as_factor(series):
    return (series if isinstance(series.dtype, pd.CategoricalDtype)
            else series.astype("category"))


def _fit_mixed(data, node, parents, children, is_discrete, keep_fitted,
               replace_unidentifiable):
    """bn.fit.backend.mixedcg(): pick the estimator the node's own type and
    its parents' types call for.

    A discrete node is discrete however its parents look; a continuous node
    with only continuous parents is an ordinary regression; only a continuous
    node with at least one discrete parent needs a regression per
    configuration.
    """
    if is_discrete(node):
        table = discrete_parameters(
            data, node, parents, iss=None,
            replace_unidentifiable=replace_unidentifiable)
        return DiscreteNode(node, parents, children, table)

    discrete_parents = [p for p in parents if is_discrete(p)]
    continuous_parents = [p for p in parents if not is_discrete(p)]

    if not discrete_parents:
        result = gaussian_parameters(
            data, node, parents, keep_fitted=keep_fitted,
            replace_unidentifiable=replace_unidentifiable)
        return GaussianNode(node, parents, children, result)

    configs = parent_configurations(data[discrete_parents])
    result = conditional_gaussian_parameters(
        data, node, continuous_parents, configs, keep_fitted=keep_fitted,
        replace_unidentifiable=replace_unidentifiable)

    levels = {p: list(_as_factor(data[p]).cat.categories)
              for p in discrete_parents}
    return ConditionalGaussianNode(node, parents, children, discrete_parents,
                                   continuous_parents, levels, result)


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
    method : {"parents", "bayes-lw", "exact"}
        "parents" conditions only on the node's parents and is exact given
        them.  "bayes-lw" conditions on every other observed variable, by
        likelihood weighting, so it is a Monte Carlo estimate -- seed with
        `set_seed()` to make it reproducible.  "exact" also conditions on
        every other observed variable but computes rather than samples, by
        junction tree for a discrete network and by conditioning the global
        multivariate normal for a Gaussian one.
    predictors : sequence of str, optional
        For "bayes-lw" and "exact", which variables to condition on; defaults
        to every other variable present in both the data and the network.
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
    if method not in ("parents", "bayes-lw", "exact"):
        raise ValueError("method must be 'parents', 'bayes-lw' or 'exact'")

    if prob and not isinstance(fitted[node], DiscreteNode):
        raise ValueError(
            "prediction probabilities are only available for discrete nodes")

    if method == "exact":
        from .exact import exact_predict
        return exact_predict(fitted, node, data, predictors=predictors,
                             prob=prob)

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


# ---------------------------------------------------------------------------
# building a fitted network by hand, from R/custom.fit.R
# ---------------------------------------------------------------------------

def custom_fit(dag, dist):
    """Attach parameters you supply to a structure, without any data.

    This is how a network from a paper, a textbook or an expert gets into
    pybnlearn, and how a simulation study gets a network to sample from.

    `dist` maps each node to its local distribution.  Which kind of node it
    is is read off the shape of what you give, exactly as R reads it:

    * ``{"prob": array, "levels": ...}`` is a discrete node.  The array's
      first axis is the node and the rest are its parents, in the order the
      network lists them.
    * ``{"coef": {"(Intercept)": a, parent: b, ...}, "sd": s}`` is a
      Gaussian node.
    * ``{"coef": 2-d array, "sd": sequence}`` is a conditional Gaussian
      node: one column of coefficients, and one standard deviation, per
      configuration of the discrete parents.

    The nodes are built in topological order, so that by the time a node is
    reached its parents are already typed -- which is what lets a discrete
    parent be rejected for a Gaussian node, and what supplies the levels a
    conditional Gaussian node's configurations are counted from.

    Parameters
    ----------
    dag : BayesianNetwork
        Completely directed and acyclic.
    dist : dict
        One entry per node.

    Returns
    -------
    FittedNetwork
    """
    from .graph import acyclic, directed
    from .structure import BayesianNetwork

    if not isinstance(dag, BayesianNetwork):
        raise TypeError("custom_fit() needs a BayesianNetwork")
    if not directed(dag):
        raise ValueError("the graph is only partially directed")
    if not acyclic(dag, directed=True):
        raise ValueError("the graph contains cycles")

    if not isinstance(dist, dict):
        raise TypeError("the local distributions must be a dict of node -> "
                        "parameters")
    missing = set(dag.nodes) - set(dist)
    extra = set(dist) - set(dag.nodes)
    if missing or extra:
        raise ValueError(
            "the local distributions do not match the nodes of the graph"
            + (": missing " + ", ".join(sorted(missing)) if missing else "")
            + (": unknown " + ", ".join(sorted(extra)) if extra else ""))

    fitted = {}
    for node in dag.topological_order():
        parents = dag.parents(node)
        spec = dist[node]

        if not isinstance(spec, dict):
            raise TypeError(
                f"the local distribution of {node!r} must be a dict")

        if "prob" in spec:
            fitted[node] = _custom_discrete(node, parents, dag.children(node),
                                            spec, fitted)
        elif np.ndim(spec.get("coef")) == 2:
            fitted[node] = _custom_conditional_gaussian(
                node, parents, dag.children(node), spec, fitted)
        else:
            fitted[node] = _custom_gaussian(node, parents, dag.children(node),
                                            spec, fitted)

    method = _custom_method(fitted)
    return FittedNetwork({node: fitted[node] for node in dag.nodes}, method,
                         dag.learning)


def _custom_method(fitted):
    kinds = {type(entry) for entry in fitted.values()}
    if kinds == {DiscreteNode}:
        return "custom"
    if kinds <= {GaussianNode}:
        return "custom-g"
    return "custom-cg"


def _custom_levels(node, parents, spec, fitted):
    """The levels of the node and of its parents, in axis order.

    The parents' levels are not taken from `spec`: they are already fixed by
    the parents' own tables, and a table that disagreed with them would be a
    different network from the one the graph describes.
    """
    variables = [node] + list(parents)
    supplied = spec.get("levels")

    if isinstance(supplied, dict):
        own = supplied.get(node)
    elif supplied is not None:
        own = list(supplied)[0] if len(supplied) else None
    else:
        own = None

    if own is None:
        raise ValueError(
            f"the levels of {node!r} are missing; give them as "
            "levels={node: [...]} alongside prob")

    levels = [[str(v) for v in own]]
    for parent in parents:
        entry = fitted[parent]
        if not isinstance(entry, DiscreteNode):
            raise ValueError(
                f"{node!r} is discrete but has the continuous parent "
                f"{parent!r}")
        levels.append(list(entry.levels[0]))

    return variables, levels


def _custom_discrete(node, parents, children, spec, fitted):
    variables, levels = _custom_levels(node, parents, spec, fitted)

    table = np.asarray(spec["prob"], dtype=float)
    shape = tuple(len(l) for l in levels)

    if table.shape != shape:
        raise ValueError(
            f"the probability table of {node!r} has shape {table.shape}, "
            f"but the network wants {shape} -- the node on the first axis "
            "and its parents after it, in the order the network lists them")
    if (table < 0).any() or not np.isfinite(table).all():
        raise ValueError(
            f"the probabilities of {node!r} are not all finite and "
            "non-negative")

    # R rejects a distribution that is more than 1% away from summing to one
    # and silently rescales the rest, on the grounds that a small gap is
    # rounding in what you typed and a large one is a mistake.
    totals = table.sum(axis=0)
    if (np.abs(totals - 1.0) > 0.01).any():
        raise ValueError(
            f"some conditional probability distributions of {node!r} do not "
            "sum to one")

    return DiscreteNode(node, parents, children,
                        {"values": table / totals, "variables": variables,
                         "levels": levels})


def _custom_gaussian(node, parents, children, spec, fitted):
    for parent in parents:
        if isinstance(fitted[parent], DiscreteNode):
            raise ValueError(
                f"{node!r} is Gaussian but has the discrete parent "
                f"{parent!r}; give it a matrix of coefficients, one column "
                "per configuration, to make it conditional Gaussian")

    wanted = ["(Intercept)"] + list(parents)
    coefficients = _custom_coefficients(node, spec, wanted)

    if "sd" not in spec:
        raise ValueError(f"the residual standard deviation of {node!r} is "
                         "missing")
    sd = float(spec["sd"])
    if not sd >= 0:
        raise ValueError(
            f"the residual standard deviation of {node!r} must be "
            "non-negative")

    return GaussianNode(node, parents, children,
                        {"coefficients": coefficients, "sd": sd})


def _custom_coefficients(node, spec, wanted):
    coefficients = spec.get("coef")
    if coefficients is None:
        raise ValueError(f"the coefficients of {node!r} are missing")

    if isinstance(coefficients, dict):
        if set(coefficients) != set(wanted):
            raise ValueError(
                f"wrong regression coefficients for {node!r}; expected "
                + ", ".join(wanted))
        return {name: float(coefficients[name]) for name in wanted}

    values = np.asarray(coefficients, dtype=float).reshape(-1)
    if len(values) != len(wanted):
        raise ValueError(
            f"{node!r} has {len(values)} coefficients but {len(wanted)} are "
            "needed (the intercept and one per parent)")
    return dict(zip(wanted, (float(v) for v in values)))


def _custom_conditional_gaussian(node, parents, children, spec, fitted):
    discrete_parents = [p for p in parents
                        if isinstance(fitted[p], DiscreteNode)]
    continuous_parents = [p for p in parents if p not in discrete_parents]

    if not discrete_parents:
        raise ValueError(
            f"{node!r} has a matrix of coefficients, so it is conditional "
            "Gaussian, but it has no discrete parent")

    levels = {p: list(fitted[p].levels[0]) for p in discrete_parents}
    configurations = 1
    for parent in discrete_parents:
        configurations *= len(levels[parent])

    coefficients = np.asarray(spec["coef"], dtype=float)
    shape = (1 + len(continuous_parents), configurations)
    if coefficients.shape != shape:
        raise ValueError(
            f"the coefficients of {node!r} have shape "
            f"{coefficients.shape}, but the network wants {shape} -- the "
            "intercept and the continuous parents down the rows, one column "
            "per configuration of the discrete parents")

    sd = np.atleast_1d(np.asarray(spec.get("sd"), dtype=float))
    if len(sd) != configurations:
        raise ValueError(
            f"{node!r} has {len(sd)} standard deviations but "
            f"{configurations} configurations")
    if (sd < 0).any():
        raise ValueError(
            f"the residual standard deviations of {node!r} must be "
            "non-negative")

    return ConditionalGaussianNode(
        node, parents, children, discrete_parents, continuous_parents, levels,
        {"coefficients": coefficients, "sd": sd,
         "coefnames": ["(Intercept)"] + continuous_parents})


def bn_net(fitted):
    """The structure of a fitted network, with the parameters dropped."""
    from .nodes import _graph

    return _graph(fitted)
