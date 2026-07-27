"""Filling in missing values, and learning from data that has them.

This covers bnlearn's R/impute.R, R/most.probable.explanation.R and the
structural EM in R/frontend-missingdata.R.

Three ways to fill a gap, and they differ in what they are allowed to look
at.  "parents" uses only the node's parents, which is fast and throws away
everything the rest of the observation says.  "bayes-lw" conditions on every
observed value in the row, by sampling.  "exact" conditions on the same
thing but computes rather than samples, and is the only one that returns the
most probable *joint* explanation rather than each variable's own best guess.

The exact method is not one big query.  `_partition` splits the missing
variables into groups that can be answered separately -- a variable whose
Markov blanket is fully observed needs nothing else -- which is what keeps
the tables from growing with the number of gaps rather than with the size of
the largest group.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fit import DiscreteNode, FittedNetwork, GaussianNode, fit

__all__ = ["impute", "structural_em"]

_METHODS = ("parents", "bayes-lw", "exact")


def impute(fitted, data, method="bayes-lw", n=500, strict=True):
    """Fill in the missing values of a data set from a fitted network.

    Parameters
    ----------
    fitted : FittedNetwork
    data : pandas.DataFrame
        May contain missing values; complete rows are left alone.
    method : {"bayes-lw", "parents", "exact"}
        How much of each observation to condition on, and whether to sample
        or compute.  See the module docstring.
    n : int
        Particles per observation, for "bayes-lw".  Seed with `set_seed()`.
    strict : bool
        Raise if any value could not be imputed.  A gap survives when the
        network gives its evidence probability zero, or when a variable is
        latent -- never observed at all.

    Returns
    -------
    pandas.DataFrame
    """
    if not isinstance(fitted, FittedNetwork):
        raise TypeError("impute() needs a fitted network; call fit() first")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if method not in _METHODS:
        raise ValueError("method must be one of " + ", ".join(_METHODS))

    missing = set(fitted.nodes) - {str(c) for c in data.columns}
    if missing:
        raise ValueError(
            "the data are missing the variable(s) "
            + ", ".join(sorted(missing)))

    if not len(data):
        return data.copy()

    if method == "parents":
        imputed = _impute_from_parents(fitted, data)
    elif method == "bayes-lw":
        imputed = _impute_by_weighting(fitted, data, int(n))
    else:
        imputed = _impute_exactly(fitted, data)

    if strict and imputed[list(fitted.nodes)].isna().any().any():
        raise ValueError(
            "imputation unsuccessful, the data still contain missing values")

    return imputed


def _incomplete_rows(data, nodes):
    return np.flatnonzero(data[list(nodes)].isna().any(axis=1).to_numpy())


def _impute_from_parents(fitted, data):
    """impute.backend.parents(): predict each node from its parents.

    Topological order is what makes this work at all: by the time a node is
    reached its parents have already been filled in, so a chain of gaps
    resolves in one pass rather than needing several.
    """
    from ._core import predict_parents
    from .nodes import _graph

    out = data.copy()

    for node in _graph(fitted).topological_order():
        gaps = out[node].isna().to_numpy()
        if not gaps.any():
            continue

        parents = list(fitted[node].parents)
        values, _ = predict_parents(fitted, node, out.loc[gaps, parents],
                                    prob=False)
        out[node] = _filled(out[node], gaps, values)

    return out


def _filled(column, gaps, values):
    """Put predicted values into a column's gaps, keeping its type.

    A categorical column cannot be assigned into piecemeal when the network
    knows a level the data have never shown -- which is exactly the case
    with a latent variable -- so the column is rebuilt from the network's
    levels rather than the data's.
    """
    if not isinstance(column.dtype, pd.CategoricalDtype):
        out = column.to_numpy(dtype=float).copy()
        out[gaps] = np.asarray(values, dtype=float)
        return out

    filled = column.astype(object).to_numpy().copy()
    filled[gaps] = np.asarray(values).astype(str)

    categories = list(column.cat.categories)
    for level in dict.fromkeys(np.asarray(values).astype(str)):
        if level not in categories:
            categories.append(level)

    return pd.Categorical(filled, categories=categories)


def _impute_by_weighting(fitted, data, n):
    """impute.backend.likelihood.weighting(): condition on everything the
    row does say, by sampling.

    A discrete variable is filled with the level that carries the most
    weight and a continuous one with the weighted mean, which is why this
    imputes each variable separately rather than jointly.
    """
    from .inference import cpdist

    out = data.copy()
    nodes = list(fitted.nodes)

    for position in _incomplete_rows(data, nodes):
        row = out.iloc[position]
        observed = [node for node in nodes if not pd.isna(row[node])]
        wanted = [node for node in nodes if node not in observed]

        evidence = {node: (str(row[node])
                           if isinstance(fitted[node], DiscreteNode)
                           else float(row[node]))
                    for node in observed}

        particles, weights = cpdist(fitted, wanted, evidence or None,
                                    method="lw", n=n)

        weights = np.nan_to_num(np.asarray(weights, dtype=float))

        for node in wanted:
            column = particles[node]

            if isinstance(fitted[node], DiscreteNode):
                totals = {}
                for level, weight in zip(np.asarray(column).astype(str),
                                         weights):
                    totals[level] = totals.get(level, 0.0) + weight
                if not totals or max(totals.values()) <= 0:
                    continue
                value = max(totals, key=totals.get)
            else:
                total = weights.sum()
                if total <= 0:
                    continue
                value = float(np.average(column.to_numpy(dtype=float),
                                         weights=weights))

            _assign(out, position, node, value)

    return out


def _assign(frame, position, node, value):
    """Write one imputed value, widening the column's categories if the
    network knows a level the data never showed."""
    column = frame[node]
    if isinstance(column.dtype, pd.CategoricalDtype) \
            and value not in column.cat.categories:
        frame[node] = column.cat.add_categories([value])

    frame.iloc[position, frame.columns.get_loc(node)] = value


def _impute_exactly(fitted, data):
    """impute.backend.exact(): the most probable explanation of the gaps.

    Unlike the other two this is a joint answer: the group of variables a
    partition puts together is filled with the combination that is most
    likely as a whole, not with each variable's own most likely value.
    """
    discrete = all(isinstance(fitted[n], DiscreteNode) for n in fitted.nodes)
    gaussian = all(isinstance(fitted[n], GaussianNode) for n in fitted.nodes)

    if discrete:
        return _impute_exactly_discrete(fitted, data)
    if gaussian:
        return _impute_exactly_gaussian(fitted, data)

    raise NotImplementedError(
        "exact imputation needs a network that is all discrete or all "
        "Gaussian")


def _partition(fitted, wanted, observed):
    """query.partitioning(): break the gaps into independent subqueries.

    A missing variable whose Markov blanket is fully observed can be
    answered on its own; one whose blanket contains another missing variable
    has to be answered together with it.  What is left over -- variables
    whose blanket is not observable at all -- goes into a single query
    conditioned on whatever is available.
    """
    from .nodes import _graph, mb

    graph = _graph(fitted)
    blankets = {node: mb(graph, node) for node in fitted.nodes}

    remaining = list(wanted)
    queries = []
    catchall = []

    while remaining:
        group = [remaining[0]]
        blanket = list(blankets[group[0]])

        # pull in every other missing variable the blanket touches, and keep
        # going: their blankets may reach further still.
        while set(remaining) & set(blanket):
            group += [n for n in remaining if n in blanket and n not in group]
            blanket = [n for node in group for n in blankets[node]]
            blanket = [n for n in dict.fromkeys(blanket) if n not in group]

        remaining = [n for n in remaining if n not in group]

        if all(node in observed for node in blanket):
            queries.append((group, blanket))
        else:
            catchall.extend(group)

    if catchall:
        evidence = list(observed)

        # one variable left over can often be answered from less than
        # everything, and a smaller conditioning set is a smaller table.
        if len(catchall) == 1:
            from .graph import dsep

            graph = _graph(fitted)
            for node in list(evidence):
                if dsep(graph, catchall[0], node,
                        [n for n in evidence if n != node]):
                    evidence.remove(node)

        queries.append((catchall, evidence))

    return queries


def _impute_exactly_discrete(fitted, data):
    from ._core import sample_indices
    from .exact import _JunctionTree

    out = data.copy()
    nodes = list(fitted.nodes)
    tree = _JunctionTree(fitted)
    threshold = np.sqrt(np.finfo(float).eps)

    for position in _incomplete_rows(data, nodes):
        row = out.iloc[position]
        observed = [node for node in nodes if not pd.isna(row[node])]
        wanted = [node for node in nodes if node not in observed]

        for group, evidence in _partition(fitted, wanted, observed):
            given = {node: str(row[node]) for node in evidence}

            tree.set_evidence(given)
            joint = tree.unnormalised(group)
            if joint.values.sum() <= threshold:
                continue

            # The most probable combination, not each variable's own best.
            # Ties are broken by a draw from R's generator, and the draw
            # happens whether or not there is a tie to break -- so skipping
            # it when the maximum is unique would put the stream out of step
            # for every later observation.
            maxima = np.flatnonzero(
                joint.values.reshape(-1, order="F") == joint.values.max())
            chosen = maxima[int(sample_indices(len(maxima), 1,
                                               replace=False)[0]) - 1]
            where = np.unravel_index(chosen, joint.values.shape, order="F")

            for node, index in zip(group, where):
                _assign(out, position, node, joint.levels[node][index])

        tree.set_evidence(None)

    return out


def _impute_exactly_gaussian(fitted, data):
    from .mvnorm import gbn2mvnorm

    out = data.copy()
    nodes = list(fitted.nodes)

    mvn = gbn2mvnorm(fitted)
    if not (np.isfinite(mvn.mean).all() and np.isfinite(mvn.cov).all()):
        return out

    for position in _incomplete_rows(data, nodes):
        row = out.iloc[position]
        observed = [node for node in nodes if not pd.isna(row[node])]
        wanted = [node for node in nodes if node not in observed]

        for group, evidence in _partition(fitted, wanted, observed):
            given = {node: float(row[node]) for node in evidence}

            # the conditional expectation, which for a normal is also the
            # most probable value.
            conditional = mvn.condition(given).marginal(group)

            for node, value in zip(group, conditional.mean):
                _assign(out, position, node, float(value))

    return out


# ---------------------------------------------------------------------------
# structural EM
# ---------------------------------------------------------------------------

def structural_em(data, maximize="hc", maximize_args=None, method="mle",
                  fit_args=None, impute_method="bayes-lw", impute_args=None,
                  start=None, max_iter=5, return_all=False):
    """Learn a structure from data with gaps, by structural EM.

    Alternate two steps: fill the gaps in using the network you have, then
    learn a better network from the completed data.  The imputation is the
    expectation step and the structure learning the maximisation step, and
    the loop stops when the parameters stop changing.

    Parameters
    ----------
    data : pandas.DataFrame
    maximize : str
        The score-based algorithm for the maximisation step.
    maximize_args, fit_args, impute_args : dict, optional
        Passed to structure learning, parameter learning and imputation.
    method : str
        The parameter estimator.
    impute_method : {"bayes-lw", "parents", "exact"}
    start : BayesianNetwork or FittedNetwork, optional
        Where to begin.  A network with parameters is *required* when a
        variable is latent -- never observed -- because there is nothing to
        estimate its distribution from.
    max_iter : int
        How many rounds at most.
    return_all : bool
        Also return the completed data and the fitted network.

    Returns
    -------
    BayesianNetwork, or `(network, imputed, fitted)` when `return_all` is set.
    """
    from . import structure
    from .fit import identifiable
    from .graph import empty_graph
    from .nodes import _graph

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    _MAXIMIZE = {"hc": structure.hc, "tabu": structure.tabu}
    if maximize not in _MAXIMIZE:
        raise ValueError("maximize must be 'hc' or 'tabu'")

    nodes = [str(c) for c in data.columns]
    latent = [node for node in nodes if data[node].isna().all()]

    if latent and not isinstance(start, FittedNetwork):
        raise ValueError(
            "the data contain the latent variable(s) "
            + ", ".join(latent)
            + "; start must be a fitted network, because there is nothing to "
              "estimate their distributions from")

    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    if start is None:
        dag = empty_graph(nodes)
        fitted = fit(dag, data, method=method, **(fit_args or {}))
    elif isinstance(start, FittedNetwork):
        dag, fitted = _graph(start), start
    else:
        dag = start
        fitted = fit(dag, data, method=method, **(fit_args or {}))

    if not identifiable(fitted):
        raise ValueError(
            "the starting network has unidentifiable parameters")

    completed = data
    for _ in range(max_iter):
        # expectation: fill the gaps with the network we have.
        completed = impute(fitted, data, method=impute_method,
                           strict=False, **(impute_args or {}))

        # maximisation: a better structure, then better parameters.
        dag = _MAXIMIZE[maximize](completed, start=dag,
                                  **(maximize_args or {}))
        updated = fit(dag, completed, method=method, **(fit_args or {}))

        if _same_parameters(fitted, updated):
            fitted = updated
            break

        fitted = updated

    dag.learning = dict(dag.learning)
    dag.learning.update(algo="structural.em", maximize=maximize,
                        impute=impute_method, fit=method)

    if return_all:
        return dag, completed, fitted
    return dag


def _same_parameters(a, b):
    """The stopping rule: the fitted network has stopped moving.

    R compares the whole objects with all.equal, which is a tolerance
    comparison on the numbers and an exact one on the structure.
    """
    if set(a.nodes) != set(b.nodes):
        return False

    for node in a.nodes:
        first, second = a[node], b[node]
        if type(first) is not type(second):
            return False
        if list(first.parents) != list(second.parents):
            return False

        if isinstance(first, DiscreteNode):
            if first.probabilities.shape != second.probabilities.shape:
                return False
            if not np.allclose(first.probabilities, second.probabilities,
                               rtol=1.5e-8, atol=1.5e-8, equal_nan=True):
                return False
        else:
            if list(first.coefficients) != list(second.coefficients):
                return False
            if not np.allclose(
                    [first.coefficients[k] for k in first.coefficients],
                    [second.coefficients[k] for k in second.coefficients],
                    rtol=1.5e-8, atol=1.5e-8, equal_nan=True):
                return False
            if not np.allclose(first.sd, second.sd, rtol=1.5e-8, atol=1.5e-8,
                               equal_nan=True):
                return False

    return True
