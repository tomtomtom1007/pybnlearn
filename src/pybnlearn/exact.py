"""Exact inference by junction tree.

This is the one part of pybnlearn that is not a port.  bnlearn does not
implement exact inference itself: `cpquery(method = "exact")` and
`predict(method = "exact")` hand the network to the gRain package, which is
not vendored here.  So the algorithm below is written from scratch.

That changes what "matching R" can mean.  Everywhere else, agreeing with R
required reproducing its choices -- the order tests are run in, which arc a
search considers first, how a tie is broken -- because those choices change
the answer.  Exact inference has no such freedom: the marginal of a variable
given evidence is a property of the network, not of the algorithm.  Any
correct implementation agrees with gRain to within floating-point error,
whichever elimination order it picks.  The parity suite therefore compares
probabilities to a tolerance rather than reproducing gRain's internals, and
that is a weaker claim than the rest of the suite makes, deliberately.

The implementation is the standard one: moralise, triangulate by a min-fill
heuristic, take the cliques of the triangulated graph, join them into a tree
that satisfies the running intersection property, multiply each conditional
probability table into a clique that covers it, and calibrate the tree with a
collect-distribute pass.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .fit import DiscreteNode, FittedNetwork

__all__ = ["Factor", "query"]


class Factor:
    """A function over a set of discrete variables.

    `values` is an array whose axes line up with `variables`, in that order.
    Everything else here is arithmetic on those axes.
    """

    __slots__ = ("variables", "levels", "values")

    def __init__(self, variables, levels, values):
        self.variables = list(variables)
        self.levels = {v: list(levels[v]) for v in self.variables}
        self.values = np.asarray(values, dtype=np.float64)

        expected = tuple(len(self.levels[v]) for v in self.variables)
        if self.values.shape != expected:
            raise ValueError(
                f"values have shape {self.values.shape}, expected {expected}")

    @property
    def scope(self):
        return set(self.variables)

    def _aligned(self, order):
        """This factor's values, transposed and broadcast to `order`."""
        axes = [self.variables.index(v) for v in order if v in self.variables]
        moved = self.values.transpose(axes)
        shape = [len(self.levels[v]) if v in self.variables else 1
                 for v in order]
        return moved.reshape(shape)

    def __mul__(self, other):
        order = self.variables + [v for v in other.variables
                                  if v not in self.scope]
        levels = {**self.levels, **other.levels}
        return Factor(order, levels,
                      self._aligned(order) * other._aligned(order))

    def marginalise(self, keep):
        """Sum out everything except `keep`.

        The variables that remain keep *this factor's* order, not the order
        they were asked for in; use `reorder()` if that matters.
        """
        keep = [v for v in self.variables if v in set(keep)]
        drop = tuple(i for i, v in enumerate(self.variables)
                     if v not in set(keep))
        summed = self.values.sum(axis=drop) if drop else self.values
        # summing leaves the remaining axes in their original relative order.
        return Factor(keep, self.levels, summed)

    def observe(self, evidence):
        """Zero out the entries that contradict the evidence.

        The variable is kept rather than dropped, so the factor still lines up
        with the clique it lives in; the zeros do the work.
        """
        values = self.values
        for variable, value in evidence.items():
            if variable not in self.scope:
                continue
            axis = self.variables.index(variable)
            levels = self.levels[variable]
            if value not in levels:
                raise ValueError(
                    f"{value!r} is not a level of {variable!r}; "
                    f"levels are {levels}")
            mask = np.zeros(len(levels))
            mask[levels.index(value)] = 1.0
            shape = [1] * values.ndim
            shape[axis] = len(levels)
            values = values * mask.reshape(shape)
        return Factor(self.variables, self.levels, values)

    def reorder(self, variables):
        """The same factor with its axes in the given order."""
        variables = list(variables)
        if variables == self.variables:
            return self
        if set(variables) != self.scope:
            raise ValueError("reorder() needs exactly the same variables")
        shape = tuple(len(self.levels[v]) for v in variables)
        return Factor(variables, self.levels,
                      self._aligned(variables).reshape(shape))

    def normalise(self):
        total = self.values.sum()
        if total == 0:
            raise ZeroDivisionError(
                "the evidence has probability zero under this network, so "
                "nothing can be conditioned on it")
        return Factor(self.variables, self.levels, self.values / total)

    def to_frame(self):
        """The factor in long format, one row per cell."""
        index = pd.MultiIndex.from_product(
            [self.levels[v] for v in self.variables], names=self.variables)
        return pd.DataFrame({"probability": self.values.reshape(-1)},
                            index=index).reset_index()

    def __repr__(self):
        shape = "x".join(str(n) for n in self.values.shape)
        return f"Factor({', '.join(self.variables)}; {shape})"


# ---------------------------------------------------------------------------
# building the tree
# ---------------------------------------------------------------------------

def _moral_graph(fitted):
    """The moral graph: parents joined to their children and to each other."""
    adjacency = {node: set() for node in fitted.nodes}

    for node in fitted.nodes:
        parents = fitted[node].parents
        for parent in parents:
            adjacency[node].add(parent)
            adjacency[parent].add(node)
        # marry the parents: they share a child, so they appear together in
        # its conditional probability table and cannot be separated.
        for a, b in itertools.combinations(parents, 2):
            adjacency[a].add(b)
            adjacency[b].add(a)

    return adjacency


def _triangulate(adjacency, cardinalities):
    """Eliminate the vertices by min-fill, adding the edges that requires.

    Ties are broken on the cluster's state space and then on the name, so the
    result does not depend on dictionary ordering -- worth having even though
    the answer does not depend on the ordering at all, because it makes a
    disagreement reproducible.
    """
    graph = {v: set(neighbours) for v, neighbours in adjacency.items()}
    remaining = set(graph)
    order, cliques = [], []

    while remaining:
        def cost(vertex):
            neighbours = graph[vertex] & remaining
            fill = sum(1 for a, b in itertools.combinations(neighbours, 2)
                       if b not in graph[a])
            size = cardinalities[vertex]
            for n in neighbours:
                size *= cardinalities[n]
            return (fill, size, vertex)

        chosen = min(remaining, key=cost)
        neighbours = graph[chosen] & remaining

        for a, b in itertools.combinations(neighbours, 2):
            graph[a].add(b)
            graph[b].add(a)

        cliques.append(frozenset(neighbours | {chosen}))
        order.append(chosen)
        remaining.discard(chosen)

    return order, cliques


def _maximal(cliques):
    """Drop the clusters contained in another one."""
    out = []
    for clique in sorted(cliques, key=len, reverse=True):
        if not any(clique <= kept for kept in out):
            out.append(clique)
    return out


def _junction_tree(cliques):
    """A maximum-weight spanning tree over shared variables.

    Weighting the clique graph by separator size and taking a maximum spanning
    tree is the standard way to get the running intersection property, which
    is what makes local propagation give globally correct marginals.
    """
    if len(cliques) == 1:
        return []

    edges = sorted(
        ((len(cliques[i] & cliques[j]), i, j)
         for i, j in itertools.combinations(range(len(cliques)), 2)),
        key=lambda e: (-e[0], e[1], e[2]))

    parent = list(range(len(cliques)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = []
    for weight, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j))

    return tree


class _JunctionTree:
    """A calibrated tree: every clique's potential is its joint distribution
    with the evidence folded in."""

    def __init__(self, fitted, evidence=None):
        self.fitted = fitted
        self.evidence = {}

        discrete = [n for n in fitted.nodes
                    if isinstance(fitted[n], DiscreteNode)]
        if len(discrete) != len(fitted.nodes):
            raise NotImplementedError(
                "exact inference is implemented for discrete networks only")

        self.levels = {node: list(fitted[node].levels[0])
                       for node in fitted.nodes}
        cardinalities = {n: len(v) for n, v in self.levels.items()}

        adjacency = _moral_graph(fitted)
        _, clusters = _triangulate(adjacency, cardinalities)
        self.cliques = _maximal(clusters)
        self.edges = _junction_tree(self.cliques)

        # the conditional probability tables never change, so they are built
        # once; only the evidence and the calibration are redone per query.
        self._tables = {node: self._cpt(node) for node in fitted.nodes}

        self.set_evidence(evidence)

    def set_evidence(self, evidence):
        """Re-fold the evidence in and recalibrate, reusing the tree."""
        self.evidence = dict(evidence or {})
        self._check_evidence()
        self._assign()
        self._calibrate()
        return self

    def _check_evidence(self):
        for variable, value in self.evidence.items():
            if variable not in self.levels:
                raise ValueError(f"unknown node {variable!r}")
            if value not in self.levels[variable]:
                raise ValueError(
                    f"{value!r} is not a level of {variable!r}; "
                    f"levels are {self.levels[variable]}")

    # -- setting up ---------------------------------------------------------

    def _cpt(self, node):
        """A node's conditional probability table as a Factor.

        bn.fit stores it with the node on the first axis and the parents after
        it, which is the order the variables are listed in here.
        """
        entry = self.fitted[node]
        variables = [node] + list(entry.parents)
        levels = {v: list(self.levels[v]) for v in variables}
        return Factor(variables, levels, entry.probabilities)

    def _assign(self):
        """Multiply each table into one clique that covers it."""
        self.potentials = []
        for clique in self.cliques:
            variables = sorted(clique)
            shape = tuple(len(self.levels[v]) for v in variables)
            self.potentials.append(
                Factor(variables, self.levels, np.ones(shape)))

        for node in self.fitted.nodes:
            factor = self._tables[node]
            for i, clique in enumerate(self.cliques):
                if factor.scope <= clique:
                    self.potentials[i] = self.potentials[i] * factor
                    break
            else:  # pragma: no cover - triangulation guarantees a home
                raise AssertionError(
                    f"no clique covers {node!r} and its parents; the "
                    "triangulation is wrong")

        if self.evidence:
            self.potentials = [p.observe(self.evidence)
                               for p in self.potentials]

    # -- propagation --------------------------------------------------------

    def _neighbours(self):
        out = {i: [] for i in range(len(self.cliques))}
        for i, j in self.edges:
            out[i].append(j)
            out[j].append(i)
        return out

    def _calibrate(self):
        """Collect towards a root, then distribute back out.

        Two passes suffice on a tree: after them every clique's potential is
        the joint over its own variables, so any marginal can be read off
        whichever clique happens to contain the variable.
        """
        if not self.cliques:
            return

        neighbours = self._neighbours()
        root = 0

        order = []
        seen = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            order.append(node)
            for nxt in neighbours[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

        parents = {}
        for node in order:
            for nxt in neighbours[node]:
                if nxt not in parents and nxt != root:
                    parents.setdefault(nxt, node)

        messages = {}

        # collect: leaves first.
        for node in reversed(order):
            if node == root:
                continue
            separator = self.cliques[node] & self.cliques[parents[node]]
            message = self.potentials[node].marginalise(separator)
            messages[(node, parents[node])] = message
            self.potentials[parents[node]] = \
                self.potentials[parents[node]] * message

        # distribute: root first, dividing out what was sent up.
        for node in order:
            for child in neighbours[node]:
                if parents.get(child) != node:
                    continue
                separator = self.cliques[node] & self.cliques[child]
                down = self.potentials[node].marginalise(separator)
                up = messages[(child, node)]

                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(up.values > 0,
                                     down.values / np.where(up.values > 0,
                                                            up.values, 1.0),
                                     0.0)

                self.potentials[child] = self.potentials[child] * Factor(
                    down.variables, down.levels, ratio)

    # -- queries ------------------------------------------------------------

    def unnormalised(self, variables):
        """The joint of `variables` with the evidence, *not* normalised.

        Its total is the probability of the evidence, which prediction needs
        in order to refuse impossible rows rather than divide by zero.
        """
        variables = [str(v) for v in variables]
        unknown = set(variables) - set(self.fitted.nodes)
        if unknown:
            raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))

        wanted = set(variables)
        for i, clique in enumerate(self.cliques):
            if wanted <= clique:
                result = self.potentials[i].marginalise(variables)
                break
        else:
            result = self._brute_force(variables)

        return result.reorder(variables)

    def marginal(self, variables):
        """The joint distribution of `variables`, given the evidence."""
        variables = [str(v) for v in variables]
        unknown = set(variables) - set(self.fitted.nodes)
        if unknown:
            raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))

        return self.unnormalised(variables).normalise()

    def _brute_force(self, variables):
        """Fall back to multiplying every table together.

        Only reached when the queried variables do not all sit in one clique,
        which the caller avoids where it can: this is exponential in the
        network's size.
        """
        joint = None
        for node in self.fitted.nodes:
            factor = self._cpt(node)
            joint = factor if joint is None else joint * factor
        if self.evidence:
            joint = joint.observe(self.evidence)
        return joint.marginalise(variables)


def query(fitted, nodes, evidence=None):
    """Exact conditional probabilities, by junction tree.

    Unlike `cpquery`, this is not a Monte Carlo estimate: the answer is the
    network's, computed rather than sampled, so it does not depend on a seed
    or on how many particles you can afford.

    Parameters
    ----------
    fitted : FittedNetwork
        A fitted discrete network.
    nodes : str or sequence of str
        The variable, or variables, to compute the distribution of.
    evidence : dict, optional
        `{node: value}` to condition on.

    Returns
    -------
    Factor
        Use `.values` for the array, `.to_frame()` for a DataFrame.
    """
    if not isinstance(fitted, FittedNetwork):
        raise TypeError("query() needs a fitted network; call fit() first")

    if isinstance(nodes, str):
        nodes = [nodes]

    tree = _JunctionTree(fitted, evidence)
    return tree.marginal(nodes)


def exact_predict(fitted, node, data, predictors=None, prob=False):
    """Predict a node exactly, one observation at a time.

    For each row the observed predictors become evidence and the class with
    the largest posterior wins.  Two details of bnlearn's version are kept:
    a row whose evidence has probability zero under the network is left
    unpredicted rather than assigned an arbitrary class, and ties are broken
    at random, drawing from the same generator R does -- so tied rows consume
    the random stream and the result depends on the seed.
    """
    node = str(node)
    if node not in fitted:
        raise ValueError(f"unknown node {node!r}")

    if predictors is None:
        predictors = [str(c) for c in data.columns
                      if str(c) != node and str(c) in fitted.nodes]
    else:
        predictors = [str(p) for p in predictors]
        if node in predictors:
            raise ValueError(
                f"{node!r} is both a predictor and the node being predicted")

    from ._core import sample_indices

    tree = _JunctionTree(fitted)
    levels = tree.levels[node]

    predicted = [None] * len(data)
    posteriors = np.full((len(data), len(levels)), np.nan)

    # R treats an evidence probability at or below sqrt(eps) as impossible.
    threshold = np.sqrt(np.finfo(float).eps)

    frame = data[predictors].astype(str) if predictors else None

    for position in range(len(data)):
        evidence = ({name: frame.iloc[position][name] for name in predictors}
                    if predictors else {})
        tree.set_evidence(evidence)

        marginal = tree.unnormalised([node])
        total = marginal.values.sum()
        if total <= threshold:
            continue

        probabilities = marginal.values / total
        posteriors[position] = probabilities

        maxima = np.flatnonzero(probabilities == probabilities.max())
        if len(maxima) == 1:
            predicted[position] = levels[maxima[0]]
        else:
            draw = int(sample_indices(len(maxima), 1, replace=False)[0])
            predicted[position] = levels[maxima[draw - 1]]

    values = pd.Categorical(predicted, categories=levels)

    if not prob:
        return values

    return values, pd.DataFrame(posteriors, columns=levels, index=data.index)
