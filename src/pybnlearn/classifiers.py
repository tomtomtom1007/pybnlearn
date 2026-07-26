"""Bayesian network classifiers.

Ports of bnlearn's naive.bayes() and tree.bayes(), in
R/learning-algorithms.R.  Both build a structure directly rather than
searching for one: naive Bayes makes every feature a child of the class, and
TAN adds a tree over the features on top of that.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import pandas as pd

from ._core import chow_liu_arcs, orient_tree
from .graph import _check_mi
from .structure import BayesianNetwork, _check_complete, _data_type, \
    build_blacklist

__all__ = ["classify", "naive_bayes", "tree_bayes"]


def _check_classifier(data, training, explanatory):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if _data_type(data) != "discrete":
        raise ValueError(
            "the classifiers only work with discrete data; every column must "
            "be categorical")

    _check_complete(data)

    variables = [str(c) for c in data.columns]
    training = str(training)
    if training not in variables:
        raise ValueError(f"unknown training node {training!r}")

    if explanatory is None:
        explanatory = [v for v in variables if v != training]
    else:
        explanatory = [str(v) for v in explanatory]
        unknown = set(explanatory) - set(variables)
        if unknown:
            raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))

    if not explanatory:
        raise ValueError("at least one explanatory variable is required")
    if training in explanatory:
        raise ValueError(
            f"node {training!r} is both the training and an explanatory "
            "variable")

    return training, explanatory


def naive_bayes(data, training, explanatory=None):
    """The naive Bayes classifier, as bnlearn's naive.bayes().

    Every explanatory variable becomes a child of the class, so the features
    are assumed independent given it.  Nothing is learned from the data here
    beyond the variable names -- the structure is fixed by definition, and the
    data only matter once fit() estimates the parameters.
    """
    training, explanatory = _check_classifier(data, training, explanatory)

    return BayesianNetwork(
        [training] + explanatory,
        [(training, feature) for feature in explanatory],
        learning={
            "algo": "naive.bayes",
            "test": "none",
            "args": {"training": training},
            "training": training,
            "explanatory": explanatory,
        },
    )


def tree_bayes(data, training, explanatory=None, whitelist=None,
               blacklist=None, mi=None, root=None):
    """The tree-augmented naive Bayes classifier, as bnlearn's tree.bayes().

    On top of naive Bayes, the features are joined by a maximum-weight
    spanning tree over their mutual information *given the class*, then
    oriented away from a root.
    """
    training, explanatory = _check_classifier(data, training, explanatory)

    nodes = [training] + explanatory
    estimator = _check_mi(mi, data)

    whitelist = [(str(a), str(b)) for a, b in (whitelist or ())]
    blacklist = [(str(a), str(b)) for a, b in (blacklist or ())]
    for a, b in whitelist + blacklist:
        if a not in nodes or b not in nodes:
            raise ValueError(f"unknown node in arc ({a}, {b})")
        if training in (a, b):
            raise ValueError(
                "arcs to and from the training node cannot be whitelisted or "
                "blacklisted: they are fixed by the model")

    blacklist = build_blacklist(blacklist, whitelist)
    # the tree is undirected, so only edges banned both ways constrain it.
    both = [(a, b) for a, b in blacklist if (b, a) in blacklist]

    root = explanatory[0] if root is None else str(root)
    if root not in explanatory:
        raise ValueError(f"the root must be an explanatory variable, not "
                         f"{root!r}")

    tree = chow_liu_arcs(
        data[explanatory], explanatory, estimator,
        whitelist or None, both or None,
        conditional=data[training])
    tree = orient_tree(tree, explanatory, root)

    arcs = [(training, feature) for feature in explanatory] + list(tree)

    return BayesianNetwork(
        nodes, arcs,
        learning={
            "algo": "tree.bayes",
            "test": estimator,
            "args": {"estimator": estimator, "root": root,
                     "training": training},
            "training": training,
            "explanatory": explanatory,
            "whitelist": whitelist,
            "blacklist": blacklist,
        },
    )


def classify(fitted, data, training=None, prior=None, prob=False):
    """Predict the class with a fitted naive Bayes or TAN classifier.

    Exact, not approximate: the class is the only unobserved variable and the
    features are all children of it, so the posterior is a product over the
    features.  This is what bnlearn's predict() does for a classifier, and
    what its "pred-exact" cross-validation loss uses.

    Parameters
    ----------
    fitted : FittedNetwork
        A fitted naive Bayes or TAN structure.
    data : pandas.DataFrame
    training : str, optional
        The class node; taken from the structure when it was built by
        naive_bayes() or tree_bayes().
    prior : sequence of float, optional
        The prior over the classes; defaults to the fitted marginal, as
        bnlearn does.
    prob : bool
        Also return the posterior over the classes.
    """
    from ._core import predict_classifier

    training = training or fitted.learning.get("training")
    if training is None:
        raise ValueError(
            "classify() needs to know which node is the class; either fit a "
            "network built by naive_bayes() or tree_bayes(), or pass "
            "training=...")
    if training not in fitted:
        raise ValueError(f"unknown node {training!r}")

    # The C routine assumes the classifier's shape and indexes the parameters
    # accordingly: it reads past the end of them on an arbitrary network
    # rather than complaining.  R never reaches that because its predict()
    # only dispatches here for objects classed bn.naive or bn.tan; this check
    # is what stands in for that.
    if fitted[training].parents:
        raise ValueError(
            f"{training!r} is not a classifier's class node: it has parents "
            f"({', '.join(fitted[training].parents)})")
    for node in fitted.nodes:
        if node != training and training not in fitted[node].parents:
            raise ValueError(
                f"this is not a naive Bayes or TAN structure: {node!r} is not "
                f"a child of {training!r}. Build one with naive_bayes() or "
                "tree_bayes(), or use predict() for a general network.")

    if prior is None:
        prior = fitted[training].probabilities.reshape(-1)

    values, probabilities = predict_classifier(
        fitted, training, data, prior, prob=prob)

    if not prob:
        return values

    table, levels = probabilities
    return values, pd.DataFrame(table, columns=levels, index=data.index)
