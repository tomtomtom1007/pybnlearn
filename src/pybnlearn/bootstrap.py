"""Bootstrap arc strengths and cross-validation.

Ports of bnlearn's arc.strength.boot() in R/arc.strength.R and the
cross-validation in R/cross-validation.R.  Both resample the data, so both
reproduce R's results only when the resampling draws the same rows in the same
order -- which is what `sample_indices` is for.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import (arc_strength_coefficients, arc_strength_counters,
                    consistent_extension, cpdag_arcs, network_loglikelihood,
                    sample_indices)
from . import constraint, graph, hybrid, structure
from .fit import DiscreteNode, fit, predict
from .structure import BayesianNetwork, _check_complete

__all__ = ["CrossValidation", "bn_boot", "bn_cv", "boot_strength",
           "loss"]


_ALGORITHMS = {
    "hc": structure.hc,
    "tabu": structure.tabu,
    "gs": constraint.gs,
    "iamb": constraint.iamb,
    "inter.iamb": constraint.inter_iamb,
    "iamb.fdr": constraint.iamb_fdr,
    "mmpc": constraint.mmpc,
    "si.hiton.pc": constraint.si_hiton_pc,
    "pc.stable": constraint.pc_stable,
    "chow.liu": graph.chow_liu,
    "aracne": graph.aracne,
    "mmhc": hybrid.mmhc,
    "rsmax2": hybrid.rsmax2,
}


def boot_strength(data, algorithm="hc", replicates=200, m=None,
                  algorithm_args=None, cpdag=True, shuffle=True):
    """Estimate arc strengths by bootstrap, as bnlearn's boot.strength() does.

    Each replicate resamples the rows with replacement, optionally shuffles the
    columns, learns a network, and counts which arcs appear.  `strength` is the
    fraction of replicates containing the arc in either direction; `direction`
    is the fraction of *those* that orient it as given.

    Parameters
    ----------
    data : pandas.DataFrame
    algorithm : str
        Any of the ported structure learning algorithms.
    replicates : int
        How many bootstrap samples to draw.
    m : int, optional
        Rows per replicate; defaults to the size of the data.
    algorithm_args : dict, optional
        Passed through to the learning algorithm.
    cpdag : bool
        Reduce each learned network to its CPDAG first, so that arcs which are
        not identifiable from the data do not get counted as if they were.
    shuffle : bool
        Permute the columns of each replicate.  This is on by default in
        bnlearn because several algorithms are sensitive to variable order,
        and shuffling averages that out.

    Returns
    -------
    pandas.DataFrame with columns from, to, strength, direction.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; available are "
            + ", ".join(sorted(_ALGORITHMS)))

    _check_complete(data)

    replicates = int(replicates)
    if replicates < 1:
        raise ValueError("replicates must be a positive integer")

    nrow, ncol = data.shape
    m = nrow if m is None else int(m)
    if not 0 < m <= nrow:
        raise ValueError("m must be between 1 and the number of rows")

    nodes = [str(c) for c in data.columns]
    learn = _ALGORITHMS[algorithm]

    learned = []
    for _ in range(replicates):
        # The two draws happen in this order in R, and each consumes the
        # generator, so swapping them would desynchronise every later
        # replicate as well as this one.
        rows = sample_indices(nrow, m, replace=True)
        columns = (sample_indices(ncol, ncol, replace=False) if shuffle
                   else np.arange(1, ncol + 1, dtype=np.int32))

        replicate = data.iloc[rows - 1, columns - 1]
        learned.append(learn(replicate, **(algorithm_args or {})))

    result = _count_networks(learned, nodes, cpdag=cpdag)

    from .strength import inclusion_threshold

    result.attrs.update(nodes=nodes, method="bootstrap",
                        threshold=inclusion_threshold(result))
    return result


def _count_networks(networks, nodes, weights=None, cpdag=True):
    """arc.strength.custom(): how often each arc appears, and which way round.

    An undirected arc counts half in each direction, so that adding the
    strength and the direction back together gives one appearance rather
    than two -- which is why this cannot be done with a plain tally.

    The networks are passed in whole rather than as arc sets because
    cpdag(wlbl = True) consults what they were learned under: an arc a
    constraint-based run was told to leave alone must not be re-oriented
    here.  Reducing them to arcs first silently changed the answer for gs
    and iamb.
    """
    if weights is None:
        weights = np.ones(len(networks))
    weights = np.asarray(weights, dtype=float)

    counts = np.zeros((len(nodes), len(nodes)), dtype=np.float64)

    for network, weight in zip(networks, weights):
        if not isinstance(network, BayesianNetwork):
            network = BayesianNetwork(nodes, network)
        arcs = graph.cpdag(network, wlbl=True).arcs if cpdag else network.arcs
        counts = arc_strength_counters(counts, arcs, nodes, weight=weight)

    counts /= weights.sum()

    result = arc_strength_coefficients(counts, nodes)
    return pd.DataFrame({
        "from": result["from"],
        "to": result["to"],
        "strength": result["strength"],
        "direction": result["direction"],
    })


class CrossValidation:
    """The result of bn_cv(): the loss on each fold and their weighted mean."""

    def __init__(self, folds, loss, method, mean):
        self.folds = list(folds)
        self.loss = loss
        self.method = method
        self.mean = mean

    def __len__(self):
        return len(self.folds)

    def __iter__(self):
        return iter(self.folds)

    def __getitem__(self, i):
        return self.folds[i]

    def __repr__(self):
        return (f"CrossValidation({len(self.folds)} folds, loss={self.loss!r}, "
                f"method={self.method!r}, mean={self.mean:.6g})")


def _folds(n, k, m, method, folds):
    """The same partitions R draws, in the same order.

    Each call to sample() advances the generator, so the number and order of
    the draws is part of the result, not an implementation detail.
    """
    if method == "custom-folds":
        if not folds:
            raise ValueError("method='custom-folds' needs the folds")
        return [np.asarray(f, dtype=np.int64) for f in folds]

    if method == "k-fold":
        # split(sample(n), seq_len(k)): a permutation dealt round-robin into
        # k groups, so fold i takes every k-th element starting at i.
        permutation = sample_indices(n, n, replace=False)
        return [permutation[i::k].astype(np.int64) for i in range(k)]

    if method == "hold-out":
        return [sample_indices(n, m, replace=False).astype(np.int64)
                for _ in range(k)]

    raise ValueError("method must be 'k-fold', 'hold-out' or 'custom-folds'")


def _as_dag(net):
    """bn.cv needs a DAG; a learned CPDAG gets a consistent extension, as R
    does before fitting each fold."""
    if not any((b, a) in net.arcs for a, b in net.arcs):
        return net

    moralised = cpdag_arcs(net.arcs, net.nodes, moral=True, wlbl=False)
    return BayesianNetwork(net.nodes,
                           consistent_extension(moralised, net.nodes))


_PREDICTIVE_LOSSES = ("pred", "cor", "mse", "f1", "auroc")


def _classification_error(observed, predicted):
    """clerr.loss(): the proportion misclassified, over the observations both
    are defined for."""
    complete = ~(pd.isna(observed) | pd.isna(predicted))
    if not complete.any():
        return float("nan")
    return float((observed[complete] != predicted[complete]).mean())


def _f1(observed, predicted, levels):
    """f1.loss(): the F1 of the first level for a binary target, the mean of
    the per-level F1s otherwise."""
    observed = pd.Categorical(observed, categories=levels)
    predicted = pd.Categorical(predicted, categories=levels)
    matrix = pd.crosstab(observed, predicted, dropna=False).reindex(
        index=levels, columns=levels, fill_value=0).to_numpy(dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.diag(matrix) / matrix.sum(axis=0)
        recall = np.diag(matrix) / matrix.sum(axis=1)
        f1 = 2 * precision * recall / (precision + recall)

    f1 = np.nan_to_num(f1, nan=0.0)
    return float(f1[0] if len(levels) == 2 else f1.mean())


def _auc(labels, scores, levels):
    """One binary AUC, by the trapezoid rule over the ROC points R walks."""
    complete = ~(pd.isna(labels) | pd.isna(scores))
    labels = np.asarray(labels)[complete]
    scores = np.asarray(scores, dtype=float)[complete]

    negative, positive = levels[0], levels[1]
    cutoffs = np.append(np.unique(scores)[::-1], 0.0)

    npos = int((labels == positive).sum())
    nneg = int((labels == negative).sum())
    if npos == 0 or nneg == 0:
        return float("nan")

    predicted_positive = scores[None, :] > cutoffs[:, None]
    fp = (predicted_positive & (labels == negative)[None, :]).sum(axis=1)
    tp = (predicted_positive & (labels == positive)[None, :]).sum(axis=1)

    fpr, tpr = fp / nneg, tp / npos
    if len(fpr) < 2:
        return float("nan")

    return float(np.sum(0.5 * np.diff(fpr) * (tpr[1:] + tpr[:-1])))


def _predictive_loss(loss, observed, predicted, probabilities, levels):
    if loss == "pred":
        return _classification_error(observed, predicted)

    if loss == "f1":
        return _f1(observed, predicted, levels)

    if loss == "auroc":
        if probabilities is None or len(levels) != 2:
            raise NotImplementedError(
                "the auroc loss is only implemented for binary targets")
        return _auc(observed, probabilities[levels[1]], levels)

    numeric_observed = np.asarray(observed, dtype=float)
    numeric_predicted = np.asarray(predicted, dtype=float)
    complete = np.isfinite(numeric_observed) & np.isfinite(numeric_predicted)

    if loss == "mse":
        return float(np.mean((numeric_observed[complete]
                              - numeric_predicted[complete]) ** 2))

    # "cor": R's cor() returns NA when either variable is constant, which
    # happens whenever the target has no parents and so is predicted by a
    # single number.  Constancy is tested exactly rather than by comparing a
    # computed standard deviation against zero: summing 500 copies of the same
    # double and dividing gives a mean that is several ULP off, so the
    # "variance" comes out around 1e-29 rather than 0, and the guard would
    # never fire.  (bnlearn's own cgsd() has this problem -- it returns 6e-15
    # for a constant vector -- but R's cor() does its own, accurate check.)
    if (np.ptp(numeric_observed[complete]) == 0
            or np.ptp(numeric_predicted[complete]) == 0):
        return float("nan")
    return float(np.corrcoef(numeric_observed[complete],
                             numeric_predicted[complete])[0, 1])


def bn_cv(data, bn, loss=None, k=10, m=None, method="k-fold", folds=None,
          algorithm_args=None, fit_method=None, fit_args=None,
          target=None, predict_method="parents", predict_args=None):
    """Cross-validate a network or a learning algorithm, as bnlearn's bn.cv().

    Parameters
    ----------
    data : pandas.DataFrame
    bn : str or BayesianNetwork
        An algorithm name to cross-validate structure learning, or a fixed
        structure to cross-validate its parameters.
    loss : str, optional
        Only "logl" -- the negated log-likelihood per observation -- is
        implemented; the prediction-based losses need predict(), which is not
        ported yet.
    k : int
        Folds, or repetitions for hold-out.
    m : int, optional
        Test set size for hold-out; defaults to a tenth of the data.
    method : {"k-fold", "hold-out", "custom-folds"}
    folds : sequence of index arrays, optional
        Required for "custom-folds"; one-based, as R's are.

    Returns
    -------
    CrossValidation
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    _check_complete(data)

    loss = loss or "logl"
    if loss not in ("logl",) + _PREDICTIVE_LOSSES:
        raise ValueError(
            f"unknown loss {loss!r}; available are logl, "
            + ", ".join(_PREDICTIVE_LOSSES))
    if loss in _PREDICTIVE_LOSSES and target is None:
        raise ValueError(f"the {loss!r} loss needs a target node")

    n = len(data)
    k = int(k)
    m = int(m) if m is not None else -(-n // 10)

    if isinstance(bn, str) and bn not in _ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {bn!r}; available are "
            + ", ".join(sorted(_ALGORITHMS)))

    partitions = _folds(n, k, m, method, folds)
    results = []

    for test in partitions:
        # one-based indices, as R's are.
        mask = np.zeros(n, dtype=bool)
        mask[test - 1] = True
        train, held_out = data[~mask], data[mask]

        if isinstance(bn, str):
            net = _ALGORITHMS[bn](train, **(algorithm_args or {}))
        else:
            net = bn

        net = _as_dag(net)
        fitted = fit(net, train, method=fit_method, **(fit_args or {}))

        record = {"test": test, "network": net, "fitted": fitted}

        if loss == "logl":
            value = network_loglikelihood(fitted, held_out)
            record["loss"] = -value["loglik"] / value["nobs"]
            record["effective.size"] = value["nobs"]
        else:
            wants_probabilities = (loss == "auroc"
                                   and isinstance(fitted[target],
                                                  DiscreteNode))
            out = predict(fitted, target, held_out, method=predict_method,
                          prob=wants_probabilities, **(predict_args or {}))
            predicted, probabilities = (out if wants_probabilities
                                        else (out, None))

            record.update({
                "observed": held_out[target],
                "predicted": predicted,
                "probabilities": probabilities,
                # the loss is computed over the pooled folds, not per fold,
                # so it is filled in afterwards.
                "loss": None,
                "effective.size": len(held_out),
            })

        results.append(record)

    levels = (list(fitted[target].levels[0])
              if loss in ("pred", "f1", "auroc") else None)

    if loss == "logl":
        weights = np.array([r["effective.size"] for r in results])
        values = np.array([r["loss"] for r in results])
        mean = float(np.average(values, weights=weights))
    elif method == "hold-out":
        # hold-out scores each repetition on its own and averages the scores.
        for record in results:
            record["loss"] = _predictive_loss(
                loss, record["observed"], record["predicted"],
                record["probabilities"], levels)
        mean = float(np.mean([r["loss"] for r in results]))
    else:
        # k-fold pools every fold's predictions and scores them once, which is
        # not the same as averaging per-fold scores when the folds differ in
        # size or class balance.
        observed = pd.concat([pd.Series(r["observed"]).reset_index(drop=True)
                              for r in results], ignore_index=True)
        predicted = pd.concat([pd.Series(r["predicted"]).reset_index(drop=True)
                               for r in results], ignore_index=True)
        probabilities = (
            pd.concat([r["probabilities"].reset_index(drop=True)
                       for r in results], ignore_index=True)
            if results[0]["probabilities"] is not None else None)
        mean = _predictive_loss(loss, observed, predicted, probabilities,
                                levels)

    return CrossValidation(results, loss, method, mean)


def loss(result):
    """The loss a cross-validation run measured.

    A single run has one; a list of runs has one each, which is how the
    results of several `bn_cv` calls are compared.
    """
    if isinstance(result, CrossValidation):
        return result.mean
    if isinstance(result, (list, tuple)):
        return [loss(r) for r in result]

    raise TypeError("loss() needs the result of bn_cv()")


def bn_boot(data, statistic, replicates=200, m=None, algorithm="hc",
            algorithm_args=None, statistic_args=None):
    """Apply a statistic to networks learned from bootstrap samples.

    The general form of `boot_strength`: instead of counting arcs, it hands
    each learned network to a function of yours and collects the answers.
    That is how you get a bootstrap distribution for anything a network has
    -- the number of arcs, a particular node's parents, a score.

    Seed with `set_seed()`; the resamples come from R's generator, so the
    same seed draws the same rows in the same order R draws them.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; available are "
            + ", ".join(sorted(_ALGORITHMS)))
    if not callable(statistic):
        raise TypeError("statistic must be callable")

    replicates = int(replicates)
    if replicates < 1:
        raise ValueError("replicates must be a positive integer")

    nrow = len(data)
    m = nrow if m is None else int(m)
    if not 0 < m <= nrow:
        raise ValueError("m must be between 1 and the number of rows")

    learn = _ALGORITHMS[algorithm]

    out = []
    for _ in range(replicates):
        rows = sample_indices(nrow, m, replace=True)
        replicate = data.iloc[rows - 1]

        network = learn(replicate, **(algorithm_args or {}))
        out.append(statistic(network, **(statistic_args or {})))

    return out
