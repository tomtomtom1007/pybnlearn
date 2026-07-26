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
from .fit import fit
from .structure import BayesianNetwork, _check_complete

__all__ = ["CrossValidation", "bn_cv", "boot_strength"]


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
    counts = np.zeros((ncol, ncol), dtype=np.float64)

    for _ in range(replicates):
        # The two draws happen in this order in R, and each consumes the
        # generator, so swapping them would desynchronise every later
        # replicate as well as this one.
        rows = sample_indices(nrow, m, replace=True)
        columns = (sample_indices(ncol, ncol, replace=False) if shuffle
                   else np.arange(1, ncol + 1, dtype=np.int32))

        replicate = data.iloc[rows - 1, columns - 1]

        net = learn(replicate, **(algorithm_args or {}))
        if cpdag:
            net = graph.cpdag(net, wlbl=True)

        counts = arc_strength_counters(counts, net.arcs, nodes)

    counts /= replicates

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


def bn_cv(data, bn, loss=None, k=10, m=None, method="k-fold", folds=None,
          algorithm_args=None, fit_method=None, fit_args=None):
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
    if loss != "logl":
        raise NotImplementedError(
            f"loss {loss!r} needs predict(), which is not ported yet; only "
            "'logl' is available")

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

        value = network_loglikelihood(fitted, held_out)
        results.append({
            "test": test,
            "network": net,
            "fitted": fitted,
            "loss": -value["loglik"] / value["nobs"],
            "effective.size": value["nobs"],
        })

    weights = np.array([r["effective.size"] for r in results])
    values = np.array([r["loss"] for r in results])
    mean = float(np.average(values, weights=weights))

    return CrossValidation(results, loss, method, mean)
