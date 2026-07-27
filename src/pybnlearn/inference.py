"""Simulation and approximate inference.

A port of bnlearn's R/simulation.R and R/cpq.R: draw observations from a
fitted network, and answer conditional probability queries by Monte Carlo --
either by logic sampling, which generates freely and discards what does not
match the evidence, or by likelihood weighting, which pins the evidence and
weights each observation by how likely it was.

Both reproduce R's numbers exactly for the same seed, because the sampling
runs through the same C code over R's own Mersenne-Twister.  That only holds
if the *same nodes are sampled in the same order*, which is why
`_upper_closure` below reproduces R's reduction of the network before
sampling rather than sampling the whole thing.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import random_sample, set_seed, weighted_sample
from ._validate import check_positive_integer
from .fit import FittedNetwork

__all__ = ["cpdist", "cpquery", "rbn", "set_seed"]


def _as_frame(columns):
    """Rebuild a DataFrame, keeping categoricals categorical."""
    return pd.DataFrame({name: values for name, values in columns.items()})


def _nodes_of(spec):
    """The nodes a query condition refers to."""
    if spec is None or spec is True:
        return set()
    if isinstance(spec, dict):
        return set(map(str, spec))
    if callable(spec):
        nodes = getattr(spec, "nodes", None)
        if nodes is None:
            raise ValueError(
                "a callable event or evidence must carry a `nodes` attribute "
                "listing the variables it looks at, so that the network can "
                "be reduced the way bnlearn reduces it")
        return set(map(str, nodes))
    raise TypeError("event and evidence must be a dict, a callable, or True")


def _upper_closure(fitted, targets):
    """reduce.fitted(): the targets and everything they descend from.

    R gets this from topological.ordering(..., reverse = TRUE) and then keeps
    `names(fitted) %in% upper.closure`, so only the *set* matters -- the
    original node order is preserved by the subsetting.
    """
    closure = set()
    stack = list(targets)

    while stack:
        node = stack.pop()
        if node in closure:
            continue
        closure.add(node)
        stack.extend(fitted[node].parents)

    return closure


def _reduce(fitted, event, evidence):
    """Restrict the network to the nodes the query can possibly depend on.

    Sampling fewer nodes draws fewer random numbers, so skipping this would
    put the generator out of step with R even though the network is the same.
    """
    targets = _nodes_of(event) | _nodes_of(evidence)
    if not targets:
        return fitted

    keep = _upper_closure(fitted, targets)
    if keep == set(fitted.nodes):
        return fitted

    # keep the original order, as R's logical subsetting does.
    return FittedNetwork(
        {name: fitted[name] for name in fitted.nodes if name in keep},
        fitted.method)


def _mask(spec, frame):
    """Turn an event or evidence specification into a boolean mask."""
    if spec is None or spec is True:
        return np.ones(len(frame), dtype=bool)

    if callable(spec):
        out = np.asarray(spec(frame))
        if out.dtype != bool or len(out) != len(frame):
            raise ValueError(
                "an event or evidence callable must return a boolean array "
                f"of length {len(frame)}")
        return out

    mask = np.ones(len(frame), dtype=bool)
    for node, wanted in spec.items():
        column = frame[str(node)]
        if isinstance(wanted, (list, tuple, set, frozenset)):
            mask &= column.isin(list(wanted)).to_numpy()
        elif callable(wanted):
            mask &= np.asarray(wanted(column), dtype=bool)
        else:
            mask &= (column == wanted).to_numpy()

    return mask


def _batches(n, batch):
    """R generates the particles in batches; the split has to match, because
    each call to rbn() draws from the generator in sequence."""
    full, rest = divmod(n, batch)
    return [batch] * full + ([rest] if rest else [])


def _default_particles(fitted):
    """check.particles(): how many observations bnlearn uses by default."""
    nparams = 0
    for node in fitted:
        if hasattr(node, "probabilities"):
            shape = node.probabilities.shape
            nparams += int(np.prod(shape)) - int(np.prod(shape[1:]))
        else:
            nparams += len(node.coefficients) + 1

    if fitted.method == "mle-g":
        return 500 * nparams
    return 5000 * max(1, round(np.log10(max(nparams, 1))))


def rbn(fitted, n=1):
    """Draw observations from a fitted network, as bnlearn's rbn() does.

    Seed the generator with `set_seed()` to reproduce R's `set.seed()`.
    """
    if not isinstance(fitted, FittedNetwork):
        raise TypeError("rbn() needs a fitted network; call fit() first")
    n = check_positive_integer(n, "the number of observations to be generated")

    return _as_frame(random_sample(fitted, n))


def cpdist(fitted, nodes, evidence=None, method="lw", n=None, batch=None):
    """Sample from a conditional distribution, as bnlearn's cpdist() does.

    Parameters
    ----------
    fitted : FittedNetwork
    nodes : sequence of str
        The variables to return.
    evidence : dict or callable, optional
        `{node: value}` pins a node; a sequence of values accepts any of them.
        Likelihood weighting requires a dict of single values, which is what it
        pins.  Logic sampling also accepts a callable taking the sampled frame
        and returning a boolean array, which must carry a `nodes` attribute.
    method : {"lw", "ls"}
    n : int, optional
        Number of particles; defaults to bnlearn's rule.
    batch : int, optional
        Particles per batch for logic sampling; defaults to min(n, 10000).

    Returns
    -------
    For "ls", a DataFrame of the accepted observations.  For "lw", a
    (DataFrame, weights) pair, since the observations are only meaningful
    together with their weights.
    """
    if not isinstance(fitted, FittedNetwork):
        raise TypeError("cpdist() needs a fitted network; call fit() first")
    if method not in ("lw", "ls"):
        raise ValueError("method must be 'lw' or 'ls'")

    nodes = [str(v) for v in nodes]
    unknown = set(nodes) - set(fitted.nodes)
    if unknown:
        raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))

    if method == "lw" and evidence is not None \
            and not isinstance(evidence, dict):
        raise ValueError(
            "likelihood weighting pins the evidence, so it needs a dict of "
            "single values; use method='ls' for anything else")

    reduced = _reduce(fitted, {node: True for node in nodes}, evidence)
    n = (_default_particles(reduced) if n is None else
         check_positive_integer(n, "the number of observations to be sampled"))

    if method == "lw":
        columns, weights = weighted_sample(reduced, nodes, n, fix=evidence)
        return _as_frame(columns), weights

    batch = int(batch) if batch is not None else min(n, 10 ** 4)
    frames = []
    for size in _batches(n, batch):
        generated = _as_frame(random_sample(reduced, size))
        frames.append(generated[_mask(evidence, generated)])

    kept = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return kept[nodes] if len(kept) else kept


def cpquery(fitted, event, evidence=None, method="ls", n=None, batch=None):
    """Estimate P(event | evidence), as bnlearn's cpquery() does.

    `event` and `evidence` take the same forms as cpdist()'s evidence.
    """
    if not isinstance(fitted, FittedNetwork):
        raise TypeError("cpquery() needs a fitted network; call fit() first")
    if method not in ("lw", "ls"):
        raise ValueError("method must be 'lw' or 'ls'")

    if method == "lw" and evidence is not None \
            and not isinstance(evidence, dict):
        raise ValueError(
            "likelihood weighting pins the evidence, so it needs a dict of "
            "single values; use method='ls' for anything else")

    reduced = _reduce(fitted, event, evidence)
    n = (_default_particles(reduced) if n is None else
         check_positive_integer(n, "the number of observations to be sampled"))
    batch = int(batch) if batch is not None else min(n, 10 ** 4)

    if method == "ls":
        matched = accepted = 0

        for size in _batches(n, batch):
            generated = _as_frame(random_sample(reduced, size))
            keep = _mask(evidence, generated)
            hit = keep & _mask(event, generated)
            accepted += int(keep.sum())
            matched += int(hit.sum())

        return matched / accepted if accepted else 0.0

    numerator = denominator = 0.0

    for size in _batches(n, batch):
        columns, weights = weighted_sample(
            reduced, list(reduced.nodes), size, fix=evidence)
        generated = _as_frame(columns)
        hit = _mask(event, generated)
        denominator += float(np.sum(weights))
        numerator += float(np.sum(weights[hit]))

    return numerator / denominator if denominator else 0.0
