"""Score-based structure learning.

This mirrors bnlearn's R/hill-climbing.R and the parts of
R/learning-algorithms.R that set the search up.  Almost all the work happens
in the C core -- filling the score cache and picking the best arc operation --
so what is here is the loop around it, plus the argument checking that decides
which score is used and with what hyperparameters.

The defaults matter as much as the arithmetic: bnlearn's BIC carries a penalty
of log(n)/2 rather than the more common log(n), and getting that wrong would
produce networks that look plausible and disagree with R.  They are taken from
R/sanitization-scores.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._core import (Search, alpha_star_value, reset_test_counter,
                    test_counter, topological_order)
from ._core import acyclic as _acyclic

__all__ = ["BF", "BayesianNetwork", "alpha_star", "blacklist", "hc",
           "ntests", "score", "tabu", "whitelist"]


# Scores that are score-equivalent, and so let the search reuse the cached
# delta of an arc when its reverse is considered (is.score.equivalent).
_EQUIVALENT = {
    "loglik", "loglik-g", "pred-loglik", "pred-loglik-g",
    "aic", "aic-g", "bic", "bic-g", "ebic", "ebic-g",
    "qnml",
}

_DISCRETE_SCORES = {
    "loglik", "aic", "bic", "ebic", "bde", "bds", "bdj", "k2", "fnml", "qnml",
}
_CONTINUOUS_SCORES = {"loglik-g", "aic-g", "bic-g", "ebic-g", "bge"}
_MIXED_SCORES = {"loglik-cg", "aic-cg", "bic-cg", "ebic-cg"}

# Which extra arguments each score accepts (score.extra.args in R/globals.R),
# restricted to the scores implemented here.
_SCORE_ARGS = {
    "loglik": (), "loglik-g": (),
    "aic": ("k",), "aic-g": ("k",),
    "bic": ("k",), "bic-g": ("k",),
    "ebic": ("k", "gamma"), "ebic-g": ("k", "gamma"),
    "bde": ("prior", "beta", "iss"),
    "bds": ("prior", "beta", "iss"),
    "bdj": ("prior", "beta"),
    "k2": (),
    "fnml": (), "qnml": (),
    "bge": ("prior", "beta", "nu", "iss.mu", "iss.w"),
    "loglik-cg": (), "aic-cg": ("k",), "bic-cg": ("k",),
    "ebic-cg": ("k", "gamma"),
}


class BayesianNetwork:
    """A learned network: its nodes, its arcs, and how it was learned."""

    def __init__(self, nodes, arcs, learning=None):
        self.nodes = list(nodes)
        self.arcs = [(str(a), str(b)) for a, b in arcs]
        self.learning = learning or {}

    def parents(self, node):
        """The nodes with a *directed* arc into this one.

        An arc listed in both directions is undirected, and an undirected
        arc has no parent at the far end of it -- so it is excluded here,
        the way cache.structure() excludes it in R.  The order is the node
        order, again as R's is: it decides how parents are listed in a model
        string and how a conditional probability table's axes come out.
        """
        present = set(self.arcs)
        rank = {n: i for i, n in enumerate(self.nodes)}
        return sorted((a for a, b in self.arcs
                       if b == node and (b, a) not in present),
                      key=rank.__getitem__)

    def children(self, node):
        present = set(self.arcs)
        rank = {n: i for i, n in enumerate(self.nodes)}
        return sorted((b for a, b in self.arcs
                       if a == node and (b, a) not in present),
                      key=rank.__getitem__)

    @property
    def narcs(self):
        """How many arcs, counting an undirected one once rather than as the
        two rows it is stored as."""
        return len({frozenset(arc) if (arc[1], arc[0]) in set(self.arcs)
                    else arc for arc in self.arcs})

    def amat(self):
        """The adjacency matrix, rows and columns ordered as `nodes`."""
        index = {n: i for i, n in enumerate(self.nodes)}
        out = np.zeros((len(self.nodes), len(self.nodes)), dtype=np.int32)
        for a, b in self.arcs:
            out[index[a], index[b]] = 1
        return out

    def topological_order(self):
        """The nodes in the order bnlearn's topological.ordering() gives."""
        return topological_order(self.nodes, self.arcs)

    def modelstring(self):
        """The model string, in bnlearn's format.

        The nodes come out in topological order and the parents in the order
        the C code lists them, so the string is character-for-character what
        R's modelstring() produces for the same network.
        """
        # cache.structure() lists a node's parents in the order the nodes
        # themselves are in, not the order the arcs were added, so the parents
        # are sorted the same way here.
        rank = {node: i for i, node in enumerate(self.nodes)}
        parents = {node: [] for node in self.nodes}
        for a, b in self.arcs:
            parents[b].append(a)
        for node in parents:
            parents[node].sort(key=rank.__getitem__)

        parts = []
        for node in self.topological_order():
            ps = parents[node]
            parts.append(f"[{node}" + ("|" + ":".join(ps) if ps else "") + "]")
        return "".join(parts)

    def __repr__(self):
        return (f"BayesianNetwork(nodes={len(self.nodes)}, "
                f"arcs={len(self.arcs)}, score={self.learning.get('test')!r})")


# ---------------------------------------------------------------------------
# argument checking, following R/sanitization-scores.R
# ---------------------------------------------------------------------------

def is_discrete_column(series):
    """Whether a column reaches the C core as a factor.

    This has to agree with what the conversion layer in _core.pyx actually
    does, which is to treat anything that is not numeric -- categorical,
    string, object, boolean -- as a factor.  Deciding it differently here
    would send a discrete dataset to a continuous score.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    return not pd.api.types.is_numeric_dtype(series)


def _data_type(data):
    discrete = [is_discrete_column(data[c]) for c in data.columns]
    if all(discrete):
        return "discrete"
    if not any(discrete):
        return "continuous"
    return "mixed-cg"


def _check_complete(data):
    """Reject data with missing or infinite values.

    None of the scores wired up so far can handle incomplete data, and the C
    code does not check: a missing value reaches the discrete scores as R's
    NA_INTEGER, which is INT_MIN, and is used unguarded as a contingency-table
    index.  That is an out-of-bounds write, so this has to be caught here.

    An infinity is quieter and worse.  It is not missing, so nothing rejects
    it, and it reaches the Gaussian scores as a number: the mean and variance
    come out non-finite, every score comes out NaN, no comparison between two
    NaNs is ever true, and the search accepts no move at all.  The answer is
    an empty network that looks like a finding.  R refuses the data instead,
    and so does this.
    """
    incomplete = [str(c) for c in data.columns if data[c].isna().any()]
    if incomplete:
        raise ValueError(
            "the data contain missing values in "
            + ", ".join(incomplete[:5])
            + (", ..." if len(incomplete) > 5 else "")
            + "; none of the available scores support incomplete data. Note "
              "that pandas.read_csv() treats strings such as 'NA', 'None' and "
              "'N/A' as missing by default -- pass keep_default_na=False if "
              "they are meant to be category labels.")

    unbounded = [str(c) for c in data.columns
                 if pd.api.types.is_numeric_dtype(data[c])
                 and not np.isfinite(data[c].to_numpy()).all()]
    if unbounded:
        raise ValueError(
            "the data contain non-finite values in "
            + ", ".join(unbounded[:5])
            + (", ..." if len(unbounded) > 5 else ""))


def _check_maxp(maxp):
    """check.maxp(): the parent limit has to leave room for a parent.

    Zero or a negative number is not a degenerate case that happens to give
    the empty network -- it is a request that cannot be met, and returning
    the empty network for it would look like a result.
    """
    if maxp is None or maxp == float("inf"):
        return float("inf")
    if isinstance(maxp, float) and not maxp.is_integer():
        raise ValueError("maxp must be a positive integer number")
    maxp = int(maxp)
    if maxp < 1:
        raise ValueError("maxp must be a positive integer number")
    return maxp


def _check_start(start, nodes):
    """check.bn.vs.data() for a starting network: it has to describe the same
    variables the data do.

    A network over a subset silently learned a network over that subset and
    called it an answer, which is a wrong result rather than an error.
    """
    if start is None:
        return []
    if not isinstance(start, BayesianNetwork):
        raise TypeError("'start' must be a BayesianNetwork")
    if list(start.nodes) != list(nodes):
        if len(start.nodes) != len(nodes):
            raise ValueError(
                "the network and the data have different numbers of "
                f"variables: {len(start.nodes)} and {len(nodes)}")
        raise ValueError("the network and the data have different variables")
    return list(start.arcs)


def check_whitelist(whitelist, nodes, score_based):
    """check.whitelist(): what a whitelist is not allowed to ask for.

    Two things, and they are not the same thing.

    Both directions of one arc is how an *undirected* constraint is written,
    which the constraint-based algorithms understand and the score-based ones
    do not -- hill climbing orients every arc, so it has no way to honour a
    request not to.  R refuses it there and accepts it here; dropping one of
    the two directions instead, which is what this used to do, silently
    answers a question the user did not ask.

    A whitelist that cannot be part of any acyclic graph is impossible for
    every algorithm.  Left unchecked the score-based searches reported it as
    a cycle in the *starting* network, which is misleading when the caller
    passed no starting network at all.
    """
    whitelist = [(str(a), str(b)) for a, b in (whitelist or ())]
    if not whitelist:
        return whitelist

    unknown = {n for arc in whitelist for n in arc} - set(nodes)
    if unknown:
        raise ValueError("unknown node(s) in the whitelist: "
                         + ", ".join(sorted(unknown)))

    if score_based:
        both = sorted({tuple(sorted(a)) for a in whitelist
                       if (a[1], a[0]) in set(whitelist)})
        if both:
            raise ValueError(
                "score-based algorithms do not support whitelisting both "
                "directions of an arc: "
                + ", ".join(f"({a}, {b})" for a, b in both))

    # An undirected pair constrains adjacency but not direction, so it cannot
    # close a cycle by itself and is left out of the check.
    present = set(whitelist)
    directed = [a for a in whitelist if (a[1], a[0]) not in present]
    if not _acyclic(list(nodes), directed, directed=True):
        raise ValueError("this whitelist does not allow an acyclic graph")

    return whitelist


def build_blacklist(blacklist, whitelist):
    """build.blacklist(): reconcile the two lists.

    Whitelisting x -> y implicitly blacklists y -> x, unless that direction is
    whitelisted too (which is how an undirected constraint is expressed).  This
    is what actually pins a whitelisted arc's direction down: without it the
    orientation phase has no reason to prefer one direction over the other, and
    the arc comes back undirected.
    """
    whitelist = list(whitelist or ())
    out = list(blacklist or ())

    for a, b in whitelist:
        if (b, a) not in whitelist and (b, a) not in out:
            out.append((b, a))

    # an arc cannot be both required and forbidden; the whitelist wins.
    return [arc for arc in out if arc not in whitelist]


def _check_score(score, data):
    kind = _data_type(data)

    if score is None:
        # check.score()'s defaults for complete data.
        return {"discrete": "bic", "continuous": "bic-g",
                "mixed-cg": "bic-cg"}[kind]

    if score not in _SCORE_ARGS:
        raise ValueError(
            f"score {score!r} is not implemented yet; available scores are "
            + ", ".join(sorted(_SCORE_ARGS)))
    if kind != "discrete" and score in _DISCRETE_SCORES:
        raise ValueError(f"score {score!r} may only be used with discrete data")
    if kind != "continuous" and score in _CONTINUOUS_SCORES:
        raise ValueError(
            f"score {score!r} may only be used with continuous data")
    if kind != "mixed-cg" and score in _MIXED_SCORES:
        raise ValueError(
            f"score {score!r} may only be used with a mixture of discrete and "
            "continuous data")

    return score


def _check_score_args(score, data, extra_args):
    """Fill in the defaults R would, and reject arguments the score ignores."""
    extra_args = dict(extra_args or {})
    accepted = _SCORE_ARGS.get(score, ())

    unused = set(extra_args) - set(accepted)
    if unused:
        raise ValueError(
            f"score {score!r} does not take the argument(s) "
            + ", ".join(sorted(unused)))

    out = {}

    if "k" in accepted:
        # check.penalty(): AIC penalises by 1, BIC by log(n)/2.
        if "k" in extra_args:
            out["k"] = float(extra_args["k"])
        elif score.startswith("aic"):
            out["k"] = 1.0
        else:
            out["k"] = math.log(len(data)) / 2

    if "gamma" in accepted:
        out["gamma"] = float(extra_args.get("gamma", 0.5))

    if "prior" in accepted:
        out["prior"] = extra_args.get("prior", "uniform")
        if out["prior"] not in _GRAPH_PRIORS:
            raise ValueError(
                f"the {out['prior']!r} graph prior is not one of "
                + ", ".join(sorted(_GRAPH_PRIORS)))

        beta = _check_graph_prior(out["prior"], extra_args.get("beta"), data)
        if beta is not None:
            out["beta"] = beta

    if "iss" in accepted:
        # check.iss(): the de facto standard imaginary sample size is 1.
        out["iss"] = float(extra_args.get("iss", 1))

    if "iss.mu" in accepted:
        out["iss.mu"] = float(extra_args.get("iss.mu", 1))
    if "iss.w" in accepted:
        out["iss.w"] = float(extra_args.get("iss.w", data.shape[1] + 2))
    if "nu" in accepted:
        # check.nu(): the prior mean vector defaults to the column means, and
        # keeps the variable names -- the C side looks nu up by name.
        nu = extra_args.get("nu")
        if nu is None:
            out["nu"] = data.mean()
        elif isinstance(nu, pd.Series):
            out["nu"] = nu.astype(float)
        else:
            out["nu"] = pd.Series(np.asarray(nu, dtype=float),
                                  index=[str(c) for c in data.columns])

    return out


_GRAPH_PRIORS = ("uniform", "vsp", "cs", "marginal")

# the priors that make a Bayesian score depend on more than one node's
# family at a time.
_NON_DECOMPOSABLE_PRIORS = ("cs", "marginal")

_BAYESIAN_SCORES = ("bde", "bds", "bdj", "bge")


def _check_graph_prior(prior, beta, data):
    """check.graph.hyperparameters(): the prior's own parameter.

    A graph prior says how likely a *structure* is before the data are seen,
    which is a different thing from the parameter priors the Bayesian scores
    already have.  Three of the four need a hyperparameter, and each needs a
    different shape of one.
    """
    from ._core import MarginalPrior, complete_castelo_prior

    nodes = [str(c) for c in data.columns]

    if prior == "uniform":
        if beta is not None:
            raise ValueError("the uniform graph prior takes no beta")
        return None

    if prior == "vsp":
        # each arc is included independently with probability beta.
        beta = 1 / (len(nodes) - 1) if beta is None else float(beta)
        if not 0 <= beta < 1:
            raise ValueError("beta must be a probability smaller than one")
        return beta

    if prior == "marginal":
        beta = 0.5 if beta is None else float(beta)
        if not 0 <= beta < 1:
            raise ValueError("beta must be a probability smaller than one")
        return MarginalPrior(beta, nodes)

    # Castelo & Siebes: a probability for each arc you have an opinion
    # about, completed to cover both directions of every pair.
    if beta is None:
        return complete_castelo_prior([], [], [], nodes)

    if isinstance(beta, pd.DataFrame):
        if list(beta.columns) != ["from", "to", "prob"]:
            raise ValueError(
                "beta must have the columns from, to and prob, in that order")
        frm = [str(v) for v in beta["from"]]
        to = [str(v) for v in beta["to"]]
        probability = [float(v) for v in beta["prob"]]
    else:
        try:
            frm = [str(a) for a, _, _ in beta]
            to = [str(b) for _, b, _ in beta]
            probability = [float(p) for _, _, p in beta]
        except (TypeError, ValueError):
            raise ValueError(
                "beta must be a DataFrame with columns from, to and prob, or "
                "a sequence of (from, to, probability) triples") from None

    unknown = {n for n in frm + to} - set(nodes)
    if unknown:
        raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))
    if any(not 0 <= p <= 1 for p in probability):
        raise ValueError("the arc priors must be probabilities")

    return complete_castelo_prior(frm, to, probability, nodes)


# The graph priors that leave BDe and BGe score equivalent.  vsp and
# marginal treat every arc alike, so reversing one does not change the
# prior; the Castelo & Siebes prior has an opinion about direction, and so
# breaks the equivalence.
_EQUIVALENT_PRIORS = ("uniform", "marginal", "vsp")


def _is_score_equivalent(score, extra_args):
    if score in _EQUIVALENT:
        return True
    if (score in ("bde", "bge")
            and extra_args.get("prior") in _EQUIVALENT_PRIORS):
        return True
    return False


def _is_score_decomposable(score, extra_args):
    """is.score.decomposable(): whether a node's score depends only on its
    own family.

    The Castelo & Siebes and marginal priors put a probability on every
    *pair* of nodes, adjacent or not, so changing one arc changes the prior
    contribution of nodes the arc does not touch.  The search then has to
    rescore everything rather than only what moved.
    """
    if (score in _BAYESIAN_SCORES
            and extra_args.get("prior") in _NON_DECOMPOSABLE_PRIORS):
        return False
    return True


# ---------------------------------------------------------------------------
# arc operations, following R/arc.operations.R
# ---------------------------------------------------------------------------

def _set_arc(arcs, frm, to):
    """set.arc.direction(): add from -> to, replacing the reverse if present."""
    if (to, frm) in arcs and (frm, to) in arcs:      # undirected
        return [a for a in arcs if a != (to, frm)]
    if (to, frm) in arcs:
        return [a for a in arcs if a != (to, frm)] + [(frm, to)]
    if (frm, to) in arcs:
        return list(arcs)
    return list(arcs) + [(frm, to)]


def _drop_arc(arcs, frm, to):
    """drop.arc.backend(): remove any arc between the two nodes."""
    return [a for a in arcs if a != (frm, to) and a != (to, frm)]


def _reverse_arc(arcs, frm, to):
    """reverse.arc.backend(): flip whichever direction is present."""
    if (to, frm) in arcs and (frm, to) in arcs:
        raise ValueError("an undirected arc cannot be reversed")
    if (to, frm) in arcs:
        return [a for a in arcs if a != (to, frm)] + [(frm, to)]
    if (frm, to) in arcs:
        return [a for a in arcs if a != (frm, to)] + [(to, frm)]
    raise ValueError(f"no arc to be reversed between {frm} and {to}")


def _apply(arcs, op, frm, to):
    return {"set": _set_arc, "drop": _drop_arc,
            "reverse": _reverse_arc}[op](arcs, frm, to)


# ---------------------------------------------------------------------------
# hill climbing
# ---------------------------------------------------------------------------

def score(network, data, type=None, by_node=False, **extra_args):
    """The score of a network, as bnlearn's score() computes it.

    Parameters
    ----------
    network : BayesianNetwork
    data : pandas.DataFrame
    type : str, optional
        Defaults to the score the network was learned with, then to the
        default for the data type.
    by_node : bool
        Return each node's contribution instead of the total.
    """
    _check_complete(data)

    nodes = [str(c) for c in data.columns]
    if set(nodes) != set(network.nodes):
        raise ValueError("the network and the data have different variables")

    # R reuses the score and its hyperparameters from the bn object when they
    # are not given explicitly (check.score / check.score.args).
    if type is None:
        type = network.learning.get("test")
    type = _check_score(type, data)

    if not extra_args and network.learning.get("test") == type:
        extra = dict(network.learning.get("args") or {})
    else:
        extra = _check_score_args(type, data, extra_args)

    zeros = np.zeros((len(nodes), len(nodes)), dtype=np.int32)
    with Search(data, nodes, type, extra, zeros, zeros) as search:
        values = search.node_scores(network.arcs, nodes)

    if by_node:
        return dict(zip(nodes, (float(v) for v in values)))
    return float(values.sum())


def hc(data, start=None, whitelist=None, blacklist=None, score=None,
       max_iter=float("inf"), maxp=float("inf"), optimized=True,
       restart=0, perturb=1, **extra_args):
    """Learn a network structure by hill climbing, as bnlearn's hc() does.

    Parameters
    ----------
    data : pandas.DataFrame
        Categorical columns are treated as discrete, numeric ones as Gaussian.
    start : BayesianNetwork, optional
        The network to start from; an empty graph by default.
    whitelist, blacklist : sequence of (from, to), optional
        Arcs that must, or must not, appear in the result.
    score : str, optional
        Defaults to "bic" for discrete data and "bic-g" for continuous data,
        matching R.
    maxp : int, optional
        The largest number of parents any node may have.
    optimized : bool
        Reuse cached score deltas between iterations. Only affects speed.
    restart : int
        How many times to perturb the network it settles on and climb
        again, keeping the best result.  This is how hill climbing escapes a
        local optimum, and the perturbations come from R's generator -- seed
        with `set_seed()`.
    perturb : int
        How many arcs to change at each restart.
    **extra_args
        Score hyperparameters, e.g. ``iss=10`` for bde or ``k=1`` for bic.

    Returns
    -------
    BayesianNetwork
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.shape[1] < 2:
        raise ValueError("at least two variables are needed")
    restart = int(restart)
    if restart < 0:
        raise ValueError("restart must be a non-negative integer")
    perturb = int(perturb)
    if restart and perturb < 1:
        raise ValueError("perturb must be a positive integer")

    _check_complete(data)
    maxp = _check_maxp(maxp)

    nodes = [str(c) for c in data.columns]
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    score = _check_score(score, data)
    extra = _check_score_args(score, data, extra_args)

    def _amat_of(pairs):
        out = np.zeros((n, n), dtype=np.int32)
        for a, b in pairs or ():
            if a not in index or b not in index:
                raise ValueError(f"unknown node in arc ({a}, {b})")
            out[index[a], index[b]] = 1
        return out

    # whitelisting an arc forbids its reverse, which is what stops the
    # search from considering the other direction.
    whitelist = check_whitelist(whitelist, nodes, score_based=True)
    blacklist = build_blacklist(blacklist, whitelist)

    # ntests() reports how much work the search did, which only agrees with
    # R if the search took the same path -- a sharper check on the search
    # than the arc set, where two different paths can meet.  The counter
    # lives in the C core and is process-wide, so it is reset here and read
    # back below; the core lock keeps a concurrent search out of it.
    reset_test_counter()

    blmat = _amat_of(blacklist)
    wlmat = _amat_of(whitelist)

    arcs = _check_start(start, nodes)
    # whitelisted arcs are forced into the starting network, as greedy.search()
    # does before handing over to the search proper.
    for a, b in whitelist or ():
        arcs = _set_arc(arcs, a, b)
    # and blacklisted ones are removed from it.
    arcs = [(a, b) for a, b in arcs if not blmat[index[a], index[b]]]

    equivalence = _is_score_equivalent(score, extra) and optimized
    decomposable = _is_score_decomposable(score, extra)

    with Search(data, nodes, score, extra, blmat, wlmat) as search:

        if not search.acyclic(arcs):
            raise ValueError("the starting network contains cycles")

        # the reference score of the starting network, per node.
        search.set_reference(search.node_scores(arcs, nodes))

        updated = list(range(n))
        iterations = 0

        # random restarts: the best network seen so far, kept aside so that
        # a restart that goes badly can be discarded.
        best_arcs = best_total = None
        restarts_left = restart

        while True:
            amat = search.arcs_to_amat(arcs)
            nparents = amat.sum(axis=0).astype(np.float64)

            # refreshes the cache in place; only `updated` nodes are rescored
            # when optimized is on.
            search.fill_cache(
                arcs,
                updated if optimized else list(range(n)),
                amat, equivalence, decomposable)

            candidates = search.to_be_added(amat, nparents, maxp)

            # also folds the winning delta into the reference scores.
            best = search.best_step(amat, candidates, nparents, maxp)

            if best is None:
                if not restart or restarts_left < 0:
                    break

                total = float(np.sum(search.get_reference()))

                if restarts_left == restart:
                    # the first time through, there is nothing to compare
                    # against yet.
                    best_arcs, best_total = list(arcs), total
                    reverted = False
                elif _robust_score_difference(best_total, total) > 0:
                    # the restart made things worse; go back and perturb the
                    # network we came from instead.
                    arcs = list(best_arcs)
                    reverted = True
                else:
                    best_arcs, best_total = list(arcs), total
                    reverted = False

                if restarts_left == 0:
                    break
                if iterations >= max_iter:
                    break

                iterations += 1
                restarts_left -= 1

                from .graph import perturb as _perturb

                before = {node: set(BayesianNetwork(nodes, arcs).parents(node))
                          for node in nodes}
                arcs = _perturb(
                    BayesianNetwork(nodes, arcs,
                                    {"whitelist": list(whitelist or ()),
                                     "blacklist": list(blacklist or ())}),
                    perturb, maxp=maxp).arcs
                after = {node: set(BayesianNetwork(nodes, arcs).parents(node))
                         for node in nodes}

                if reverted:
                    # everything moved: the network being perturbed is not
                    # the one the reference scores describe.
                    search.set_reference(search.node_scores(arcs, nodes))
                    updated = list(range(n))
                else:
                    touched = [node for node in nodes
                               if before[node] != after[node]]
                    if touched:
                        reference = search.get_reference()
                        fresh = search.node_scores(arcs, touched)
                        for name, value in zip(touched, fresh):
                            reference[index[name]] = value
                        search.set_reference(reference)
                    updated = [index[node] for node in touched]

                continue

            frm, to, op = best["from"], best["to"], best["op"]
            cache = search.get_cache()
            arcs = _apply(arcs, op, frm, to)

            # An informative graph prior contributes to every node's score,
            # so a move changes the two endpoints' reference scores by more
            # than the cached delta accounts for.  Recompute them.  R does
            # this in hc() and not in tabu(), which is why the two land on
            # different members of the same equivalence class.
            if extra.get("prior") in _NON_DECOMPOSABLE_PRIORS:
                reference = search.get_reference()
                for name, value in zip((frm, to),
                                       search.node_scores(arcs, [frm, to])):
                    reference[index[name]] = value
                search.set_reference(reference)

            # A node whose reference score is -Inf has no baseline for the
            # cached delta to be a difference from, so if the delta itself is
            # finite the score has to be recomputed rather than added to.
            reference = search.get_reference()
            stale = [to] + ([frm] if op == "reverse" else [])
            recompute = []
            for name in stale:
                other = frm if name == to else to
                if (reference[index[name]] == -np.inf
                        and np.isfinite(cache[index[other], index[name]])):
                    recompute.append(name)
            if recompute:
                for name, value in zip(recompute,
                                       search.node_scores(arcs, recompute)):
                    reference[index[name]] = value
                search.set_reference(reference)

            # the nodes whose cached deltas are now stale.
            updated = ([index[frm], index[to]] if op == "reverse"
                       else [index[to]])

            iterations += 1
            if iterations >= max_iter:
                break

    return BayesianNetwork(
        nodes, arcs,
        learning={
            "algo": "hc",
            "test": score,
            "args": extra,
            "optimized": optimized,
            "whitelist": list(whitelist or ()),
            "blacklist": list(blacklist or ()),
            "iterations": iterations,
            "ntests": test_counter(),
        },
    )


def _robust_score_difference(new, old):
    """robust.score.difference(): compare two network scores.

    Networks with a singular node score -inf, and R is careful that -inf minus
    -inf reads as "no improvement" rather than NaN, and that anything finite
    beats -inf.  The tolerance keeps floating-point noise from registering as
    an improvement and letting the search cycle.
    """
    if new == -math.inf and old == -math.inf:
        return -math.inf
    if new != -math.inf and old == -math.inf:
        return abs(new)
    if abs(new - old) < math.sqrt(np.finfo(float).eps):
        return 0.0
    return new - old


def tabu(data, start=None, whitelist=None, blacklist=None, score=None,
         tabu=10, max_iter=float("inf"), maxp=float("inf"), optimized=True,
         **extra_args):
    """Learn a network structure by tabu search, as bnlearn's tabu() does.

    Unlike hill climbing, this keeps going when no move improves the score: it
    takes the least bad move instead, refusing any that would return to one of
    the last `tabu` networks, and gives up only after `tabu` consecutive
    iterations without beating the best network it has seen.  That best network
    is what comes back, not wherever the walk happened to stop.

    Parameters
    ----------
    data : pandas.DataFrame
    start : BayesianNetwork, optional
    whitelist, blacklist : sequence of (from, to), optional
    score : str, optional
        Defaults to "bic" for discrete data and "bic-g" for continuous data.
    tabu : int
        How many previous networks to remember, and how many fruitless
        iterations to tolerate.
    maxp : int, optional
        The largest number of parents any node may have.

    Returns
    -------
    BayesianNetwork
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.shape[1] < 2:
        raise ValueError("at least two variables are needed")
    if int(tabu) < 1:
        raise ValueError("the tabu list must have at least one slot")

    _check_complete(data)
    maxp = _check_maxp(maxp)

    tabu = int(tabu)
    nodes = [str(c) for c in data.columns]
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    score = _check_score(score, data)
    extra = _check_score_args(score, data, extra_args)

    whitelist = check_whitelist(whitelist, nodes, score_based=True)
    blacklist = build_blacklist(blacklist, whitelist)

    # ntests() reports how much work the search did, which only agrees with
    # R if the search took the same path -- a sharper check on the search
    # than the arc set, where two different paths can meet.  The counter
    # lives in the C core and is process-wide, so it is reset here and read
    # back below; the core lock keeps a concurrent search out of it.
    reset_test_counter()

    def _amat_of(pairs):
        out = np.zeros((n, n), dtype=np.int32)
        for a, b in pairs or ():
            if a not in index or b not in index:
                raise ValueError(f"unknown node in arc ({a}, {b})")
            out[index[a], index[b]] = 1
        return out

    blmat = _amat_of(blacklist)
    wlmat = _amat_of(whitelist)

    arcs = _check_start(start, nodes)
    for a, b in whitelist or ():
        arcs = _set_arc(arcs, a, b)
    arcs = [(a, b) for a, b in arcs if not blmat[index[a], index[b]]]

    equivalence = _is_score_equivalent(score, extra) and optimized
    decomposable = _is_score_decomposable(score, extra)

    with Search(data, nodes, score, extra, blmat, wlmat) as search:
        search.enable_tabu(tabu)

        if not search.acyclic(arcs):
            raise ValueError("the starting network contains cycles")

        search.set_reference(search.node_scores(arcs, nodes))

        updated = list(range(n))
        iterations = 1
        loss_iter = 0
        best_score = -math.inf
        best_arcs = None

        while True:
            current = (iterations - 1) % tabu
            reference = search.get_reference()
            total = float(reference.sum())

            # Keep the best network seen so far; the search returns that, not
            # wherever it stops.
            if total == -math.inf and best_score == -math.inf and iterations > 1:
                old = search.node_scores(best_arcs, nodes)
                singular_old = old == -math.inf
                singular_new = reference == -math.inf

                if np.array_equal(singular_old, singular_new):
                    # the same nodes are singular in both, so compare only the
                    # nodes that actually have a score.
                    delta = _robust_score_difference(
                        float(reference[~singular_new].sum()),
                        float(old[~singular_old].sum()))
                elif singular_new.sum() > singular_old.sum():
                    delta = -math.inf
                elif singular_new.sum() < singular_old.sum():
                    delta = math.inf
                else:
                    delta = _robust_score_difference(total, best_score)

                if delta > 0:
                    best_arcs, best_score = list(arcs), total
            elif (_robust_score_difference(total, best_score) > 0
                  or iterations == 1):
                best_arcs, best_score = list(arcs), total

            amat = search.arcs_to_amat(arcs)
            nparents = amat.sum(axis=0).astype(np.float64)

            search.hash_network(amat, current)
            search.fill_cache(
                arcs, updated if optimized else list(range(n)),
                amat, equivalence, decomposable)
            candidates = search.to_be_added(amat, nparents, maxp)

            best = search.tabu_best_step(amat, candidates, nparents, maxp,
                                         current, 0.0)

            if best is None:
                if loss_iter >= tabu:
                    arcs = best_arcs
                    break
                loss_iter += 1

                # nothing improves the score: take the least bad move instead.
                best = search.tabu_best_step(amat, candidates, nparents, maxp,
                                             current, -math.inf)
                if best is None:
                    if loss_iter > 0:
                        arcs = best_arcs
                    break
            elif _robust_score_difference(
                    float(search.get_reference().sum()), best_score) > 0:
                # The reference scores have to be re-read here: tabu_best_step
                # folded the chosen move's delta into them, so this compares
                # the score *after* the move, as R does.  Using the value from
                # the top of the loop resets the counter one iteration late and
                # the search stops in the wrong place.
                loss_iter = 0

            frm, to, op = best["from"], best["to"], best["op"]
            arcs = _apply(arcs, op, frm, to)

            updated = ([index[frm], index[to]] if op == "reverse"
                       else [index[to]])

            if iterations >= max_iter:
                if loss_iter > 0:
                    arcs = best_arcs
                break
            iterations += 1

    return BayesianNetwork(
        nodes, arcs,
        learning={
            "algo": "tabu",
            "test": score,
            "args": extra,
            "optimized": optimized,
            "whitelist": list(whitelist or ()),
            "blacklist": list(blacklist or ()),
            "tabu": tabu,
            "iterations": iterations,
            "ntests": test_counter(),
        },
    )


def alpha_star(network, data):
    """The imaginary sample size that makes BDe as close as possible to the
    empirical distribution.

    A BDe score needs one, and the usual answer is to pick a small number and
    hope; this computes the one that fits the data at hand.
    """
    _check_complete(data)

    nodes = [str(c) for c in data.columns]
    if set(nodes) != set(network.nodes):
        raise ValueError("the network and the data have different variables")
    if _data_type(data) != "discrete":
        raise ValueError("alpha_star() needs discrete data")

    present = set(network.arcs)
    if any((b, a) in present for a, b in present):
        raise ValueError("the graph is only partially directed")

    return alpha_star_value(network.nodes, network.arcs, data)


def BF(num, den, data, type=None, log=True, **extra_args):
    """The Bayes factor between two networks.

    How much more the data support one structure than the other, which for a
    Bayesian score is just the difference of the two scores -- the marginal
    likelihoods are what those scores already are.
    """
    if set(num.nodes) != set(den.nodes):
        raise ValueError("the two networks have different node sets")

    if type is None:
        type = {"discrete": "bde", "continuous": "bge",
                "mixed-cg": "bic-cg"}[_data_type(data)]

    difference = (score(num, data, type=type, **extra_args)
                  - score(den, data, type=type, **extra_args))

    if log:
        return difference

    # the Bayes factor of two networks that disagree at all is enormous, and
    # R returns Inf rather than failing.
    try:
        return math.exp(difference)
    except OverflowError:
        return math.inf if difference > 0 else 0.0


def whitelist(network):
    """The arcs the search was told to keep, if it was told anything."""
    return list(network.learning.get("whitelist") or [])


def blacklist(network):
    """The arcs the search was told to avoid, if it was told anything."""
    return list(network.learning.get("blacklist") or [])


def ntests(network):
    """How many tests or scores the search evaluated."""
    return network.learning.get("ntests")
