"""Graph utilities and network comparison.

Ports of bnlearn's R/cpdag.R, R/utils-graph.R and the comparison functions in
R/frontend-bn.R and R/frontend-graph.R.  Most of these are a few lines around
an entry point the C core already exports.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import re

from ._core import (chow_liu_arcs, aracne_arcs, count_parameters, cpdag_arcs,
                    extend_pdag, structural_hamming, topological_order,
                    undirected_arcs)
from .structure import BayesianNetwork, _data_type, build_blacklist

__all__ = [
    "aracne", "chow_liu", "compare", "cpdag", "empty_graph", "hamming",
    "model2network", "moral", "nparams", "pdag2dag", "shd", "skeleton",
    "subgraph",
]


def _match_nodes(a, b):
    if set(a.nodes) != set(b.nodes):
        raise ValueError("the two networks have different node sets")


def empty_graph(nodes):
    """A graph with the given nodes and no arcs."""
    return BayesianNetwork([str(n) for n in nodes], [])


def cpdag(net, wlbl=False):
    """The completed partially directed acyclic graph: the equivalence class
    the network belongs to."""
    arcs = cpdag_arcs(
        net.arcs, net.nodes,
        whitelist=net.learning.get("whitelist") if wlbl else None,
        blacklist=net.learning.get("blacklist") if wlbl else None,
        moral=False, fix=False, wlbl=wlbl)
    return BayesianNetwork(net.nodes, arcs, dict(net.learning))


def pdag2dag(net, ordering=None):
    """Orient any remaining undirected arcs, giving a DAG in the same
    equivalence class."""
    ordering = list(ordering) if ordering else topological_order(
        net.nodes, net.arcs)
    return BayesianNetwork(net.nodes, extend_pdag(net.arcs, ordering),
                           dict(net.learning))


def skeleton(net):
    """The underlying undirected graph."""
    return BayesianNetwork(net.nodes, undirected_arcs(net.nodes, net.arcs),
                           dict(net.learning))


def moral(net):
    """The moral graph: the skeleton with the parents of each node joined."""
    return BayesianNetwork(net.nodes,
                           undirected_arcs(net.nodes, net.arcs, moral=True),
                           dict(net.learning))


def shd(learned, true, use_cpdag=True):
    """Structural Hamming distance.

    By default the networks are reduced to their CPDAGs first, so that two
    graphs that encode the same independencies score zero even when they orient
    a reversible arc differently -- which is what makes the measure meaningful
    for comparing learned structures.
    """
    _match_nodes(learned, true)
    if use_cpdag:
        learned = cpdag(learned)
        true = cpdag(true)
    return structural_hamming(learned.nodes, learned.arcs, true.arcs)


def hamming(learned, true):
    """Hamming distance between the two skeletons: direction is ignored."""
    _match_nodes(learned, true)
    return structural_hamming(learned.nodes,
                              undirected_arcs(learned.nodes, learned.arcs),
                              undirected_arcs(true.nodes, true.arcs))


def _directed(arcs):
    return [(a, b) for a, b in arcs if (b, a) not in arcs]


def _undirected(arcs):
    """One entry per undirected edge, in the order the arcs appear."""
    seen, out = set(), []
    for a, b in arcs:
        if (b, a) in arcs and (b, a) not in seen:
            seen.add((a, b))
            out.append((a, b))
    return out


def compare(target, current, arcs=False):
    """Count true positives, false positives and false negatives.

    Directed and undirected arcs are compared separately, as compare.backend()
    does: a directed arc counts as recovered only if its direction matches too.
    """
    _match_nodes(target, current)

    t_dir, t_und = _directed(target.arcs), _undirected(target.arcs)
    c_dir, c_und = _directed(current.arcs), _undirected(current.arcs)

    c_dir_set, t_dir_set = set(c_dir), set(t_dir)
    c_und_set = {frozenset(a) for a in c_und}
    t_und_set = {frozenset(a) for a in t_und}

    tp = ([a for a in t_dir if a in c_dir_set]
          + [a for a in t_und if frozenset(a) in c_und_set])
    fn = ([a for a in t_dir if a not in c_dir_set]
          + [a for a in t_und if frozenset(a) not in c_und_set])
    fp = ([a for a in c_dir if a not in t_dir_set]
          + [a for a in c_und if frozenset(a) not in t_und_set])

    if arcs:
        return {"tp": tp, "fp": fp, "fn": fn}
    return {"tp": len(tp), "fp": len(fp), "fn": len(fn)}


def nparams(net, data, estimator=None):
    """The number of free parameters the network implies for this data."""
    if estimator is None:
        estimator = "bic" if _data_type(data) == "discrete" else "bic-g"
    return count_parameters(net.nodes, net.arcs, data, estimator)


def subgraph(net, nodes):
    """The subgraph spanning a subset of the nodes."""
    nodes = [str(n) for n in nodes]
    unknown = set(nodes) - set(net.nodes)
    if unknown:
        raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))
    keep = set(nodes)
    return BayesianNetwork(
        nodes, [(a, b) for a, b in net.arcs if a in keep and b in keep])


_MODEL_TOKEN = re.compile(r"\[([^\]]+)\]")


def model2network(modelstring):
    """Build a network from a model string such as "[A][C][B|A:C]"."""
    nodes, arcs = [], []

    tokens = _MODEL_TOKEN.findall(modelstring)
    if not tokens or "".join(f"[{t}]" for t in tokens) != modelstring.strip():
        raise ValueError(f"malformed model string: {modelstring!r}")

    for token in tokens:
        node, _, parents = token.partition("|")
        node = node.strip()
        if not node:
            raise ValueError(f"malformed model string: {modelstring!r}")
        nodes.append(node)
        for parent in filter(None, (p.strip() for p in parents.split(":"))):
            arcs.append((parent, node))

    unknown = {a for arc in arcs for a in arc} - set(nodes)
    if unknown:
        raise ValueError("model string refers to undeclared node(s): "
                         + ", ".join(sorted(unknown)))

    if len(set(nodes)) != len(nodes):
        raise ValueError(f"model string declares a node twice: {modelstring!r}")

    # R sorts the nodes rather than keeping the order they were declared in,
    # and the node order decides how modelstring() lists each node's parents,
    # so the round trip only reproduces R's output if this sorts too.  The
    # comparison is byte-wise, as R's is in the C locale.
    return BayesianNetwork(sorted(nodes), arcs)


# ---------------------------------------------------------------------------
# pairwise mutual-information learners
# ---------------------------------------------------------------------------

def _check_mi(mi, data):
    """check.mi.estimator(): the default depends on the data type."""
    kind = _data_type(data)
    if mi is None:
        return {"discrete": "mi", "continuous": "mi-g"}.get(kind) or _unsupported(kind)
    # only the maximum-likelihood estimators are available here; the
    # shrinkage variants apply to independence tests, not to these.
    if mi not in ("mi", "mi-g"):
        raise ValueError(
            f"unknown mutual information estimator {mi!r}; use 'mi' for "
            "discrete data or 'mi-g' for continuous data")
    return mi


def _unsupported(kind):
    raise ValueError(
        f"{kind} data is not supported by the pairwise learners yet")


def _prepare_lists(whitelist, blacklist, nodes):
    """chow.liu() and aracne() work on undirected graphs, so the C side is
    given exactly one direction per constrained edge."""
    whitelist = [(str(a), str(b)) for a, b in (whitelist or ())]
    blacklist = [(str(a), str(b)) for a, b in (blacklist or ())]

    for a, b in whitelist + blacklist:
        if a not in nodes or b not in nodes:
            raise ValueError(f"unknown node in arc ({a}, {b})")

    blacklist = build_blacklist(blacklist, whitelist)
    # only edges banned in both directions constrain an undirected graph.
    both = [(a, b) for a, b in blacklist if (b, a) in blacklist]
    blacklist = extend_pdag(both, nodes) if both else []
    whitelist = extend_pdag(whitelist, nodes) if whitelist else []

    return whitelist, blacklist


def chow_liu(data, whitelist=None, blacklist=None, mi=None):
    """Chow-Liu: the maximum-weight spanning tree of pairwise mutual
    information."""
    nodes = [str(c) for c in data.columns]
    estimator = _check_mi(mi, data)
    whitelist, blacklist = _prepare_lists(whitelist, blacklist, nodes)

    if len(whitelist) > len(nodes):
        raise ValueError("the whitelist contains more arcs than a tree can hold")

    arcs = chow_liu_arcs(data, nodes, estimator,
                         whitelist or None, blacklist or None)
    return BayesianNetwork(nodes, arcs, {
        "algo": "chow.liu", "test": estimator, "undirected": True,
        "whitelist": whitelist, "blacklist": blacklist,
    })


def aracne(data, whitelist=None, blacklist=None, mi=None):
    """ARACNE: pairwise mutual information filtered by the data processing
    inequality."""
    nodes = [str(c) for c in data.columns]
    estimator = _check_mi(mi, data)
    whitelist, blacklist = _prepare_lists(whitelist, blacklist, nodes)

    arcs = aracne_arcs(data, estimator, whitelist or None, blacklist or None)
    return BayesianNetwork(nodes, arcs, {
        "algo": "aracne", "test": estimator, "undirected": True,
        "whitelist": whitelist, "blacklist": blacklist,
    })
