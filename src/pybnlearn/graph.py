"""Graph utilities and network comparison.

Ports of bnlearn's R/cpdag.R, R/utils-graph.R and the comparison functions in
R/frontend-bn.R and R/frontend-graph.R.  Most of these are a few lines around
an entry point the C core already exports.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import re

from ._core import (chow_liu_arcs, aracne_arcs, components, count_parameters,
                    cpdag_arcs, extend_pdag, structural_hamming,
                    tier_blacklist, topological_order, undirected_arcs)
from ._core import acyclic as _acyclic
from ._core import path_exists as _path_exists
from .structure import BayesianNetwork, _data_type, build_blacklist

__all__ = [
    "aracne", "chow_liu", "compare", "cpdag", "empty_graph", "hamming",
    "model2network", "moral", "nparams", "pdag2dag", "shd", "skeleton",
    "subgraph",
    "acyclic", "connected_components", "directed", "leaf_nodes",
    "node_ordering", "ordering2blacklist", "path_exists", "root_nodes",
    "set2blacklist", "tiers2blacklist", "valid_cpdag", "valid_dag",
    "valid_ug",
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


# ---------------------------------------------------------------------------
# graph properties, from R/frontend-graph.R
# ---------------------------------------------------------------------------

def _as_graph(x):
    """A fitted network answers structural questions from its parameters."""
    from .nodes import _graph

    return _graph(x)


def acyclic(x, directed=False):
    """Whether the graph contains no cycles.

    With `directed` left off, an undirected arc counts as a cycle of length
    two -- which is the check that matters for a partially directed graph,
    and the reason the flag exists.
    """
    net = _as_graph(x)
    return _acyclic(net.nodes, net.arcs, directed=bool(directed))


def directed(x):
    """Whether every arc has a direction."""
    net = _as_graph(x)
    present = set(net.arcs)
    return not any((b, a) in present for a, b in present)


def valid_dag(x):
    """Whether the graph is a directed acyclic graph."""
    net = _as_graph(x)
    return directed(net) and acyclic(net, directed=False)


def valid_ug(x):
    """Whether the graph is undirected.

    The empty graph counts, being both directed and undirected at once.
    """
    net = _as_graph(x)
    if not net.arcs:
        return True
    present = set(net.arcs)
    return all((b, a) in present for a, b in present)


def valid_cpdag(x):
    """Whether the graph is the equivalence class of some DAG.

    Being acyclic and partially directed is not enough: the undirected part
    has to be chordal, and the directions that *are* there have to be
    exactly the ones the v-structures compel.  The last check is the strong
    one, and it is done by comparing against cpdag() itself.
    """
    net = _as_graph(x)

    if not acyclic(net, directed=True):
        return False
    if not all(chordal for _, chordal in connected_components(net)):
        return False

    return set(cpdag(net).arcs) == set(net.arcs)


def connected_components(x):
    """The connected components of the undirected part, and whether each is
    chordal.

    bnlearn asks igraph whether a component is chordal; there is no igraph
    here, so the test below is a maximum-cardinality search.  Chordality is
    a property of the graph rather than of how you look for it, so any
    correct test agrees -- unlike most of this package, this is not a port.
    """
    net = _as_graph(x)
    loose = [(a, b) for a, b in net.arcs if (b, a) in set(net.arcs)]

    return [(component, _chordal(component, loose))
            for component in components(net.nodes, loose)]


def _chordal(nodes, arcs):
    """Maximum cardinality search: number the nodes so that each one is
    numbered right after as many of its numbered neighbours as possible,
    then check that every node's earlier neighbours form a clique."""
    if len(nodes) < 2:
        return True

    adjacent = {node: set() for node in nodes}
    for a, b in arcs:
        if a in adjacent and b in adjacent:
            adjacent[a].add(b)
            adjacent[b].add(a)

    weight = {node: 0 for node in nodes}
    order, numbered = [], set()

    for _ in nodes:
        node = max((n for n in nodes if n not in numbered),
                   key=lambda n: (weight[n], -list(nodes).index(n)))
        order.append(node)
        numbered.add(node)
        for other in adjacent[node] - numbered:
            weight[other] += 1

    position = {node: i for i, node in enumerate(order)}
    for node in order:
        earlier = {n for n in adjacent[node] if position[n] < position[node]}
        if not earlier:
            continue
        # the neighbour numbered last among them must be adjacent to the rest
        parent = max(earlier, key=position.__getitem__)
        if not (earlier - {parent}) <= adjacent[parent]:
            return False

    return True


def path_exists(x, frm, to, direct=True, underlying_graph=False):
    """Whether `to` can be reached from `frm`.

    `direct=False` ignores the arc between the two nodes, which is how you
    ask whether they are connected by anything *other* than being adjacent.
    """
    net = _as_graph(x)
    frm, to = str(frm), str(to)

    for node in (frm, to):
        if node not in net.nodes:
            raise ValueError(f"unknown node {node!r}")
    if frm == to:
        raise ValueError("'frm' and 'to' must be different from each other")

    return _path_exists(net.nodes, net.arcs, frm, to, direct=bool(direct),
                        underlying=bool(underlying_graph))


def node_ordering(x):
    """The nodes in topological order; the graph has to be a DAG."""
    net = _as_graph(x)
    if not directed(net):
        raise ValueError("the graph is only partially directed")
    return net.topological_order()


def _adjacent(net):
    """Every node's neighbours, undirected arcs included."""
    around = {node: set() for node in net.nodes}
    for a, b in net.arcs:
        around[a].add(b)
        around[b].add(a)
    return around


def root_nodes(x):
    """The nodes with no parents.

    A node touched by an undirected arc is not a root: the arc might turn
    out to point at it.  So having no parents is not enough -- every
    neighbour has to be a child.
    """
    net = _as_graph(x)
    around = _adjacent(net)
    return [n for n in net.nodes
            if not net.parents(n) and len(around[n]) == len(net.children(n))]


def leaf_nodes(x):
    """The nodes with no children, and none of whose arcs might become one."""
    net = _as_graph(x)
    around = _adjacent(net)
    return [n for n in net.nodes
            if not net.children(n) and len(around[n]) == len(net.parents(n))]


# ---------------------------------------------------------------------------
# turning orderings into blacklists
# ---------------------------------------------------------------------------

def ordering2blacklist(nodes):
    """Every arc that would run backwards through a node ordering.

    Blacklisting these is how you tell a learning algorithm the order the
    variables come in without telling it which arcs to draw.
    """
    if isinstance(nodes, BayesianNetwork):
        nodes = nodes.topological_order()
    elif hasattr(nodes, "nodes") and not isinstance(nodes, (list, tuple)):
        nodes = _as_graph(nodes).topological_order()

    return tier_blacklist([str(n) for n in nodes])


def tiers2blacklist(tiers):
    """The same, for an ordering that leaves some nodes tied.

    Each element of `tiers` is either a node or a group of nodes; arcs
    within a group are allowed in both directions, arcs from a later group
    to an earlier one are not.
    """
    return tier_blacklist(tiers)


def set2blacklist(nodes):
    """Every arc between the given nodes, in both directions.

    Blacklisting these keeps a group of variables from being connected to
    each other while leaving them free to connect to everything else.
    """
    nodes = [str(n) for n in nodes]
    if len(set(nodes)) != len(nodes):
        raise ValueError("the node labels are not unique")

    # expand.grid() varies the first factor fastest, so the arcs come out
    # grouped by their target rather than by their source.
    return [(a, b) for b in nodes for a in nodes if a != b]
