"""Reading and editing a network's nodes and arcs.

This covers bnlearn's R/frontend-nodes.R and R/frontend-arcs.R.  Nothing
here touches data: these are questions about, and edits to, a graph.

Everything returns a new network rather than modifying the one it was given,
which is also what R does -- `set.arc(dag, "A", "B")` leaves `dag` alone --
though for a different reason.

The one thing worth reading carefully is what the five edit operations do
when the arc is already there in some form.  They are not "add" and
"remove": `set_arc` on an undirected arc orients it, `reverse_arc` flips
whichever direction happens to be present, `set_edge` on a directed arc
takes the direction away, and `drop_edge` refuses to touch a directed one.
Those are bnlearn's semantics and they are reproduced here rather than
simplified.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

from ._core import acyclic as _acyclic
from ._core import node_structure
from .fit import FittedNetwork
from .structure import BayesianNetwork

__all__ = [
    "add_node", "ancestors", "arcs", "children", "compelled_arcs",
    "degree", "descendants", "directed_arcs", "drop_arc", "drop_edge",
    "in_degree", "incident_arcs", "incoming_arcs", "isolated_nodes", "mb",
    "narcs", "nbr", "nnodes", "out_degree", "outgoing_arcs", "parents",
    "remove_node", "rename_nodes", "reverse_arc", "reversible_arcs",
    "set_arc", "set_edge", "spouses", "undirected_arcs",
]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _graph(x):
    """Both kinds of network answer these questions; a fitted one answers
    them from its parameters."""
    if isinstance(x, BayesianNetwork):
        return x
    if isinstance(x, FittedNetwork):
        return BayesianNetwork(
            x.nodes,
            [(parent, node) for node in x.nodes for parent in x[node].parents],
            x.learning)
    raise TypeError("a BayesianNetwork or a FittedNetwork is required")


def _check_node(network, node):
    node = str(node)
    if node not in network.nodes:
        raise ValueError(f"unknown node {node!r}")
    return node


def arcs(x):
    """Every arc, in the order the network holds them."""
    return list(_graph(x).arcs)


def nnodes(x):
    return len(_graph(x).nodes)


def narcs(x):
    """How many arcs, counting an undirected one once."""
    return _graph(x).narcs


def parents(x, node):
    return _graph(x).parents(_check_node(_graph(x), node))


def children(x, node):
    return _graph(x).children(_check_node(_graph(x), node))


def mb(x, node):
    """The Markov blanket: the parents, the children, and the children's
    other parents -- everything that has to be known before the node becomes
    independent of the rest of the network."""
    network = _graph(x)
    node = _check_node(network, node)
    return list(node_structure(network.nodes, network.arcs)[node]["mb"])


def nbr(x, node, max_dist=1):
    """The neighbourhood: the nodes an arc away, and with `max_dist` above
    one, the nodes reachable in that many arcs."""
    network = _graph(x)
    node = _check_node(network, node)

    if not isinstance(max_dist, int) or max_dist < 1:
        raise ValueError("the maximum distance must be a positive integer")
    max_dist = min(max_dist, len(network.nodes))

    structure = node_structure(network.nodes, network.arcs)

    def around(name):
        return list(structure[name]["nbr"])

    reached = around(node)
    frontier = reached

    for _ in range(2, max_dist + 1):
        expanded = [n for name in frontier for n in around(name)]
        frontier = [n for n in dict.fromkeys(expanded)
                    if n not in reached and n != node]
        reached = list(dict.fromkeys(reached + frontier))

        if len(reached) == len(network.nodes):
            break

    return reached


def spouses(x, node):
    """The other parents of this node's children."""
    network = _graph(x)
    node = _check_node(network, node)

    out = []
    for child in network.children(node):
        out.extend(network.parents(child))
    return [n for n in dict.fromkeys(out) if n != node]


def ancestors(x, node):
    """Every node with a directed path to this one."""
    network = _graph(x)
    node = _check_node(network, node)
    return _reachable(network, node, reverse=True)


def descendants(x, node):
    """Every node this one has a directed path to."""
    network = _graph(x)
    node = _check_node(network, node)
    return _reachable(network, node, reverse=False)


def _reachable(network, start, reverse):
    """topological.ordering(start = node): the nodes downstream of a node,
    or upstream of it, in the order R returns them.

    R sorts by the depth the traversal assigns, and its sort is stable, so
    nodes at the same depth come out in the network's own node order.
    """
    step = network.parents if reverse else network.children

    depth = {start: 0}
    frontier, level = [start], 0
    while frontier:
        level += 1
        nxt = []
        for node in frontier:
            for other in step(node):
                if other not in depth:
                    depth[other] = level
                    nxt.append(other)
        frontier = nxt

    found = [n for n in network.nodes if n in depth and n != start]
    return sorted(found, key=lambda n: depth[n])


def in_degree(x, node):
    return len(parents(x, node))


def out_degree(x, node):
    return len(children(x, node))


def degree(x, node):
    """How many nodes are adjacent, counting an undirected arc once."""
    network = _graph(x)
    node = _check_node(network, node)
    return len(node_structure(network.nodes, network.arcs)[node]["nbr"])


def isolated_nodes(x):
    """The nodes with no arcs at all."""
    network = _graph(x)
    structure = node_structure(network.nodes, network.arcs)
    return [n for n in network.nodes if not structure[n]["nbr"]]


# ---------------------------------------------------------------------------
# reading arcs
# ---------------------------------------------------------------------------

def directed_arcs(x):
    """The arcs that appear in one direction only."""
    network = _graph(x)
    present = set(network.arcs)
    return [(a, b) for a, b in network.arcs if (b, a) not in present]


def undirected_arcs(x):
    """The arcs that appear in both directions, listed both ways round, as
    R lists them."""
    network = _graph(x)
    present = set(network.arcs)
    return [(a, b) for a, b in network.arcs if (b, a) in present]


def incoming_arcs(x, node):
    node = _check_node(_graph(x), node)
    return [(a, b) for a, b in directed_arcs(x) if b == node]


def outgoing_arcs(x, node):
    node = _check_node(_graph(x), node)
    return [(a, b) for a, b in directed_arcs(x) if a == node]


def incident_arcs(x, node):
    """Every arc that touches the node, in either direction."""
    network = _graph(x)
    node = _check_node(network, node)
    return [(a, b) for a, b in network.arcs if a == node or b == node]


def compelled_arcs(x):
    """The arcs whose direction the data fix: those still directed in the
    equivalence class."""
    from .graph import cpdag

    return directed_arcs(cpdag(_graph(x)))


def reversible_arcs(x):
    """The arcs whose direction is arbitrary: those the equivalence class
    leaves undirected."""
    from .graph import cpdag

    network = _graph(x)
    loose = {frozenset(arc) for arc in undirected_arcs(cpdag(network))}
    return [arc for arc in network.arcs if frozenset(arc) in loose]


# ---------------------------------------------------------------------------
# editing arcs
# ---------------------------------------------------------------------------

def set_arc(x, frm, to, check_cycles=True):
    """Make the arc run from `frm` to `to`.

    Orients it if it is there undirected, turns it round if it is there the
    other way, and adds it if it is not there at all.
    """
    network, frm, to = _edit_arguments(x, frm, to)
    present = set(network.arcs)

    if (to, frm) in present and (frm, to) in present:
        updated = [arc for arc in network.arcs if arc != (to, frm)]
    elif (to, frm) in present:
        updated = [arc for arc in network.arcs if arc != (to, frm)] + [(frm, to)]
    elif (frm, to) in present:
        updated = list(network.arcs)
    else:
        updated = list(network.arcs) + [(frm, to)]

    return _rebuild(network, updated, check_cycles)


def drop_arc(x, frm, to):
    """Remove any arc between the two nodes, whichever way it points."""
    network, frm, to = _edit_arguments(x, frm, to)
    updated = [arc for arc in network.arcs
               if arc not in ((frm, to), (to, frm))]
    return _rebuild(network, updated, check_cycles=False)


def reverse_arc(x, frm, to, check_cycles=True):
    """Turn round whichever arc runs between the two nodes."""
    network, frm, to = _edit_arguments(x, frm, to)
    present = set(network.arcs)

    if (to, frm) in present and (frm, to) in present:
        raise ValueError("an undirected arc cannot be reversed")
    if (to, frm) in present:
        updated = [arc for arc in network.arcs if arc != (to, frm)] + [(frm, to)]
    elif (frm, to) in present:
        updated = [arc for arc in network.arcs if arc != (frm, to)] + [(to, frm)]
    else:
        raise ValueError(f"no arc to be reversed between {frm!r} and {to!r}")

    return _rebuild(network, updated, check_cycles)


def set_edge(x, frm, to, check_cycles=True):
    """Make the two nodes adjacent without committing to a direction."""
    network, frm, to = _edit_arguments(x, frm, to)
    present = set(network.arcs)

    if (to, frm) in present and (frm, to) in present:
        updated = list(network.arcs)
    elif (to, frm) in present:
        updated = list(network.arcs) + [(frm, to)]
    elif (frm, to) in present:
        updated = list(network.arcs) + [(to, frm)]
    else:
        updated = list(network.arcs) + [(frm, to), (to, frm)]

    return _rebuild(network, updated, check_cycles)


def drop_edge(x, frm, to):
    """Remove an undirected arc.

    A directed arc is left alone: dropping it would be a decision about the
    graph rather than the removal of an undecided adjacency, and `drop_arc`
    is what makes that decision.
    """
    network, frm, to = _edit_arguments(x, frm, to)
    present = set(network.arcs)

    if (frm, to) in present and (to, frm) in present:
        updated = [arc for arc in network.arcs
                   if arc not in ((frm, to), (to, frm))]
    else:
        updated = list(network.arcs)

    return _rebuild(network, updated, check_cycles=False)


def _edit_arguments(x, frm, to):
    if not isinstance(x, BayesianNetwork):
        raise TypeError("a BayesianNetwork is required to edit arcs")

    frm, to = _check_node(x, frm), _check_node(x, to)
    if frm == to:
        raise ValueError("'frm' and 'to' must be different from each other")
    return x, frm, to


def _rebuild(network, updated, check_cycles):
    if check_cycles and not _acyclic(network.nodes, updated, directed=True):
        raise ValueError("the resulting graph contains cycles")
    return BayesianNetwork(network.nodes, updated, network.learning)


# ---------------------------------------------------------------------------
# editing nodes
# ---------------------------------------------------------------------------

def add_node(x, node):
    """Add an isolated node.

    It goes on the end, as R's does.  Where a node sits in the node order is
    not cosmetic -- it decides the order parents are listed in a model
    string, among other things -- so it is not sorted into place.
    """
    if not isinstance(x, BayesianNetwork):
        raise TypeError("a BayesianNetwork is required")

    node = str(node)
    if node in x.nodes:
        raise ValueError(f"node {node!r} is already present in the graph")

    return BayesianNetwork(list(x.nodes) + [node], x.arcs, x.learning)


def remove_node(x, node):
    """Remove a node and every arc that touched it."""
    if not isinstance(x, BayesianNetwork):
        raise TypeError("a BayesianNetwork is required")

    node = _check_node(x, node)
    if len(x.nodes) == 1:
        raise ValueError("trying to remove the only node in the graph")

    # R keeps the learning metadata here, unlike subgraph(), so that a
    # network still knows what it was learned with after a node goes.
    kept = [n for n in x.nodes if n != node]
    return BayesianNetwork(
        kept, [(a, b) for a, b in x.arcs if a != node and b != node],
        x.learning)


def rename_nodes(x, names):
    """Relabel every node, in order.

    The learning metadata is relabelled too, so a whitelist or blacklist
    still refers to the nodes it was meant to.
    """
    if not isinstance(x, BayesianNetwork):
        raise TypeError("a BayesianNetwork is required")

    names = [str(n) for n in names]
    if len(names) != len(x.nodes):
        raise ValueError(
            f"{len(names)} labels for {len(x.nodes)} nodes")
    if len(set(names)) != len(names):
        raise ValueError("the new node labels are not unique")

    rename = dict(zip(x.nodes, names))
    learning = dict(x.learning)
    for key in ("whitelist", "blacklist", "illegal"):
        if learning.get(key):
            learning[key] = [(rename[a], rename[b]) for a, b in learning[key]]

    return BayesianNetwork(
        names, [(rename[a], rename[b]) for a, b in x.arcs], learning)
