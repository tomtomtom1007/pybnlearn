"""Graph utilities and network comparison.

Ports of bnlearn's R/cpdag.R, R/utils-graph.R and the comparison functions in
R/frontend-bn.R and R/frontend-graph.R.  Most of these are a few lines around
an entry point the C core already exports.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import itertools
import re

from ._core import (chow_liu_arcs, aracne_arcs, components, count_parameters,
                    cpdag_arcs, extend_pdag, structural_hamming,
                    tier_blacklist, topological_order, undirected_arcs)
from ._core import (collider_triples, complement_arcs, consistent_extension,
                    deduplicate_arcs, ide_cozman_graphs,
                    intervention_distance, ordered_graphs, sample_indices)
from ._core import acyclic as _acyclic
from ._core import path_exists as _path_exists
from .structure import BayesianNetwork, _data_type, build_blacklist

__all__ = [
    "aracne", "chow_liu", "compare", "cpdag", "empty_graph", "hamming",
    "model2network", "moral", "nparams", "pdag2dag", "shd", "skeleton",
    "subgraph",
    "cextend", "cextend_all", "colliders", "complete_graph", "count_graphs",
    "dsep", "perturb", "random_graph",
    "shielded_colliders", "sid", "unshielded_colliders", "vstructs",
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

    # "[A|B][B|A]" declares A a parent of B and B a parent of A, which is how
    # an *undirected* arc is held -- so the string describes a graph a model
    # string cannot describe, and R refuses it rather than returning
    # something that will not round trip.  A longer cycle is refused for the
    # same reason: a model string is a factorisation, and a factorisation has
    # to be acyclic to mean anything.
    if len(set(arcs)) != len(arcs) or len({frozenset(a) for a in arcs}) != len(arcs):
        raise ValueError(
            f"the graph is only partially directed: {modelstring!r}")
    if not _acyclic(sorted(nodes), arcs, directed=True):
        raise ValueError(f"the graph contains cycles: {modelstring!r}")

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


# ---------------------------------------------------------------------------
# colliders, d-separation and consistent extensions
# ---------------------------------------------------------------------------

def colliders(x, arcs=False):
    """Every triple where two nodes both point at a third.

    Set `arcs` to get the arcs that make up the colliders instead of the
    triples.
    """
    return _colliders(x, arcs, shielded=True, unshielded=True)


def unshielded_colliders(x, arcs=False):
    """The colliders whose two parents are *not* themselves adjacent.

    These are the v-structures: the ones that constrain the equivalence
    class, because they are the only colliders the data can see.
    """
    return _colliders(x, arcs, shielded=False, unshielded=True)


def shielded_colliders(x, arcs=False):
    """The colliders whose two parents are adjacent, and which therefore say
    nothing about the equivalence class."""
    return _colliders(x, arcs, shielded=True, unshielded=False)


def vstructs(x, arcs=False):
    """An alias for `unshielded_colliders`, as in R."""
    return unshielded_colliders(x, arcs=arcs)


def _colliders(x, as_arcs, shielded, unshielded):
    net = _as_graph(x)
    triples = collider_triples(net.nodes, net.arcs, shielded=shielded,
                               unshielded=unshielded)

    if not as_arcs:
        return triples

    # each collider contributes its two arcs, but R stacks all the first
    # parents' arcs and then all the second parents' rather than pairing
    # them up, and only then removes duplicates.  The order survives.
    out = ([(a, middle) for a, middle, _ in triples]
           + [(b, middle) for _, middle, b in triples])
    return deduplicate_arcs(out, net.nodes)


def dsep(bn, x, y, z=None):
    """Whether two nodes are d-separated by a set of others.

    This is the graphical counterpart of conditional independence: if it
    holds, then every distribution the graph can represent makes x and y
    independent given z.

    The implementation is Koller and Friedman's, section 4.5: take the upper
    closure of the nodes involved, moralise it, remove z, and look for a
    path.  A partially directed graph is extended to a DAG first.
    """
    net = _as_graph(bn)
    x, y = str(x), str(y)
    z = [str(v) for v in (z or ())]

    for node in [x, y] + z:
        if node not in net.nodes:
            raise ValueError(f"unknown node {node!r}")

    # a node in the conditioning set is trivially separated from everything.
    if x in z or y in z:
        return True

    if not directed(net):
        net = cextend(net)

    from .nodes import ancestors

    closure = set()
    for node in [x, y] + z:
        closure.add(node)
        closure.update(ancestors(net, node))

    upper = subgraph(net, [n for n in net.nodes if n in closure])
    moralised = moral(upper)

    kept = [n for n in moralised.nodes if n not in z]
    blocked = subgraph(moralised, kept)

    return not path_exists(blocked, x, y)


def cextend(x, strict=True):
    """A DAG in the equivalence class the graph describes.

    Orienting the undirected arcs of a CPDAG without creating a cycle or a
    new v-structure; there is not always one, and `strict` decides whether
    that is an error or a partially directed answer.
    """
    net = _as_graph(x)

    if not acyclic(net, directed=True):
        raise ValueError("the specified network contains cycles")

    # pdag_extension(), not pdag2dag(): the first is Dor and Tarsi's
    # algorithm, which orients what it can without creating a v-structure,
    # while the second imposes a node ordering and would give a different
    # DAG from the same equivalence class.
    extended = BayesianNetwork(
        net.nodes, consistent_extension(net.arcs, net.nodes),
        dict(net.learning))

    if strict and not directed(extended):
        raise ValueError("no consistent extension of the graph is possible")

    return extended


def sid(learned, true):
    """Structural intervention distance.

    How many of the ordered pairs of nodes the learned graph would give the
    wrong intervention distribution for -- which is a different question
    from how many arcs it got wrong, and the reason two graphs with the same
    Hamming distance can be very differently useful.
    """
    learned, true = _as_graph(learned), _as_graph(true)
    if set(learned.nodes) != set(true.nodes):
        raise ValueError("the two networks have different node sets")

    # the C code indexes both graphs by position, so they have to agree.
    ordered = BayesianNetwork(true.nodes, learned.arcs, learned.learning)
    return int(intervention_distance(true.nodes, ordered.arcs, true.arcs))


# ---------------------------------------------------------------------------
# generating graphs
# ---------------------------------------------------------------------------

def complete_graph(nodes):
    """The DAG with every arc the node ordering allows."""
    nodes = [str(n) for n in nodes]
    return BayesianNetwork(nodes, tier_blacklist(list(reversed(nodes))),
                           {"algo": "complete"})


def random_graph(nodes, num=1, method="ordered", prob=None, burn_in=None,
                 every=1, max_in_degree=float("inf"),
                 max_out_degree=float("inf"), max_degree=float("inf")):
    """Random DAGs, drawn the way R draws them.

    Two families, and they are not interchangeable.  "ordered" fixes the
    node ordering and includes each arc it allows independently, which is
    fast but samples DAGs non-uniformly.  "ic-dag" and "melancon" run the
    Ide-Cozman Markov chain, which samples uniformly over connected and over
    all DAGs respectively, at the cost of a burn-in.

    Seed with `set_seed()`; the draws come from R's generator, so the same
    seed gives the same graphs R gives.
    """
    nodes = [str(n) for n in nodes]
    num = int(num)
    if num < 1:
        raise ValueError("the number of graphs must be a positive integer")

    if method == "empty":
        return _one_or_many([[] for _ in range(num)], nodes, method)

    if method == "ordered":
        if prob is None:
            # this default gives about as many arcs as there are nodes.
            prob = 2 / (len(nodes) - 1) if len(nodes) > 1 else 0.0
        if not 0 <= prob <= 1:
            raise ValueError("the branching probability must be in [0, 1]")
        return _one_or_many(ordered_graphs(nodes, num, float(prob)), nodes,
                            method)

    if method not in ("ic-dag", "melancon"):
        raise ValueError(
            "method must be 'ordered', 'ic-dag', 'melancon' or 'empty'")

    if burn_in is None:
        # the magic number comes from the reference implementation.
        burn_in = 6 * len(nodes) ** 2
    every = int(every)
    if every < 1:
        raise ValueError("the thinning factor must be a positive integer")

    generated = ide_cozman_graphs(
        nodes, num * every, int(burn_in), float(max_in_degree),
        float(max_out_degree), float(max_degree), method == "ic-dag")

    if every > 1:
        generated = generated[every - 1::every]

    return _one_or_many(generated, nodes, method)


def _one_or_many(arc_sets, nodes, method):
    graphs = [BayesianNetwork(nodes, arcs, {"algo": method})
              for arcs in arc_sets]
    return graphs[0] if len(graphs) == 1 else graphs


# ---------------------------------------------------------------------------
# perturbing a graph, and counting graphs
# ---------------------------------------------------------------------------

def perturb(x, nops, ops=("set", "drop", "reverse"), maxp=float("inf")):
    """Randomly change a few arcs.

    Hill climbing uses this to restart: perturbing the network it settled on
    and climbing again is how it escapes a local optimum.  Every choice --
    which operation, which arc -- is a draw from R's generator, so seed with
    `set_seed()` and the same perturbation comes out.

    An attempt only counts once the graph has actually moved away from the
    one it started as -- an operation chosen when no legal arc is available
    for it, or one that puts the graph back where it began, is free.  That
    is why the loop is bounded at three times `nops` rather than exactly
    `nops`.
    """
    net = _as_graph(x)
    nops = int(nops)
    if nops < 1:
        raise ValueError("the number of operations must be a positive integer")

    ops = [str(o) for o in ops]
    unknown = [o for o in ops if o not in ("set", "drop", "reverse")]
    if unknown:
        raise ValueError("the operations must be 'set', 'drop' or 'reverse'")

    nodes = list(net.nodes)
    arcs = list(net.arcs)
    whitelist = list(net.learning.get("whitelist") or ())
    blacklist = list(net.learning.get("blacklist") or ())

    original = list(arcs)
    remaining = nops

    for _ in range(3 * nops):
        if remaining == 0:
            break

        parents = {node: len(BayesianNetwork(nodes, arcs).parents(node))
                   for node in nodes}

        addable = [(a, b) for a, b in complement_arcs(arcs, nodes)
                   if (a, b) not in blacklist and parents[b] < maxp]
        droppable = [arc for arc in arcs if arc not in whitelist]
        reversible = [(a, b) for a, b in arcs
                      if (b, a) not in blacklist and parents[a] < maxp]

        # the operation is chosen before it is known to be possible, and a
        # choice that turns out to be impossible still costs an attempt.
        operation = ops[int(sample_indices(len(ops), 1, replace=True)[0]) - 1]

        if operation == "set" and addable:
            frm, to = addable[
                int(sample_indices(len(addable), 1, replace=True)[0]) - 1]
            if not _path_exists(nodes, arcs, to, frm, direct=True):
                arcs = _set_direction(arcs, frm, to)
        elif operation == "drop" and droppable:
            frm, to = droppable[
                int(sample_indices(len(droppable), 1, replace=True)[0]) - 1]
            arcs = [arc for arc in arcs if arc not in ((frm, to), (to, frm))]
        elif operation == "reverse" and reversible:
            frm, to = reversible[
                int(sample_indices(len(reversible), 1, replace=True)[0]) - 1]
            if not _path_exists(nodes, arcs, frm, to, direct=False):
                arcs = [arc for arc in arcs if arc != (frm, to)] + [(to, frm)]

        # the comparison is against the network as it arrived, not against
        # the previous pass: an operation that undoes an earlier one gives
        # the budget back.
        if arcs != original:
            remaining -= 1

    return BayesianNetwork(nodes, arcs, dict(net.learning))


def _set_direction(arcs, frm, to):
    """set.arc.direction() on a bare arc list."""
    present = set(arcs)
    if (to, frm) in present and (frm, to) in present:
        return [arc for arc in arcs if arc != (to, frm)]
    if (to, frm) in present:
        return [arc for arc in arcs if arc != (to, frm)] + [(frm, to)]
    if (frm, to) in present:
        return list(arcs)
    return list(arcs) + [(frm, to)]


def cextend_all(x):
    """Every DAG in the equivalence class, not just one.

    `cextend` returns one consistent extension; this returns all of them,
    which is how you find out how much the data left undecided.  The count
    grows very fast with the size of the undirected part, so this is for
    small graphs.
    """
    net = _as_graph(x)
    present = set(net.arcs)

    loose = [(a, b) for a, b in net.arcs if (b, a) in present]
    firm = [(a, b) for a, b in net.arcs if (b, a) not in present]

    if not loose:
        return [BayesianNetwork(net.nodes, net.arcs, dict(net.learning))]

    # only the nodes the undirected part touches matter; the rest are along
    # for the ride.
    touched = sorted({n for arc in loose for n in arc},
                     key=list(net.nodes).index)

    orientations = []
    for ordering in itertools.permutations(touched):
        rank = {node: i for i, node in enumerate(ordering)}
        oriented = {(a, b) if rank[a] < rank[b] else (b, a)
                    for a, b in loose}

        candidate = BayesianNetwork(net.nodes, sorted(oriented) + firm)
        # an extension may not introduce a v-structure the class does not
        # have, which is what makes this the equivalence class rather than
        # every acyclic orientation.
        if set(cpdag(candidate).arcs) != present:
            continue

        key = tuple(sorted(oriented))
        if key not in orientations:
            orientations.append(key)

    return [BayesianNetwork(net.nodes, list(o) + firm, dict(net.learning))
            for o in orientations]


def count_graphs(type="all-dags", nodes=None, k=None, r=None, eqclass=None):
    """How many graphs of a given kind there are on n nodes.

    Exact, with Python's own big integers -- the counts outrun a double at
    about nine nodes, and outrun anything printable soon after.

    Parameters
    ----------
    type : str
        "all-dags", "dags-given-ordering", "dags-with-k-roots",
        "dags-with-r-arcs", or "dags-in-equivalence-class".
    nodes : int or sequence of int
        How many nodes.
    k, r : int
        The number of roots or of arcs, for the counts that need one.
    eqclass : BayesianNetwork
        For "dags-in-equivalence-class".
    """
    if type == "dags-in-equivalence-class":
        if eqclass is None:
            raise ValueError("this count needs an equivalence class")
        return len(cextend_all(eqclass))

    if nodes is None:
        raise ValueError("this count needs the number of nodes")
    wanted = [nodes] if isinstance(nodes, int) else [int(n) for n in nodes]
    biggest = max(wanted)

    if type == "dags-given-ordering":
        # with the ordering fixed, every arc it allows is free to be there
        # or not, independently.
        counts = [1] + [2 ** _binomial(i, 2) for i in range(1, biggest + 1)]
    elif type == "all-dags":
        counts = _count_all_dags(biggest)
    elif type == "dags-with-k-roots":
        if k is None:
            raise ValueError("this count needs the number of roots")
        counts = _count_by_roots(biggest, int(k))
    elif type == "dags-with-r-arcs":
        if r is None:
            raise ValueError("this count needs the number of arcs")
        counts = _count_by_arcs(biggest, int(r))
    else:
        raise ValueError(f"unknown count {type!r}")

    values = [counts[n] for n in wanted]
    return values[0] if len(values) == 1 else values


def _binomial(n, k):
    from math import comb

    return comb(n, k) if 0 <= k <= n else 0


def _count_all_dags(n):
    """Robinson's recursion, inclusion-exclusion over the root set."""
    a = [1] + [0] * n
    for i in range(1, n + 1):
        total = 0
        for k in range(i, 0, -1):
            sign = 1 if (k - 1) % 2 == 0 else -1
            total += sign * _binomial(i, k) * 2 ** (k * (i - k)) * a[i - k]
        a[i] = total
    return a


def _count_by_roots(n, k):
    """Graphs with exactly k roots.

    Peel the roots off: choose which j of the i nodes they are, then count
    the graphs on the remaining i - j, weighted by how many ways the roots
    can attach.  A node that used to be a root has to gain at least one
    parent among the new ones -- that is the (2^j - 1) factor -- while the
    rest may gain any.
    """
    if k > n:
        return [0] * (n + 1)

    a = [None] + [[0] * (i + 1) for i in range(1, n + 1)]

    for i in range(1, n + 1):
        for j in range(1, i + 1):
            if i == j:
                # every node a root means no arcs at all.
                a[i][j] = 1
                continue

            total = 0
            for m in range(1, i - j + 1):
                to_old_roots = (2 ** j - 1) ** m
                to_old_nonroots = 2 ** (j * (i - m - j))
                total += to_old_roots * to_old_nonroots * a[i - j][m]

            a[i][j] = total * _binomial(i, j)

    return [0 if i < k else a[i][k] for i in range(n + 1)]


def _count_by_arcs(n, r):
    """Graphs with exactly r arcs, by the same inclusion-exclusion carried
    over the arc count as well."""
    biggest = max(_binomial(n, 2), r)
    a = [[0] * (biggest + 1) for _ in range(n + 1)]
    a[0][0] = 1

    for i in range(1, n + 1):
        a[i][0] = 1
        for j in range(1, min(r, _binomial(i, 2)) + 1):
            total = 0
            for m in range(1, i):
                sign = 1 if (m - 1) % 2 == 0 else -1
                inner = 0
                for kk in range(0, min(j, _binomial(i - m, 2)) + 1):
                    inner += _binomial(m * (i - m), j - kk) * a[i - m][kk]
                total += sign * _binomial(i, m) * inner
            a[i][j] = total

    return [row[r] if r <= biggest else 0 for row in a]
