"""Check the node and arc utilities against R.

These are small functions and the values are rarely in doubt; the order and
the edge cases are.  Every case below is generated over the same eight
graphs -- a DAG, a chain, a partially directed graph, a complete undirected
triangle, a graph with an isolated node, a four-cycle whose undirected part
is not chordal, the empty graph, and one with two colliders -- and every
ordered pair of nodes within them, so that the awkward combinations are
reached by enumeration rather than by guessing which ones matter.

Fixtures come from tools/gen_r_nodes_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _records(kind):
    path = FIXTURES / "nodes.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


GRAPHS = {c["graph"]: c for c in _records("graph")}


def _network(name):
    case = GRAPHS[name]
    return pybnlearn.BayesianNetwork(case["nodes"],
                                     [tuple(a) for a in case["arcs"]])


def _arcs(pairs):
    return [tuple(a) for a in pairs]


# ---------------------------------------------------------------------------
# graph-level properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("graph"), ids=lambda c: c["graph"])
def test_graph_properties_match_r(case):
    net = _network(case["graph"])

    assert pybnlearn.nnodes(net) == case["nnodes"]
    assert pybnlearn.narcs(net) == case["narcs"]
    assert pybnlearn.acyclic(net) is case["acyclic"]
    assert pybnlearn.acyclic(net, directed=True) is case["acyclic.directed"]
    assert pybnlearn.directed(net) is case["directed"]
    assert pybnlearn.valid_dag(net) is case["valid.dag"]
    assert pybnlearn.valid_ug(net) is case["valid.ug"]
    assert pybnlearn.valid_cpdag(net) is case["valid.cpdag"]

    assert pybnlearn.root_nodes(net) == case["root.nodes"]
    assert pybnlearn.leaf_nodes(net) == case["leaf.nodes"]
    assert pybnlearn.isolated_nodes(net) == case["isolated.nodes"]

    assert pybnlearn.directed_arcs(net) == _arcs(case["directed.arcs"])
    assert pybnlearn.undirected_arcs(net) == _arcs(case["undirected.arcs"])
    assert (sorted(pybnlearn.compelled_arcs(net))
            == sorted(_arcs(case["compelled.arcs"])))

    if case["ordering"] is None:
        with pytest.raises(ValueError, match="partially directed"):
            pybnlearn.node_ordering(net)
    else:
        assert pybnlearn.node_ordering(net) == case["ordering"]


@pytest.mark.parametrize("case", _records("cyclic"), ids=lambda c: c["graph"])
def test_a_graph_with_a_directed_cycle_matches_r(case):
    """The other fixtures are all legitimate graphs, so acyclic(directed =
    True) would be True everywhere and never actually tested.  This one has
    a real cycle; most of the other functions would refuse it, so only the
    predicates that can answer are compared."""
    net = pybnlearn.BayesianNetwork(case["nodes"],
                                    [tuple(a) for a in case["arcs"]])

    assert pybnlearn.narcs(net) == case["narcs"]
    assert pybnlearn.acyclic(net) is case["acyclic"]
    assert pybnlearn.acyclic(net, directed=True) is case["acyclic.directed"]
    assert pybnlearn.directed(net) is case["directed"]
    assert pybnlearn.valid_dag(net) is case["valid.dag"]
    assert pybnlearn.valid_ug(net) is case["valid.ug"]


def test_every_property_takes_both_values_somewhere():
    """A predicate that always returned True would pass its own cases on
    whichever graphs happen to satisfy it."""
    cases = _records("graph") + _records("cyclic")

    for key in ("acyclic", "acyclic.directed", "directed", "valid.dag",
                "valid.ug"):
        assert {c[key] for c in cases} == {True, False}, key

    assert {c["valid.cpdag"] for c in _records("graph")} == {True, False}


# ---------------------------------------------------------------------------
# node-level queries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("node"),
                         ids=lambda c: f"{c['graph']}-{c['node']}")
def test_node_queries_match_r(case):
    net = _network(case["graph"])
    node = case["node"]

    assert pybnlearn.parents(net, node) == case["parents"]
    assert pybnlearn.children(net, node) == case["children"]
    assert pybnlearn.mb(net, node) == case["mb"]
    assert pybnlearn.nbr(net, node) == case["nbr1"]
    assert pybnlearn.nbr(net, node, max_dist=2) == case["nbr2"]
    assert pybnlearn.nbr(net, node, max_dist=3) == case["nbr3"]
    assert pybnlearn.spouses(net, node) == case["spouses"]
    assert pybnlearn.ancestors(net, node) == case["ancestors"]
    assert pybnlearn.descendants(net, node) == case["descendants"]

    assert pybnlearn.degree(net, node) == case["degree"]
    assert pybnlearn.in_degree(net, node) == case["in.degree"]
    assert pybnlearn.out_degree(net, node) == case["out.degree"]

    assert pybnlearn.incoming_arcs(net, node) == _arcs(case["incoming"])
    assert pybnlearn.outgoing_arcs(net, node) == _arcs(case["outgoing"])
    assert pybnlearn.incident_arcs(net, node) == _arcs(case["incident"])


def test_the_neighbourhood_actually_grows_with_the_distance():
    """nbr(max_dist = k) has to reach further than nbr(1) somewhere, or the
    expansion loop is never exercised."""
    cases = _records("node")
    assert any(len(c["nbr2"]) > len(c["nbr1"]) for c in cases)
    assert any(len(c["nbr3"]) > len(c["nbr2"]) for c in cases)


def test_ancestors_and_descendants_are_covered_beyond_one_step():
    cases = _records("node")
    assert any(len(c["ancestors"]) > 1 for c in cases)
    assert any(len(c["descendants"]) > 1 for c in cases)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("path"),
    ids=lambda c: f"{c['graph']}-{c['from']}-{c['to']}")
def test_path_existence_matches_r(case):
    net = _network(case["graph"])

    assert pybnlearn.path_exists(net, case["from"], case["to"]) is case["direct"]
    assert pybnlearn.path_exists(net, case["from"], case["to"],
                                 direct=False) is case["indirect"]
    assert pybnlearn.path_exists(net, case["from"], case["to"],
                                 underlying_graph=True) is case["underlying"]


def test_excluding_the_direct_arc_changes_the_answer_somewhere():
    """Otherwise `direct` is untested: it only matters for a pair that is
    adjacent and not otherwise connected."""
    assert any(c["direct"] != c["indirect"] for c in _records("path"))
    assert any(c["direct"] != c["underlying"] for c in _records("path"))


def test_a_node_has_no_path_to_itself(datasets):
    net = _network("dag")

    with pytest.raises(ValueError, match="different"):
        pybnlearn.path_exists(net, "A", "A")


# ---------------------------------------------------------------------------
# arc operations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("op"),
    ids=lambda c: f"{c['graph']}-{c['op']}-{c['from']}{c['to']}")
def test_arc_operations_match_r(case):
    net = _network(case["graph"])
    operation = getattr(pybnlearn, case["op"].replace(".", "_"))

    if case["error"]:
        with pytest.raises(ValueError):
            operation(net, case["from"], case["to"])
        return

    result = operation(net, case["from"], case["to"])

    assert result.arcs == _arcs(case["arcs"])
    assert result.nodes == net.nodes
    # the operations copy, as R's do
    assert net.arcs == _arcs(GRAPHS[case["graph"]]["arcs"])


def test_which_operations_can_fail_at_all():
    """Which operations have an error path is itself a fact about them:
    reverse.arc() refuses an undirected arc and a missing one, set.arc()
    refuses a change that would close a directed cycle, and the other three
    always have something sensible to do -- set.edge() included, because
    adding an undirected arc cannot create a *directed* cycle."""
    by_op = {}
    for case in _records("op"):
        by_op.setdefault(case["op"], set()).add(case["error"])

    for op in ("reverse.arc", "set.arc"):
        assert by_op[op] == {True, False}, op
    for op in ("drop.arc", "drop.edge", "set.edge"):
        assert by_op[op] == {False}, op


def test_the_operations_are_not_simply_add_and_remove():
    """The cases that make these worth porting: setting an arc that is
    already there undirected removes an arc rather than adding one, and
    dropping an edge that is directed does nothing at all."""
    net = _network("pdag")

    # A - B is undirected; setting it orients it, which *removes* a row --
    # the arc count is unchanged, because an undirected arc counted once
    # already.
    oriented = pybnlearn.set_arc(net, "A", "B")
    assert ("A", "B") in oriented.arcs and ("B", "A") not in oriented.arcs
    assert len(oriented.arcs) == len(net.arcs) - 1
    assert oriented.narcs == net.narcs

    # C -> D is directed; drop_edge leaves it, drop_arc removes it.
    assert pybnlearn.drop_edge(net, "C", "D").arcs == net.arcs
    assert ("C", "D") not in pybnlearn.drop_arc(net, "C", "D").arcs


def test_a_cycle_is_refused():
    net = _network("chain")

    with pytest.raises(ValueError, match="cycles"):
        pybnlearn.set_arc(net, "E", "A")

    assert pybnlearn.set_arc(net, "E", "A", check_cycles=False).narcs == 5


# ---------------------------------------------------------------------------
# node operations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("addnode"), ids=lambda c: c["graph"])
def test_add_node_matches_r(case):
    got = pybnlearn.add_node(_network(case["graph"]), "ZZ")

    assert got.nodes == case["nodes"]
    assert got.arcs == _arcs(case["arcs"])


@pytest.mark.parametrize("case", _records("removenode"),
                         ids=lambda c: f"{c['graph']}-{c['node']}")
def test_remove_node_matches_r(case):
    got = pybnlearn.remove_node(_network(case["graph"]), case["node"])

    assert got.nodes == case["nodes"]
    assert got.arcs == _arcs(case["arcs"])


@pytest.mark.parametrize("case", _records("rename"), ids=lambda c: c["graph"])
def test_rename_nodes_matches_r(case):
    got = pybnlearn.rename_nodes(_network(case["graph"]), case["labels"])

    assert got.nodes == case["nodes"]
    assert got.arcs == _arcs(case["arcs"])


def test_renaming_carries_the_learning_metadata_with_it(datasets):
    """A blacklist names nodes; if it is not relabelled too it silently
    starts referring to nothing."""
    data = datasets["learning.test"]
    learned = pybnlearn.hc(data, blacklist=[("A", "B")])
    learned.learning["blacklist"] = [("A", "B")]

    renamed = pybnlearn.rename_nodes(learned, [f"n{i}" for i in
                                               range(len(learned.nodes))])

    assert renamed.learning["blacklist"] == [("n0", "n1")]


def test_removing_the_last_node_is_refused():
    net = pybnlearn.empty_graph(["A"])

    with pytest.raises(ValueError, match="only node"):
        pybnlearn.remove_node(net, "A")


def test_adding_a_node_twice_is_refused():
    net = _network("dag")

    with pytest.raises(ValueError, match="already present"):
        pybnlearn.add_node(net, "A")


# ---------------------------------------------------------------------------
# blacklists from orderings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("ordering2blacklist"),
                         ids=lambda c: c["name"])
def test_ordering2blacklist_matches_r(case, datasets):
    got = pybnlearn.ordering2blacklist(case["nodes"])

    assert got == _arcs(case["blacklist"])


@pytest.mark.parametrize("case", _records("tiers2blacklist"),
                         ids=lambda c: c["name"])
def test_tiers2blacklist_matches_r(case):
    got = pybnlearn.tiers2blacklist(case["tiers"])

    assert got == _arcs(case["blacklist"])


@pytest.mark.parametrize("case", _records("set2blacklist"),
                         ids=lambda c: "-".join(c["set"]))
def test_set2blacklist_matches_r(case):
    got = pybnlearn.set2blacklist(case["set"])

    assert got == _arcs(case["blacklist"])


def test_a_blacklist_from_an_ordering_actually_constrains_a_search(datasets):
    """The point of the blacklist: learning under it must not produce an arc
    that runs backwards through the ordering."""
    data = datasets["learning.test"]
    ordering = ["F", "E", "D", "C", "B", "A"]

    learned = pybnlearn.hc(data,
                           blacklist=pybnlearn.ordering2blacklist(ordering))

    rank = {node: i for i, node in enumerate(ordering)}
    assert all(rank[a] < rank[b] for a, b in learned.arcs)


# ---------------------------------------------------------------------------
# fitted networks answer the same questions
# ---------------------------------------------------------------------------

def test_a_fitted_network_answers_structural_questions(datasets):
    data = datasets["learning.test"]
    learned = pybnlearn.hc(data)
    fitted = pybnlearn.fit(learned, data)

    for node in learned.nodes:
        assert pybnlearn.parents(fitted, node) == pybnlearn.parents(learned,
                                                                    node)
        assert pybnlearn.mb(fitted, node) == pybnlearn.mb(learned, node)
        assert pybnlearn.degree(fitted, node) == pybnlearn.degree(learned, node)

    assert pybnlearn.narcs(fitted) == learned.narcs
    assert pybnlearn.root_nodes(fitted) == pybnlearn.root_nodes(learned)
    assert pybnlearn.valid_dag(fitted)
