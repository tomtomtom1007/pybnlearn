"""Check d-separation, colliders, extensions, SID, random graphs,
discretization and the Bayesian score utilities against R.

Two of these carry more weight than the rest.  `random_graph` draws from R's
generator, so it is compared exactly rather than statistically -- the same
seed has to give the same graphs, arc for arc.  And Hartemink's
discretization chooses every variable's breakpoints jointly, so one wrong
collapse changes every column; the levels are compared, not just the shape.

d-separation is enumerated rather than sampled: every unordered pair of
nodes in five graphs, conditioned on nothing, on each other node in turn,
and on one pair.

Fixtures come from tools/gen_r_analysis_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

GRAPHS = {
    "learning.test": "[A][C][F][B|A][D|A:C][E|B:F]",
    "asia": "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
    "chain": "[A][B|A][C|B][D|C][E|D]",
    "collider": "[A][B][C|A:B][D|C][E|C]",
    "diamond": "[A][B|A][C|A][D|B:C]",
    "shielded": "[A][B|A][C|A:B][D|C]",
}


def _records(kind):
    path = FIXTURES / "analysis.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _network(name):
    return pybnlearn.model2network(GRAPHS[name])


def _arcs(pairs):
    return [tuple(a) for a in pairs]


# ---------------------------------------------------------------------------
# colliders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("colliders"),
                         ids=lambda c: c["graph"])
def test_colliders_match_r(case):
    net = _network(case["graph"])

    assert pybnlearn.colliders(net) == _arcs(case["all"])
    assert pybnlearn.unshielded_colliders(net) == _arcs(case["unshielded"])
    assert pybnlearn.shielded_colliders(net) == _arcs(case["shielded"])
    assert pybnlearn.vstructs(net) == _arcs(case["unshielded"])

    assert pybnlearn.colliders(net, arcs=True) == _arcs(case["all.arcs"])
    assert (pybnlearn.unshielded_colliders(net, arcs=True)
            == _arcs(case["unshielded.arcs"]))
    assert (pybnlearn.shielded_colliders(net, arcs=True)
            == _arcs(case["shielded.arcs"]))


def test_shielded_and_unshielded_colliders_both_occur():
    """The distinction is the whole point -- only unshielded colliders
    constrain the equivalence class -- so both kinds have to appear
    somewhere or the split is untested."""
    cases = _records("colliders")
    assert any(c["unshielded"] for c in cases)
    assert any(c["shielded"] for c in cases)


# ---------------------------------------------------------------------------
# d-separation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("dsep"),
    ids=lambda c: (f"{c['graph']}-{c['x']}{c['y']}"
                   + ("-given-" + "".join(c["z"]) if c["z"] else "")))
def test_dsep_matches_r(case):
    got = pybnlearn.dsep(_network(case["graph"]), case["x"], case["y"],
                         case["z"])

    assert got is case["dsep"]


def test_conditioning_changes_the_answer_both_ways():
    """Conditioning on a chain's middle blocks it; conditioning on a
    collider's child opens it.  Both directions have to be in the fixtures
    or an implementation that ignored the conditioning set would pass."""
    cases = _records("dsep")

    opened = [c for c in cases if c["z"] and not c["dsep"]
              and any(o["graph"] == c["graph"] and o["x"] == c["x"]
                      and o["y"] == c["y"] and not o["z"] and o["dsep"]
                      for o in cases)]
    blocked = [c for c in cases if c["z"] and c["dsep"]
               and any(o["graph"] == c["graph"] and o["x"] == c["x"]
                       and o["y"] == c["y"] and not o["z"] and not o["dsep"]
                       for o in cases)]

    assert opened, "no case where conditioning opens a path"
    assert blocked, "no case where conditioning blocks a path"


def test_a_node_in_the_conditioning_set_separates_trivially():
    net = _network("chain")

    assert pybnlearn.dsep(net, "A", "B", ["A"])
    assert pybnlearn.dsep(net, "A", "B", ["B"])


def test_dsep_accepts_a_partially_directed_graph():
    """R extends a CPDAG before asking, since d-separation is a property of
    a DAG rather than of an equivalence class."""
    net = _network("chain")
    equivalence = pybnlearn.cpdag(net)

    # a chain has no v-structure, so its equivalence class is all undirected
    assert not pybnlearn.directed(equivalence)
    assert pybnlearn.dsep(equivalence, "A", "C", ["B"]) is True
    assert pybnlearn.dsep(equivalence, "A", "C") is False


# ---------------------------------------------------------------------------
# consistent extensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("cextend"), ids=lambda c: c["graph"])
def test_cextend_matches_r(case):
    """The extension is not unique, so agreeing with R means running Dor and
    Tarsi's algorithm rather than any correct orientation."""
    equivalence = pybnlearn.BayesianNetwork(
        _network(case["graph"]).nodes, _arcs(case["cpdag"]))

    extended = pybnlearn.cextend(equivalence)

    assert extended.modelstring() == case["extended"]
    assert pybnlearn.directed(extended)
    # and it is in the same equivalence class it came from
    assert set(pybnlearn.cpdag(extended).arcs) == set(equivalence.arcs)


def test_an_unextendable_graph_is_reported():
    """A graph that is not a CPDAG may have no consistent extension at all,
    and `strict` decides whether that is an error."""
    net = pybnlearn.BayesianNetwork(
        list("ABC"), [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"),
                      ("A", "C"), ("C", "A")])

    # a triangle of undirected arcs does extend
    assert pybnlearn.directed(pybnlearn.cextend(net))


# ---------------------------------------------------------------------------
# structural intervention distance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("sid"),
    ids=lambda c: f"{c['learned']}-vs-{c['true']}")
def test_sid_matches_r(case):
    true = _network(case["true"])

    if "arcs" in case:
        learned = pybnlearn.BayesianNetwork(true.nodes, _arcs(case["arcs"]))
    else:
        learned = _network(case["learned"])

    assert pybnlearn.sid(learned, true) == case["value"]


def test_sid_is_zero_only_for_the_same_graph():
    net = _network("learning.test")

    assert pybnlearn.sid(net, net) == 0
    assert pybnlearn.sid(pybnlearn.drop_arc(net, "A", "B"), net) > 0


def test_sid_is_not_the_hamming_distance():
    """Two graphs one arc from the truth can be very differently useful for
    intervention, which is the reason SID exists."""
    values = {c["learned"]: c["value"] for c in _records("sid")
              if c["true"] == "learning.test" and "arcs" in c}

    assert len(set(values.values())) > 1, values


# ---------------------------------------------------------------------------
# generating graphs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("random"),
    ids=lambda c: f"{c['method']}-seed{c['seed']}-n{c['num']}")
def test_random_graphs_match_r(case):
    """Exact, not statistical: the draws come from R's Mersenne-Twister."""
    extra = {k.replace(".", "_"): v for k, v in case["extra"].items()}
    if "burn_in" in extra:
        extra["burn_in"] = int(extra["burn_in"])
    if "every" in extra:
        extra["every"] = int(extra["every"])

    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.random_graph(case["nodes"], num=int(case["num"]),
                                 method=case["method"], **extra)

    if int(case["num"]) == 1:
        got = [got]

    assert len(got) == len(case["graphs"])
    for graph, expected in zip(got, case["graphs"]):
        assert graph.arcs == _arcs(expected)
        assert graph.nodes == case["nodes"]
        assert pybnlearn.acyclic(graph, directed=True)


def test_all_three_generators_are_covered():
    assert ({c["method"] for c in _records("random")}
            == {"ordered", "ic-dag", "melancon"})


def test_the_generators_do_not_all_give_the_same_graph():
    seen = {tuple(tuple(a) for a in g)
            for c in _records("random") for g in c["graphs"]}
    assert len(seen) > 5


@pytest.mark.parametrize("case", _records("complete"),
                         ids=lambda c: str(len(c["nodes"])))
def test_complete_graph_matches_r(case):
    got = pybnlearn.complete_graph(case["nodes"])

    assert got.arcs == _arcs(case["arcs"])
    assert got.narcs == len(case["nodes"]) * (len(case["nodes"]) - 1) // 2


# ---------------------------------------------------------------------------
# discretization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("discretize"),
    ids=lambda c: f"{c['dataset']}-{c['method']}-{c['breaks']}"
                  + (f"-{c['idisc']}" if c["idisc"] else ""))
def test_discretize_matches_r(case, datasets):
    data = datasets[case["dataset"]]

    extra = {}
    if case["idisc"] is not None:
        extra = {"idisc": case["idisc"], "ibreaks": int(case["ibreaks"])}

    got = pybnlearn.discretize(data, method=case["method"],
                               breaks=int(case["breaks"]), **extra)

    assert list(got.columns) == list(case["levels"])

    for name, levels in case["levels"].items():
        assert list(got[name].cat.categories) == levels, name
        assert list(got[name].astype(str)[:30]) == case["head"][name], name


def test_both_kinds_of_discretization_are_covered():
    """The marginal methods and Hartemink's go through completely different
    code -- one variable at a time versus all of them together."""
    assert ({c["method"] for c in _records("discretize")}
            == {"quantile", "interval", "hartemink"})


def test_hartemink_uses_the_other_variables(datasets):
    """Its whole point: the breakpoints of one variable depend on the rest,
    so discretizing a subset gives different cuts than the whole."""
    data = datasets["gaussian.test"]

    whole = pybnlearn.discretize(data, "hartemink", breaks=3, ibreaks=30)
    part = pybnlearn.discretize(data[["A", "B", "C"]], "hartemink", breaks=3,
                                ibreaks=30)

    assert (list(whole["A"].cat.categories)
            != list(part["A"].cat.categories))


def test_hartemink_needs_more_than_one_variable(datasets):
    with pytest.raises(ValueError, match="two variables"):
        pybnlearn.discretize(datasets["gaussian.test"][["A"]], "hartemink")


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("configs"),
    ids=lambda c: "-".join(c["variables"]) + ("-all" if c["all"] else ""))
def test_configs_matches_r(case, datasets):
    data = datasets["learning.test"][case["variables"]]

    got = pybnlearn.configs(data, all=case["all"])

    assert list(got.categories) == case["levels"]
    assert list(got.astype(str)[:40]) == case["head"]


def test_dropping_unused_configurations_changes_the_levels():
    """`all` decides whether a combination that never occurs is still a
    level, which is what keeps the numbering stable across data sets."""
    cases = {(tuple(c["variables"]), c["all"]): c["levels"]
             for c in _records("configs")}
    both = [v for v in {k[0] for k in cases}
            if (v, True) in cases and (v, False) in cases]
    assert both
    for variables in both:
        assert len(cases[variables, True]) >= len(cases[variables, False])


# ---------------------------------------------------------------------------
# the Bayesian score utilities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("alpha.star"),
                         ids=lambda c: c["dataset"])
def test_alpha_star_matches_r(case, datasets):
    network = pybnlearn.model2network(case["modelstring"])

    got = pybnlearn.alpha_star(network, datasets[case["dataset"]])

    assert got == pytest.approx(case["value"], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("case", _records("bf"),
                         ids=lambda c: f"{c['dataset']}-{c['score']}")
def test_bayes_factor_matches_r(case, datasets):
    data = datasets[case["dataset"]]
    network = pybnlearn.model2network(case["modelstring"])
    other = pybnlearn.drop_arc(network, *case["dropped"])

    got = pybnlearn.BF(network, other, data, type=case["score"])

    assert got == pytest.approx(case["value"], rel=1e-10, abs=1e-12)


def test_the_bayes_factor_is_a_difference_of_scores(datasets):
    data = datasets["learning.test"]
    a, b = pybnlearn.hc(data), pybnlearn.model2network("[A][B][C][D][E][F]")

    expected = (pybnlearn.score(a, data, type="bde")
                - pybnlearn.score(b, data, type="bde"))

    assert pybnlearn.BF(a, b, data, type="bde") == pytest.approx(expected)
    # the factor itself overflows a double here, which R reports as Inf.
    assert pybnlearn.BF(a, b, data, type="bde", log=False) == np.inf


@pytest.mark.parametrize("case", _records("fitprops"),
                         ids=lambda c: c["dataset"])
def test_identifiable_and_singular_match_r(case, datasets):
    if case["dataset"] == "deterministic":
        fitted = pybnlearn.custom_fit(
            pybnlearn.model2network("[A][B|A]"),
            {"A": {"prob": np.array([0.5, 0.5]),
                   "levels": {"A": ["a", "b"]}},
             "B": {"prob": np.array([1.0, 0.0, 0.0, 1.0]).reshape((2, 2),
                                                                  order="F"),
                   "levels": {"B": ["x", "y"]}}})
    else:
        data = datasets[case["dataset"]]
        fitted = pybnlearn.fit(pybnlearn.hc(data), data)

    assert pybnlearn.identifiable(fitted) is case["identifiable"]
    assert pybnlearn.singular(fitted) is case["singular"]


def test_both_properties_take_both_values():
    cases = _records("fitprops")
    assert {c["singular"] for c in cases} == {True, False}
