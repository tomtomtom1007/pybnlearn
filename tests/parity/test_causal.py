"""Check interventions, twin networks and counterfactuals against R.

None of this is numerical on the graph side -- it is surgery -- which means
a mistake produces a plausible graph rather than a wrong number.  So the
node sets and arc sets are compared exactly, over four shapes that differ in
how far an intervention propagates: a chain, a collider, a diamond and a
wider network.  Every node of every graph is intervened on in turn, rather
than a few chosen ones.

The parameterised half is where the idea shows.  A twin network turns each
node's residual variance into a node of its own feeding *both* copies, so
the copies become deterministic given their parents and everything random
about them is shared.  That shared noise is what makes them counterfactual
rather than a second sample, and it is visible in the coefficients.

Fixtures come from tools/gen_r_causal_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _records(kind):
    path = FIXTURES / "causal.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(pairs):
    return sorted(tuple(a) for a in pairs)


@pytest.fixture(scope="session")
def gaussian(datasets):
    return pybnlearn.fit(
        pybnlearn.model2network("[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"),
        datasets["gaussian.test"], method="mle-g")


@pytest.fixture(scope="session")
def discrete(datasets):
    return pybnlearn.fit(
        pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
        datasets["learning.test"])


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("twin"), ids=lambda c: c["graph"])
def test_twin_network_matches_r(case):
    got = pybnlearn.twin(pybnlearn.model2network(case["modelstring"]))

    assert got.nodes == case["nodes"]
    assert _arcs(got.arcs) == _arcs(case["arcs"])


def test_the_twin_shares_its_noise(case_free=None):
    """The copies are joined to the originals only through the exogenous
    nodes -- that shared noise is the whole point, and it is what separates
    a twin network from two independent copies."""
    twin = pybnlearn.twin(pybnlearn.model2network("[A][B|A][C|B]"))

    factual = {"A", "B", "C"}
    copies = {"A.", "B.", "C."}

    crossing = [(a, b) for a, b in twin.arcs
                if (a in factual and b in copies)
                or (a in copies and b in factual)]
    assert not crossing

    for node in factual:
        shared = set(twin.parents(node)) & set(twin.parents(node + "."))
        assert shared == {"u" + node}


@pytest.mark.parametrize("case", _records("intervention"),
                         ids=lambda c: f"{c['graph']}-{c['node']}")
def test_intervention_matches_r(case):
    network = pybnlearn.model2network(case["modelstring"])

    got = pybnlearn.intervention(network, {case["node"]: "x"})

    assert got.nodes == case["nodes"]
    assert _arcs(got.arcs) == _arcs(case["arcs"])


def test_intervention_removes_the_incoming_arcs_only():
    """The difference between doing and seeing: an intervened variable stops
    being produced by its parents, but goes on producing its children."""
    network = pybnlearn.model2network("[A][B|A][C|B]")

    got = pybnlearn.intervention(network, {"B": "x"})

    assert got.parents("B") == []
    assert got.children("B") == ["C"]


def test_mutilated_is_the_same_function():
    assert pybnlearn.mutilated is pybnlearn.intervention


@pytest.mark.parametrize(
    "case", _records("counterfactual"),
    ids=lambda c: (f"{c['graph']}-{c['node']}"
                   + ("-merged" if c["merging"] else "-whole")))
def test_counterfactual_matches_r(case):
    network = pybnlearn.model2network(case["modelstring"])

    got = pybnlearn.counterfactual(network, {case["node"] + ".": "x"},
                                   merging=case["merging"])

    assert got.nodes == case["nodes"]
    assert _arcs(got.arcs) == _arcs(case["arcs"])


def test_merging_drops_the_copies_nothing_changed():
    """A copy differs from its original only if the intervention reached it.
    Merging is the claim that the rest would have been exactly as
    observed."""
    cases = {(c["graph"], c["node"], c["merging"]): c["nodes"]
             for c in _records("counterfactual")}

    smaller = [k for k in cases
               if k[2] and len(cases[k]) < len(cases[k[0], k[1], False])]
    assert smaller, "merging never dropped anything"

    # and the copy that *was* intervened on always survives.
    for graph, node, merging in cases:
        if merging:
            assert node + "." in cases[graph, node, merging]


def test_a_counterfactual_must_intervene_on_a_copy():
    """Intervening on the factual node would change what actually happened,
    which is not what a counterfactual asks."""
    network = pybnlearn.model2network("[A][B|A][C|B]")

    with pytest.raises(ValueError, match="full stop"):
        pybnlearn.counterfactual(network, {"B": "x"})


def test_a_counterfactual_cannot_be_taken_twice():
    network = pybnlearn.model2network("[A][B|A][C|B]")
    once = pybnlearn.counterfactual(network, {"B.": "x"})

    with pytest.raises(ValueError, match="already"):
        pybnlearn.counterfactual(once, {"C.": "x"})


# ---------------------------------------------------------------------------
# structural causal models
# ---------------------------------------------------------------------------

def test_the_causal_model_round_trips():
    """as_scm makes the error terms explicit and as_bn hides them again;
    the network in between is the one you started with."""
    for modelstring in ("[A][B|A][C|B]", "[A][B][C|A:B]",
                        "[A][C][F][B|A][D|A:C][E|B:F]"):
        network = pybnlearn.model2network(modelstring)
        back = pybnlearn.as_bn(pybnlearn.as_scm(network))

        assert back.nodes == network.nodes
        assert _arcs(back.arcs) == _arcs(network.arcs)


def test_the_causal_model_names_the_noise():
    scm = pybnlearn.as_scm(pybnlearn.model2network("[A][B|A]"))

    assert scm.factual == ["A", "B"]
    assert scm.exogenous == ["uA", "uB"]
    assert ("uA", "A") in scm.arcs and ("uB", "B") in scm.arcs


def test_a_partially_directed_graph_has_no_causal_reading():
    network = pybnlearn.BayesianNetwork(["A", "B"],
                                        [("A", "B"), ("B", "A")])

    with pytest.raises(ValueError, match="partially directed"):
        pybnlearn.as_scm(network)


# ---------------------------------------------------------------------------
# fitted networks
# ---------------------------------------------------------------------------

def _by_node(records, **selector):
    out = {}
    for case in records:
        if all(case.get(k) == v for k, v in selector.items()):
            out[case["node"]] = case
    return out


def _check_gaussian(node, case):
    assert list(node.parents) == case["parents"]
    assert list(node.children) == case["children"]
    assert list(node.coefficients) == case["coefnames"]
    for name, expected in zip(case["coefnames"], case["coefficients"]):
        assert float(node.coefficients[name]) == pytest.approx(
            expected, rel=1e-11, abs=1e-12), name
    assert node.sd == pytest.approx(case["sd"][0], rel=1e-11, abs=1e-12)


def test_twin_of_a_fitted_network_matches_r(gaussian):
    got = pybnlearn.twin(gaussian)

    nodes = _records("twin.fitted.nodes")[0]["nodes"]
    assert got.nodes == nodes

    for name, case in _by_node(_records("twin.fitted")).items():
        _check_gaussian(got[name], case)


def test_the_twins_noise_carries_the_original_variance(gaussian):
    """Each node's residual variance becomes a node of its own, entering
    both copies with a coefficient of one -- so the copies are deterministic
    and the randomness is shared."""
    got = pybnlearn.twin(gaussian)

    for node in gaussian.nodes:
        assert got["u" + node].sd == gaussian[node].sd
        assert got[node].sd == 0.0
        assert got[node + "."].sd == 0.0
        assert got[node].coefficients["u" + node] == 1.0
        assert got[node + "."].coefficients["u" + node] == 1.0


@pytest.mark.parametrize("fixed", ["A", "C", "F"])
def test_intervention_on_a_fitted_gaussian_matches_r(fixed, gaussian):
    got = pybnlearn.intervention(gaussian, {fixed: 3.5})

    for name, case in _by_node(_records("intervention.fitted"),
                               fixed=fixed).items():
        _check_gaussian(got[name], case)


@pytest.mark.parametrize("fixed", ["B", "D"])
def test_intervention_on_a_fitted_discrete_matches_r(fixed, discrete):
    got = pybnlearn.intervention(discrete, {fixed: "b"})

    for name, case in _by_node(_records("intervention.fitted.discrete"),
                               fixed=fixed).items():
        node = got[name]
        assert list(node.parents) == case["parents"]
        assert list(node.children) == case["children"]
        assert list(node.levels[0]) == case["levels"]
        assert list(node.probabilities.shape) == case["dim"]
        # R unrolls column-major, which is numpy's order="F".
        assert np.allclose(node.probabilities.reshape(-1, order="F"),
                           case["prob"], rtol=1e-12, atol=1e-14)


def test_an_intervened_node_is_a_point_mass(discrete, gaussian):
    """An intervention does not condition, it sets: the node's distribution
    puts all of its mass on the value, and has no parents left to depend
    on."""
    fixed = pybnlearn.intervention(discrete, {"D": "b"})
    levels = list(fixed["D"].levels[0])
    assert fixed["D"].parents == []
    assert list(fixed["D"].probabilities) == [
        1.0 if level == "b" else 0.0 for level in levels]

    fixed = pybnlearn.intervention(gaussian, {"C": 3.5})
    assert fixed["C"].parents == []
    assert fixed["C"].sd == 0.0
    assert fixed["C"].coefficients == {"(Intercept)": 3.5}


@pytest.mark.parametrize("fixed", ["C.", "F."])
@pytest.mark.parametrize("merging", [True, False])
def test_counterfactual_on_a_fitted_network_matches_r(fixed, merging,
                                                      gaussian):
    got = pybnlearn.counterfactual(gaussian, {fixed: 2.0}, merging=merging)

    expected = [c for c in _records("counterfactual.fitted.nodes")
                if c["fixed"] == fixed and c["merging"] == merging]
    assert got.nodes == expected[0]["nodes"]

    for name, case in _by_node(_records("counterfactual.fitted"),
                               fixed=fixed, merging=merging).items():
        _check_gaussian(got[name], case)


def test_a_counterfactual_leaves_the_factual_side_alone(gaussian):
    """The point of the exercise: what actually happened is unchanged, so
    the two sides can be compared."""
    got = pybnlearn.counterfactual(gaussian, {"C.": 2.0}, merging=False)

    for node in gaussian.nodes:
        assert (list(got[node].coefficients)
                == list(gaussian[node].coefficients) + ["u" + node])
        assert got[node].sd == 0.0


def test_only_gaussian_networks_have_parameterised_twins(discrete):
    with pytest.raises(ValueError, match="Gaussian"):
        pybnlearn.twin(discrete)
