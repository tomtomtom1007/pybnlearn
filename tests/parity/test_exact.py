"""Check exact inference against gRain.

This is the one part of the suite that is not checking a port.  bnlearn has no
exact inference of its own -- it hands the network to gRain -- so pybnlearn
implements the junction tree directly and these fixtures compare an
independent implementation's numbers.

That makes the claim different in kind from the rest of the suite.  Elsewhere,
agreeing with R means reproducing its choices, because those choices change
the answer.  Here the answer is a property of the network: any correct
implementation agrees, whatever elimination order it picks.  So a tolerance is
the right comparison, and it is not evidence that the algorithms match.

Two further checks make up for that: the junction tree is compared against
brute-force enumeration of the same network, which shares no code with it, and
against the Monte Carlo estimate from cpquery, which shares even less.

Fixtures come from tools/gen_r_exact_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

STRUCTURES = {
    "learning.test": "[A][C][F][B|A][D|A:C][E|B:F]",
    "asia": "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
    "lizards": "[Species][Diameter|Species][Height|Species]",
    # gRain cannot express these names, so it appears only in the
    # brute-force comparison below.
    "coronary": ("[Smoking][M. Work|Smoking][P. Work|Smoking]"
                 "[Pressure|Smoking][Proteins|Smoking:M. Work]"
                 "[Family|M. Work]"),
}


@pytest.fixture(scope="session")
def networks(datasets):
    return {
        name: pybnlearn.fit(pybnlearn.model2network(modelstring),
                            datasets[name])
        for name, modelstring in STRUCTURES.items()
    }


def _records(kind):
    path = FIXTURES / "exact.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


@pytest.mark.parametrize(
    "case", _records("query"),
    ids=lambda c: (f"{c['network']}-{'.'.join(c['nodes'])}"
                   + ("-given-" + ".".join(sorted(c["evidence"]))
                      if c["evidence"] else "")))
def test_query_matches_grain(case, networks):
    result = pybnlearn.query(networks[case["network"]], case["nodes"],
                             case["evidence"] or None)

    assert result.variables == case["nodes"]
    for variable, levels in zip(case["nodes"], case["levels"]):
        assert result.levels[variable] == levels

    # gRain's as.vector() unrolls column-major, which is numpy's order="F".
    got = result.values.reshape(-1, order="F")
    assert np.allclose(got, case["values"], rtol=1e-10, atol=1e-12)
    assert result.values.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "case", _records("exact.predict"),
    ids=lambda c: f"{c['network']}-{c['node']}")
def test_exact_prediction_matches_r(case, datasets, networks):
    data = datasets[case["network"]].head(int(case["n"]))

    pybnlearn.set_seed(int(case["seed"]))
    predicted, probabilities = pybnlearn.predict(
        networks[case["network"]], case["node"], data, method="exact",
        prob=True)

    assert list(np.asarray(predicted).astype(str)) == case["predicted"]
    assert list(probabilities.columns) == case["levels"]
    for level, expected in case["probabilities"].items():
        assert np.allclose(probabilities[level].to_numpy(), expected,
                           rtol=1e-10, atol=1e-12), f"level {level}"


# ---------------------------------------------------------------------------
# checks that do not go through gRain
# ---------------------------------------------------------------------------

def _brute_force(fitted, nodes, evidence=None):
    """Multiply every table together and marginalise.

    Exponential, and only usable on the small networks here, but it shares no
    code with the junction tree: it does not triangulate, build a tree or
    propagate anything.
    """
    from pybnlearn.exact import Factor

    joint = None
    for node in fitted.nodes:
        entry = fitted[node]
        variables = [node] + list(entry.parents)
        levels = {v: list(fitted[v].levels[0]) for v in variables}
        factor = Factor(variables, levels, entry.probabilities)
        joint = factor if joint is None else joint * factor

    if evidence:
        joint = joint.observe(evidence)

    # marginalise() keeps the joint's own variable order, so the result is
    # put back into the order that was asked for before it is compared.
    return joint.marginalise(nodes).reorder(nodes).normalise()


@pytest.mark.parametrize("network", sorted(STRUCTURES))
def test_junction_tree_agrees_with_brute_force(network, networks):
    """Independent of gRain, and covers coronary, whose variable names gRain
    cannot express."""
    fitted = networks[network]

    for node in fitted.nodes:
        expected = _brute_force(fitted, [node])
        got = pybnlearn.query(fitted, node)
        assert np.allclose(got.values, expected.values, rtol=1e-12,
                           atol=1e-14), f"marginal of {node}"

    # and one conditional per network, on a node with parents
    conditioned = next((n for n in fitted.nodes if fitted[n].parents), None)
    if conditioned is not None:
        parent = fitted[conditioned].parents[0]
        value = fitted[parent].levels[0][0]
        expected = _brute_force(fitted, [conditioned], {parent: value})
        got = pybnlearn.query(fitted, conditioned, {parent: value})
        assert np.allclose(got.values, expected.values, rtol=1e-12, atol=1e-14)


def test_exact_agrees_with_the_monte_carlo_estimate(networks):
    """cpquery samples where query() computes; with enough particles they
    should meet, which is a check on both."""
    fitted = networks["asia"]

    exact = pybnlearn.query(fitted, "D", {"S": "yes"})
    index = fitted["D"].levels[0].index("yes")

    pybnlearn.set_seed(1)
    sampled = pybnlearn.cpquery(fitted, {"D": "yes"}, {"S": "yes"},
                                method="lw", n=200000)

    assert sampled == pytest.approx(float(exact.values[index]), abs=5e-3)


def test_variables_that_do_not_share_a_clique(networks):
    """The tree can only answer a joint directly when one clique covers all of
    it; this exercises the fallback."""
    fitted = networks["asia"]

    got = pybnlearn.query(fitted, ["A", "S", "X", "D"])
    expected = _brute_force(fitted, ["A", "S", "X", "D"])

    assert np.allclose(got.values, expected.values, rtol=1e-12, atol=1e-14)


def test_impossible_evidence_is_reported(networks):
    """Conditioning on something the network gives probability zero has no
    answer, and saying so beats returning NaNs."""
    fitted = networks["learning.test"]

    # make one configuration impossible, then condition on it.
    fitted["B"].probabilities[:, 0] = [1.0, 0.0, 0.0]

    with pytest.raises(ZeroDivisionError, match="probability zero"):
        pybnlearn.query(fitted, "D", {"A": "a", "B": "b"})


def test_unknown_nodes_and_levels_are_reported(networks):
    fitted = networks["asia"]

    with pytest.raises(ValueError, match="unknown node"):
        pybnlearn.query(fitted, "D", {"nonesuch": "yes"})

    with pytest.raises(ValueError, match="not a level"):
        pybnlearn.query(fitted, "D", {"S": "maybe"})


def test_gaussian_networks_go_the_other_way(datasets):
    """query() dispatches on the network: a Gaussian one is a multivariate
    normal, so it never reaches the junction tree.  Covered in detail by
    test_mvnorm.py; asserted here so that the dispatch itself is tested from
    the discrete side too."""
    from pybnlearn.mvnorm import MultivariateNormal

    data = datasets["gaussian.test"]
    fitted = pybnlearn.fit(pybnlearn.hc(data), data)

    assert isinstance(pybnlearn.query(fitted, "A"), MultivariateNormal)


def test_mixed_networks_are_reported_as_unsupported(datasets):
    """Neither path handles a network that is part discrete and part
    continuous, and saying so beats reaching one of them by accident."""
    data = datasets["clgaussian.test"]
    fitted = pybnlearn.fit(pybnlearn.hc(data), data)

    with pytest.raises(NotImplementedError, match="discrete"):
        pybnlearn.query(fitted, "A")
