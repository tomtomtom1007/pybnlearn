"""Check custom_fit() against R.

There is no data here, so nothing can be checked by refitting: what can go
wrong is bookkeeping.  Which axis of a table is the node and which are its
parents; which order the parents come in; which configuration of the
discrete parents a column of conditional Gaussian coefficients belongs to.
Get any of those wrong and you get a network that reads back correctly and
samples from a different distribution.

That is why the parameter comparisons below are followed by seeded sampling
and by exact inference.  Sampling walks the tables along one path and the
junction tree along another, and both only agree with R if every axis went
where it was meant to.

Fixtures come from tools/gen_r_custom_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pytest

import pybnlearn
from pybnlearn.fit import (ConditionalGaussianNode, DiscreteNode,
                           GaussianNode)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

CLASSES = {
    "bn.fit.dnode": DiscreteNode,
    "bn.fit.gnode": GaussianNode,
    "bn.fit.cgnode": ConditionalGaussianNode,
}


def _records(kind):
    path = FIXTURES / "custom.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


# The same three networks the generator builds, written out here rather than
# read from the fixtures: if they were read back, a mistake in how the
# fixtures encode a table would cancel out instead of showing up.
def _networks():
    discrete = pybnlearn.custom_fit(
        pybnlearn.model2network("[A][B|A][C|A:B]"),
        {"A": {"prob": np.array([0.3, 0.7]),
               "levels": {"A": ["a1", "a2"]}},
         "B": {"prob": np.array([0.2, 0.8, 0.6, 0.4]).reshape((2, 2),
                                                              order="F"),
               "levels": {"B": ["b1", "b2"]}},
         "C": {"prob": np.array([0.1, 0.4, 0.5, 0.5, 0.3, 0.2,
                                 0.2, 0.2, 0.6, 0.7, 0.2, 0.1]
                                ).reshape((3, 2, 2), order="F"),
               "levels": {"C": ["c1", "c2", "c3"]}}})

    gaussian = pybnlearn.custom_fit(
        pybnlearn.model2network("[X][W][Y|X:W][Z|Y]"),
        {"X": {"coef": {"(Intercept)": 4}, "sd": 0},
         "W": {"coef": {"(Intercept)": -1}, "sd": 2},
         "Y": {"coef": {"(Intercept)": 1, "X": 2, "W": 0.5}, "sd": 1.5},
         "Z": {"coef": {"(Intercept)": 0, "Y": -1}, "sd": 0.5}})

    cg = pybnlearn.custom_fit(
        pybnlearn.model2network("[D][E][F][G|D:F][H|D:E:G]"),
        {"D": {"prob": np.array([0.4, 0.6]), "levels": {"D": ["d1", "d2"]}},
         "E": {"prob": np.array([0.25, 0.35, 0.4]),
               "levels": {"E": ["e1", "e2", "e3"]}},
         "F": {"coef": {"(Intercept)": 2}, "sd": 1},
         "G": {"coef": np.array([1, 0.5, 3, -0.5]).reshape((2, 2), order="F"),
               "sd": [0.5, 1.5]},
         "H": {"coef": np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
                                ).reshape((2, 6), order="F"),
               "sd": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}})

    return {"discrete": discrete, "gaussian": gaussian, "cg": cg}


@pytest.fixture(scope="session")
def networks():
    return _networks()


# ---------------------------------------------------------------------------
# the parameters that came out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("node"),
                         ids=lambda c: f"{c['network']}-{c['node']}")
def test_custom_parameters_match_r(case, networks):
    node = networks[case["network"]][case["node"]]

    assert isinstance(node, CLASSES[case["class"]]), case["class"]
    assert list(node.parents) == case["parents"]
    assert list(node.children) == case["children"]

    if case["class"] == "bn.fit.dnode":
        assert list(node.probabilities.shape) == case["dim"]
        assert [list(l) for l in node.levels] == case["dimnames"]
        assert list(node.variables) == case["varnames"]
        # R unrolls column-major, which is numpy's order="F".
        assert np.allclose(node.probabilities.reshape(-1, order="F"),
                           case["prob"], rtol=1e-12, atol=1e-14)
    elif case["class"] == "bn.fit.gnode":
        assert list(node.coefficients) == case["coefnames"]
        for name, expected in zip(case["coefnames"], case["coefficients"]):
            assert float(node.coefficients[name]) == pytest.approx(expected)
        assert node.sd == pytest.approx(case["sd"][0])
    else:
        assert node.discrete_parents == case["dparents"]
        assert node.continuous_parents == case["gparents"]
        assert node.coefficient_names == case["coefnames"]
        assert node.nconfigurations == case["nconfig"]
        assert np.allclose(node.coefficients.reshape(-1, order="F"),
                           case["coefficients"], rtol=1e-12, atol=1e-14)
        assert np.allclose(node.sd, case["sd"], rtol=1e-12, atol=1e-14)
        assert node.discrete_levels == case["dlevels"]


def test_all_three_node_types_are_built():
    assert {c["class"] for c in _records("node")} == set(CLASSES)


@pytest.mark.parametrize("case", _records("bn.net"),
                         ids=lambda c: c["network"])
def test_bn_net_recovers_the_structure(case, networks):
    got = pybnlearn.bn_net(networks[case["network"]])

    assert got.nodes == case["nodes"]
    assert got.modelstring() == case["modelstring"]


# ---------------------------------------------------------------------------
# sampling, which is what pins the axis order down
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("rbn"),
                         ids=lambda c: f"{c['network']}-seed{c['seed']}")
def test_sampling_from_a_hand_built_network_matches_r(case, networks):
    """Exact, not statistical: pybnlearn draws from R's Mersenne-Twister, so
    the same seed gives the same 20 observations or the tables are wired up
    differently."""
    pybnlearn.set_seed(int(case["seed"]))
    generated = pybnlearn.rbn(networks[case["network"]], int(case["n"]))

    assert list(generated.columns) == list(case["columns"])

    for name, expected in case["columns"].items():
        got = generated[name]
        if isinstance(expected[0], str):
            assert list(got.astype(str)) == expected, name
        else:
            assert np.allclose(got.to_numpy(dtype=float), expected,
                               rtol=1e-12, atol=1e-14), name


@pytest.mark.parametrize("case", _records("query"),
                         ids=lambda c: (f"{c['network']}-{'.'.join(c['nodes'])}"
                                        + ("-given" if c["evidence"] else "")))
def test_exact_inference_on_a_hand_built_network_matches_r(case, networks):
    result = pybnlearn.query(networks[case["network"]], case["nodes"],
                             case["evidence"] or None)

    assert np.allclose(result.values.reshape(-1, order="F"), case["values"],
                       rtol=1e-10, atol=1e-12)


def test_a_deterministic_node_stays_deterministic(networks):
    """X has sd = 0, which fit() could never produce and which the sampler
    and the multivariate normal both have to survive."""
    fitted = networks["gaussian"]
    assert fitted["X"].sd == 0.0

    pybnlearn.set_seed(1)
    generated = pybnlearn.rbn(fitted, 50)
    assert (generated["X"] == 4.0).all()

    mvn = pybnlearn.gbn2mvnorm(fitted)
    assert mvn.cov[mvn.variables.index("X"), mvn.variables.index("X")] == 0.0


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------

def test_a_fitted_network_can_be_rebuilt_from_its_own_parameters(datasets):
    """custom_fit() has to accept what fit() produces, or it cannot be used
    to perturb a learned network -- which is most of what it is for."""
    data = datasets["learning.test"]
    learned = pybnlearn.hc(data)
    fitted = pybnlearn.fit(learned, data)

    rebuilt = pybnlearn.custom_fit(learned, {
        node: {"prob": fitted[node].probabilities,
               "levels": {node: fitted[node].levels[0]}}
        for node in learned.nodes})

    for node in learned.nodes:
        assert np.allclose(rebuilt[node].probabilities,
                           fitted[node].probabilities)
        assert rebuilt[node].variables == fitted[node].variables
        assert rebuilt[node].levels == fitted[node].levels

    # and the two sample identically, which is the check that matters.
    pybnlearn.set_seed(3)
    from_fit = pybnlearn.rbn(fitted, 100)
    pybnlearn.set_seed(3)
    from_custom = pybnlearn.rbn(rebuilt, 100)

    for node in learned.nodes:
        assert list(from_fit[node].astype(str)) == list(
            from_custom[node].astype(str)), node


def test_a_gaussian_network_round_trips(datasets):
    data = datasets["gaussian.test"]
    learned = pybnlearn.hc(data)
    fitted = pybnlearn.fit(learned, data, method="mle-g")

    rebuilt = pybnlearn.custom_fit(learned, {
        node: {"coef": fitted[node].coefficients, "sd": fitted[node].sd}
        for node in learned.nodes})

    original = pybnlearn.gbn2mvnorm(fitted)
    recovered = pybnlearn.gbn2mvnorm(rebuilt)

    assert np.allclose(original.mean, recovered.mean)
    assert np.allclose(original.cov, recovered.cov)


# ---------------------------------------------------------------------------
# what it refuses
# ---------------------------------------------------------------------------

def test_a_table_with_the_wrong_shape_is_reported():
    dag = pybnlearn.model2network("[A][B|A]")

    with pytest.raises(ValueError, match="shape"):
        pybnlearn.custom_fit(dag, {
            "A": {"prob": np.array([0.3, 0.7]), "levels": {"A": ["a", "b"]}},
            "B": {"prob": np.array([0.2, 0.8]), "levels": {"B": ["x", "y"]}}})


def test_a_table_that_does_not_sum_to_one_is_reported():
    dag = pybnlearn.model2network("[A]")

    with pytest.raises(ValueError, match="sum to one"):
        pybnlearn.custom_fit(dag, {
            "A": {"prob": np.array([0.3, 0.3]), "levels": {"A": ["a", "b"]}}})


def test_small_rounding_in_a_table_is_absorbed():
    """R rescales a distribution that is nearly one and rejects one that is
    not, on the grounds that a small gap is what you typed and a large one
    is a mistake."""
    dag = pybnlearn.model2network("[A]")

    fitted = pybnlearn.custom_fit(dag, {
        "A": {"prob": np.array([0.333, 0.333, 0.333]),
              "levels": {"A": ["a", "b", "c"]}}})

    assert fitted["A"].probabilities.sum() == pytest.approx(1.0)


def test_mismatched_node_types_are_reported():
    dag = pybnlearn.model2network("[A][B|A]")

    with pytest.raises(ValueError, match="discrete parent"):
        pybnlearn.custom_fit(dag, {
            "A": {"prob": np.array([0.3, 0.7]), "levels": {"A": ["a", "b"]}},
            "B": {"coef": {"(Intercept)": 1, "A": 2}, "sd": 1}})

    with pytest.raises(ValueError, match="continuous parent"):
        pybnlearn.custom_fit(dag, {
            "A": {"coef": {"(Intercept)": 1}, "sd": 1},
            "B": {"prob": np.array([[0.3, 0.7]]).T,
                  "levels": {"B": ["x", "y"]}}})


def test_the_wrong_number_of_coefficients_is_reported():
    dag = pybnlearn.model2network("[A][B|A]")

    with pytest.raises(ValueError, match="coefficients"):
        pybnlearn.custom_fit(dag, {
            "A": {"coef": {"(Intercept)": 1}, "sd": 1},
            "B": {"coef": {"(Intercept)": 1}, "sd": 1}})


def test_a_partially_directed_graph_cannot_be_fitted():
    net = pybnlearn.BayesianNetwork(["A", "B"], [("A", "B"), ("B", "A")])

    with pytest.raises(ValueError, match="partially directed"):
        pybnlearn.custom_fit(net, {"A": {"coef": {"(Intercept)": 1}, "sd": 1},
                                   "B": {"coef": {"(Intercept)": 1}, "sd": 1}})


def test_the_distributions_have_to_cover_the_nodes():
    dag = pybnlearn.model2network("[A][B|A]")

    with pytest.raises(ValueError, match="missing B"):
        pybnlearn.custom_fit(dag, {"A": {"coef": {"(Intercept)": 1}, "sd": 1}})
