"""Check conditional Gaussian networks against R.

Mixed data is where the three parameter backends meet, and the thing worth
testing is that each node reaches the right one: a factor node is discrete
however its parents look, a numeric node with only numeric parents is one
ordinary regression, and only a numeric node with a factor parent gets a
regression per configuration of it.  Sending a node to the wrong backend
still produces numbers, so the tests below check the node class as well as
the values.

Fixtures come from tools/gen_r_cg_fixtures.R.

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
    path = FIXTURES / "cg.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


@pytest.mark.parametrize(
    "case", _records("structure"),
    ids=lambda c: f"{c['dataset']}-{c['algorithm']}-{c['score'] or 'default'}")
def test_mixed_structure_learning_matches_r(case, datasets):
    data = datasets[case["dataset"]]
    algorithm = getattr(pybnlearn, case["algorithm"])

    learned = algorithm(data, score=case["score"])

    assert learned.modelstring() == case["modelstring"]
    assert pybnlearn.score(learned, data, type=case["score"]) == pytest.approx(
        case["value"], rel=1e-11, abs=1e-12)


@pytest.mark.parametrize(
    "case", _records("score"),
    ids=lambda c: f"{c['dataset']}-{c['score']}-{c['modelstring'][:20]}")
def test_conditional_gaussian_scores_match_r(case, datasets):
    network = pybnlearn.model2network(case["modelstring"])

    got = pybnlearn.score(network, datasets[case["dataset"]],
                          type=case["score"])

    assert got == pytest.approx(case["value"], rel=1e-11, abs=1e-12)


# ---------------------------------------------------------------------------
# parameter learning
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_networks(datasets):
    cache = {}

    def get(dataset, modelstring):
        key = (dataset, modelstring)
        if key not in cache:
            cache[key] = pybnlearn.fit(
                pybnlearn.model2network(modelstring), datasets[dataset],
                method="mle-cg")
        return cache[key]

    return get


@pytest.mark.parametrize(
    "case", _records("fit"),
    ids=lambda c: f"{c['dataset']}-{c['node']}-{c['modelstring'][:20]}")
def test_mixed_parameters_match_r(case, fitted_networks):
    fit = fitted_networks(case["dataset"], case["modelstring"])
    node = fit[case["node"]]

    assert isinstance(node, CLASSES[case["class"]]), case["class"]
    assert list(node.parents) == case["parents"]

    if case["class"] == "bn.fit.dnode":
        # R unrolls column-major, which is numpy's order="F".
        assert list(node.probabilities.shape) == case["dim"]
        assert np.allclose(node.probabilities.reshape(-1, order="F"),
                           case["prob"], rtol=1e-12, atol=1e-14,
                           equal_nan=True)
    elif case["class"] == "bn.fit.gnode":
        assert list(node.coefficients) == case["coefnames"]
        for name, expected in zip(case["coefnames"], case["coefficients"]):
            assert float(node.coefficients[name]) == pytest.approx(
                expected, rel=1e-11, abs=1e-12), name
        assert float(node.sd) == pytest.approx(case["sd"][0], rel=1e-12,
                                               abs=1e-14)
    else:
        assert node.discrete_parents == case["dparents"]
        assert node.continuous_parents == case["gparents"]
        assert node.coefficient_names == case["coefnames"]
        assert node.nconfigurations == case["nconfig"]
        assert np.allclose(node.coefficients.reshape(-1, order="F"),
                           case["coefficients"], rtol=1e-12, atol=1e-14)
        assert np.allclose(node.sd, case["sd"], rtol=1e-12, atol=1e-14)
        fitted = np.asarray(node.fitted_values)[:20]
        assert np.allclose(fitted, case["fitted"], rtol=1e-12, atol=1e-14)

        # A residual is the difference of two numbers much larger than it is,
        # so it can only be trusted to the precision of those numbers: on the
        # widest regression here the fitted values run to ~100, and the last
        # few bits of them are not reproducible anyway, because R's dqrdc2 is
        # compiled from Fortran and pybnlearn's is a C translation, which the
        # compilers are free to contract into fused multiply-adds at
        # different points.  The tolerance is therefore taken from the scale
        # of the response rather than from the residual itself.
        scale = max(np.abs(fitted).max(), 1.0)
        assert np.allclose(np.asarray(node.residuals)[:20], case["residuals"],
                           rtol=1e-12, atol=1e-12 * scale)


def test_all_three_backends_are_exercised():
    """If a change sent every node to one backend the value comparisons above
    would still pass on whichever nodes it happened to suit, so the mix
    itself is asserted."""
    classes = {c["class"] for c in _records("fit")}
    assert classes == set(CLASSES)


def test_a_conditional_gaussian_node_has_one_regression_per_configuration(
        datasets, fitted_networks):
    """The whole point of the backend: the coefficient matrix is as wide as
    the discrete parents' configurations are numerous."""
    fit = fitted_networks("clgaussian.test",
                          "[A][B][C][H][D|A:H][F|B:C][E|B:D][G|A:D:E:F]")
    node = fit["G"]

    expected = 1
    for parent in node.discrete_parents:
        expected *= len(node.discrete_levels[parent])

    assert node.nconfigurations == expected
    assert node.coefficients.shape == (1 + len(node.continuous_parents),
                                       expected)
    assert node.sd.shape == (expected,)


def test_the_regressions_reproduce_the_fitted_values(datasets,
                                                     fitted_networks):
    """Recompute what the C code returned, from the coefficients it returned,
    without going back through it."""
    data = datasets["clgaussian.test"]
    fit = fitted_networks("clgaussian.test",
                          "[A][B][C][H][D|A:H][F|B:C][E|B:D][G|A:D:E:F]")
    node = fit["G"]

    # the configuration index R assigns, which varies the first discrete
    # parent fastest.
    index = np.zeros(len(data), dtype=int)
    stride = 1
    for parent in node.discrete_parents:
        index += stride * data[parent].cat.codes.to_numpy()
        stride *= len(node.discrete_levels[parent])

    design = np.column_stack(
        [np.ones(len(data))]
        + [data[p].to_numpy(dtype=float) for p in node.continuous_parents])
    expected = np.einsum("ij,ji->i", design, node.coefficients[:, index])

    assert np.allclose(np.asarray(node.fitted_values), expected,
                       rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", _records("rbn"),
                         ids=lambda c: f"{c['dataset']}-seed{c['seed']}")
def test_sampling_from_a_mixed_network_matches_r(case, datasets,
                                                 fitted_networks):
    """The three node types are sampled by three different pieces of C, and a
    conditional Gaussian node's parameters reach it as positions among the
    node's parents rather than by name -- so an off-by-one there would show
    up here and nowhere in the parameter comparisons above."""
    fit = fitted_networks(case["dataset"], case["modelstring"])

    pybnlearn.set_seed(int(case["seed"]))
    generated = pybnlearn.rbn(fit, int(case["n"]))

    assert list(generated.columns) == list(case["columns"])

    for name, expected in case["columns"].items():
        got = generated[name]
        if isinstance(expected[0], str):
            assert list(got.astype(str)) == expected, name
        else:
            assert np.allclose(got.to_numpy(dtype=float), expected,
                               rtol=1e-10, atol=1e-12), name


def test_the_method_has_to_suit_the_data(datasets):
    with pytest.raises(ValueError, match="mixture"):
        pybnlearn.fit(pybnlearn.hc(datasets["learning.test"]),
                      datasets["learning.test"], method="mle-cg")

    with pytest.raises(ValueError, match="discrete data"):
        pybnlearn.fit(pybnlearn.model2network("[A][B][C][D][E][F][G][H]"),
                      datasets["clgaussian.test"], method="mle")


def test_the_score_has_to_suit_the_data(datasets):
    with pytest.raises(ValueError, match="mixture"):
        pybnlearn.hc(datasets["learning.test"], score="bic-cg")

    with pytest.raises(ValueError, match="continuous"):
        pybnlearn.hc(datasets["clgaussian.test"], score="bic-g")

    with pytest.raises(ValueError, match="discrete"):
        pybnlearn.hc(datasets["clgaussian.test"], score="bic")
