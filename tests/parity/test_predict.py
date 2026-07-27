"""Check predict() and the prediction-based cross-validation losses against R.

`bayes-lw` is a Monte Carlo estimate, so those cases are seeded and compared
exactly, like the rest of the sampling here: the same particles in the same
order or nothing matches.

Fixtures come from tools/gen_r_predict_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import math
import pathlib

import numpy as np
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

STRUCTURES = {
    "learning.test": "[A][C][F][B|A][D|A:C][E|B:F]",
    "asia": "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
    "gaussian.test": "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
}


@pytest.fixture(scope="session")
def networks(datasets):
    return {
        name: pybnlearn.fit(
            pybnlearn.model2network(modelstring), datasets[name],
            method="mle-g" if name == "gaussian.test" else "mle")
        for name, modelstring in STRUCTURES.items()
    }


def _records(kind):
    path = FIXTURES / "predict.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _number(value):
    if isinstance(value, str):
        return {"Inf": math.inf, "-Inf": -math.inf, "NaN": math.nan}[value]
    return value


@pytest.mark.parametrize(
    "case", _records("predict"),
    ids=lambda c: f"{c['network']}-{c['node']}-{c['method']}")
def test_predict_matches_r(case, datasets, networks):
    data = datasets[case["network"]].head(int(case["n"]))

    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.predict(networks[case["network"]], case["node"], data,
                            method=case["method"])

    expected = case["values"]
    if isinstance(expected[0], str):
        assert list(np.asarray(got).astype(str)) == expected
    else:
        assert np.allclose(np.asarray(got, dtype=float),
                           [_number(v) for v in expected],
                           rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    "case", _records("predict.prob"),
    ids=lambda c: f"{c['network']}-{c['node']}")
def test_prediction_probabilities_match_r(case, datasets, networks):
    data = datasets[case["network"]].head(int(case["n"]))

    pybnlearn.set_seed(11)
    _, probabilities = pybnlearn.predict(
        networks[case["network"]], case["node"], data, prob=True)

    assert list(probabilities.columns) == case["levels"]
    for level, expected in case["probabilities"].items():
        assert np.allclose(probabilities[level].to_numpy(),
                           [_number(v) for v in expected],
                           rtol=1e-12, atol=1e-12), f"level {level}"

    # every observation's distribution must still be a distribution.
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "case", _records("cvloss"),
    ids=lambda c: (f"{c['dataset']}-{c['loss']}-{c['target']}-"
                   f"{c['method']}-seed{c['seed']:g}"))
def test_predictive_losses_match_r(case, datasets):
    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.bn_cv(
        datasets[case["dataset"]], "hc", loss=case["loss"],
        target=case["target"], k=int(case["k"]), method=case["method"],
        m=int(case["m"]) if case["m"] is not None else None)

    expected = _number(case["mean"])
    if isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(got.mean)
    else:
        assert got.mean == pytest.approx(expected, rel=1e-11, abs=1e-12)


def test_all_five_predictive_losses_are_covered():
    losses = {c["loss"] for c in _records("cvloss")}
    assert losses == {"pred", "f1", "auroc", "cor", "mse"}


def test_both_aggregations_are_covered():
    """k-fold pools the folds' predictions and scores once; hold-out scores
    each repetition and averages.  They are different calculations."""
    methods = {c["method"] for c in _records("cvloss")}
    assert methods == {"k-fold", "hold-out"}


def test_a_target_is_required(datasets):
    with pytest.raises(ValueError, match="target"):
        pybnlearn.bn_cv(datasets["learning.test"], "hc", loss="pred", k=2)


def test_unknown_prediction_methods_are_reported(networks, datasets):
    with pytest.raises(ValueError, match="method must be"):
        pybnlearn.predict(networks["learning.test"], "B",
                          datasets["learning.test"].head(10),
                          method="nonesuch")


def test_exact_prediction_uses_the_posterior_it_reports(networks, datasets):
    """The predicted class must be the argmax of the probabilities returned
    alongside it, or one of the two is wrong."""
    data = datasets["learning.test"].head(50)
    predicted, probabilities = pybnlearn.predict(
        networks["learning.test"], "B", data, method="exact", prob=True)

    for i, value in enumerate(np.asarray(predicted).astype(str)):
        row = probabilities.iloc[i]
        assert row[value] == pytest.approx(row.max())


def test_probabilities_are_discrete_only(networks, datasets):
    with pytest.raises(ValueError, match="discrete"):
        pybnlearn.predict(networks["gaussian.test"], "B",
                          datasets["gaussian.test"].head(10), prob=True)


def test_parents_prediction_needs_the_parents(networks, datasets):
    data = datasets["learning.test"].head(10).drop(columns=["A"])

    with pytest.raises(ValueError, match="A"):
        pybnlearn.predict(networks["learning.test"], "B", data)
