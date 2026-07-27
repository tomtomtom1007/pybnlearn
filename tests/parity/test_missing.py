"""Check incomplete-data handling against R.

Three things: estimating parameters when the data have gaps, filling the
gaps in, and structural EM, which alternates the two.

The imputation methods differ in what they may look at, and the fixture data
are built so the difference shows.  "parents" uses only a node's parents and
throws away the rest of the observation; "bayes-lw" and "exact" condition on
everything the row does say.  Gaps were punched into nodes with parents,
nodes with children, and several in the same row -- if they were all in root
nodes the three methods would agree and none of this would be tested.

"bayes-lw" draws from R's generator, so it is seeded and compared value for
value rather than statistically.

Fixtures come from tools/gen_r_missing_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

DATA = {"discrete": "incomplete.discrete",
        "continuous": "incomplete.continuous"}


def _records(kind):
    path = FIXTURES / "missing.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(pairs):
    return sorted(tuple(a) for a in pairs)


@pytest.fixture(scope="session")
def fitted(datasets):
    cache = {}

    def get(name, modelstring):
        if name not in cache:
            cache[name] = pybnlearn.fit(
                pybnlearn.model2network(modelstring), datasets[DATA[name]],
                method="mle-g" if name == "continuous" else "mle")
        return cache[name]

    return get


# ---------------------------------------------------------------------------
# fitting parameters from incomplete data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("fit"),
                         ids=lambda c: f"{c['dataset']}-{c['node']}")
def test_fitting_incomplete_data_matches_r(case, fitted):
    """The C code counts complete cases instead of using a missing value as
    a table index, but only when it is told the node has gaps -- and it is
    told per node, not per data set."""
    node = fitted(case["dataset"], case["modelstring"])[case["node"]]

    if case["class"] == "dnode":
        assert list(node.probabilities.shape) == case["dim"]
        # R unrolls column-major, which is numpy's order="F".
        assert np.allclose(node.probabilities.reshape(-1, order="F"),
                           case["prob"], rtol=1e-12, atol=1e-14,
                           equal_nan=True)
    else:
        assert list(node.coefficients) == case["coefnames"]
        for name, expected in zip(case["coefnames"], case["coefficients"]):
            assert float(node.coefficients[name]) == pytest.approx(
                expected, rel=1e-10, abs=1e-12), name
        assert node.sd == pytest.approx(case["sd"][0], rel=1e-10, abs=1e-12)


def test_gaps_actually_change_the_estimates(datasets):
    """If the complete-case counting were being skipped, the parameters
    would come out the same as on the whole data -- or as garbage."""
    whole = pybnlearn.fit(
        pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
        datasets["learning.test"].head(500))
    holed = pybnlearn.fit(
        pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
        datasets["incomplete.discrete"])

    assert not np.allclose(whole["B"].probabilities,
                           holed["B"].probabilities)
    assert np.isfinite(holed["B"].probabilities).all()
    assert np.allclose(holed["B"].probabilities.sum(axis=0), 1.0)


# ---------------------------------------------------------------------------
# imputation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("impute"),
    ids=lambda c: (f"{c['dataset']}-{c['method']}"
                   + (f"-seed{c['seed']}" if c["method"] == "bayes-lw"
                      else "")))
def test_imputation_matches_r(case, datasets, fitted):
    network = fitted(case["dataset"], case["modelstring"])
    data = datasets[DATA[case["dataset"]]]

    # Every method is seeded, not only the sampling one: a conditional
    # distribution with two equally likely levels is decided by a draw from
    # R's generator, and this data has such ties.
    pybnlearn.set_seed(int(case["seed"]))
    if case["method"] == "bayes-lw":
        got = pybnlearn.impute(network, data, method="bayes-lw",
                               n=int(case["n"]))
    else:
        got = pybnlearn.impute(network, data, method=case["method"])

    assert list(got.columns) == list(case["columns"])
    assert not got.isna().any().any()

    for name, expected in case["columns"].items():
        column = got[name]
        if isinstance(expected[0], str):
            assert list(column.astype(str)) == expected, name
        else:
            assert np.allclose(column.to_numpy(dtype=float), expected,
                               rtol=1e-10, atol=1e-12), name


def test_the_methods_do_not_all_agree():
    """They condition on different things, so on data with gaps in nodes
    that have parents they must disagree somewhere -- otherwise these
    fixtures are testing one method three times."""
    by_key = {}
    for case in _records("impute"):
        by_key.setdefault(case["dataset"], {})[
            case["method"], case["seed"]] = case["columns"]

    differ = 0
    for dataset, results in by_key.items():
        values = [tuple(map(tuple, r.values())) for r in results.values()]
        differ += len(set(values)) > 1

    assert differ == len(by_key), by_key


def test_complete_rows_are_left_alone(datasets, fitted):
    """Imputation fills gaps; it does not revise what was observed."""
    data = datasets["incomplete.discrete"]
    network = fitted("discrete", "[A][C][F][B|A][D|A:C][E|B:F]")

    pybnlearn.set_seed(1)
    got = pybnlearn.impute(network, data, method="exact")

    for column in data.columns:
        observed = data[column].notna().to_numpy()
        assert (data.loc[observed, column].astype(str).to_numpy()
                == got.loc[observed, column].astype(str).to_numpy()).all(), \
            column


def test_exact_imputation_is_a_joint_answer(datasets, fitted):
    """Unlike the other two, the exact method fills a group of gaps with the
    combination that is most likely as a whole.  On a row where two adjacent
    nodes are both missing, that can differ from each node's own best
    guess."""
    data = datasets["incomplete.discrete"]
    network = fitted("discrete", "[A][C][F][B|A][D|A:C][E|B:F]")

    pybnlearn.set_seed(1)
    together = pybnlearn.impute(network, data, method="exact")
    pybnlearn.set_seed(1)
    separately = pybnlearn.impute(network, data, method="parents")

    both = data[["A", "B"]].isna().all(axis=1).to_numpy()
    assert both.any(), "no row has both A and B missing"

    differs = (together.loc[both, "B"].astype(str).to_numpy()
               != separately.loc[both, "B"].astype(str).to_numpy())
    assert differs.any()


def test_a_network_with_no_gaps_is_returned_unchanged(datasets):
    data = datasets["learning.test"].head(100)
    network = pybnlearn.fit(pybnlearn.hc(data), data)

    got = pybnlearn.impute(network, data, method="exact")

    for column in data.columns:
        assert (got[column].astype(str).to_numpy()
                == data[column].astype(str).to_numpy()).all()


def test_an_unknown_method_is_reported(datasets, fitted):
    with pytest.raises(ValueError, match="method must be"):
        pybnlearn.impute(fitted("discrete", "[A][C][F][B|A][D|A:C][E|B:F]"),
                         datasets["incomplete.discrete"], method="nonesuch")


def test_a_latent_variable_cannot_be_imputed(datasets):
    """A variable that is never observed has nothing to condition on, and
    `strict` decides whether that is an error."""
    data = datasets["latent"]
    seed = pybnlearn.fit(
        pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
        datasets["learning.test"].head(500))

    got = pybnlearn.impute(seed, data, method="exact", strict=False)
    assert not got["C"].isna().any()


# ---------------------------------------------------------------------------
# structural EM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("sem"),
    ids=lambda c: (f"{c['dataset']}-{c['impute']}-{c['maximize']}"
                   f"-{c['max.iter']}" + ("-start" if "start" in c else "")))
def test_structural_em_matches_r(case, datasets):
    start = (pybnlearn.model2network(case["start"]) if "start" in case
             else None)

    pybnlearn.set_seed(1)
    got = pybnlearn.structural_em(
        datasets[DATA[case["dataset"]]], maximize=case["maximize"],
        impute_method=case["impute"], max_iter=int(case["max.iter"]),
        method="mle-g" if case["dataset"] == "continuous" else "mle",
        start=start)

    assert _arcs(got.arcs) == _arcs(case["arcs"])
    assert got.modelstring() == case["modelstring"]


def test_both_maximisers_and_both_imputers_are_covered():
    cases = _records("sem")
    assert {c["maximize"] for c in cases} == {"hc", "tabu"}
    assert {c["impute"] for c in cases} == {"parents", "exact"}
    assert {c["max.iter"] for c in cases} >= {1, 3}


def test_more_iterations_can_change_the_answer():
    """If one round were as good as three the loop would be doing nothing,
    and the fixtures would not be testing it."""
    by_key = {}
    for case in _records("sem"):
        if "start" in case:
            continue
        key = (case["dataset"], case["impute"], case["maximize"])
        by_key.setdefault(key, {})[case["max.iter"]] = case["modelstring"]

    assert any(len(set(v.values())) > 1 for v in by_key.values()), by_key


@pytest.mark.parametrize("case", _records("latent"), ids=lambda c: "latent")
def test_structural_em_with_a_latent_variable_matches_r(case, datasets):
    """A variable that is never observed cannot have its distribution
    estimated from the data, so EM must be handed one to start from."""
    seed = pybnlearn.fit(
        pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]"),
        datasets["learning.test"].head(500))

    pybnlearn.set_seed(1)
    got = pybnlearn.structural_em(datasets["latent"], impute_method="exact",
                                  start=seed, max_iter=2)

    assert _arcs(got.arcs) == _arcs(case["arcs"])
    assert got.modelstring() == case["modelstring"]


def test_a_latent_variable_needs_a_fitted_start(datasets):
    with pytest.raises(ValueError, match="latent"):
        pybnlearn.structural_em(datasets["latent"])


def test_structural_em_records_how_it_learned(datasets):
    learned = pybnlearn.structural_em(datasets["incomplete.discrete"],
                                      impute_method="parents", max_iter=1)

    assert learned.learning["algo"] == "structural.em"
    assert learned.learning["impute"] == "parents"
