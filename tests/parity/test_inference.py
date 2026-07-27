"""Check simulation and approximate inference against R.

These comparisons are exact, not statistical.  Both implementations run the
same C sampler over R's own Mersenne-Twister, so for a given seed they must
draw the same numbers in the same order: a Monte Carlo estimate that differs
at all means the sampling diverged, not that the estimate was noisy.

That also makes the tests sensitive to something easy to get wrong -- bnlearn
reduces the network to the upper closure of the query before sampling, and
sampling a different set of nodes consumes the generator differently, so a
port that skipped the reduction would produce plausible but different numbers.

Fixtures come from tools/gen_r_inference_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# the structures the fixtures were fitted with
STRUCTURES = {
    "learning.test": "[A][C][F][B|A][D|A:C][E|B:F]",
    "asia": "[A][S][T|A][L|S][B|S][E|T:L][X|E][D|B:E]",
    "gaussian.test": "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]",
}


@pytest.fixture(scope="session")
def networks(datasets):
    fitted = {}
    for name, modelstring in STRUCTURES.items():
        data = datasets[name]
        method = "mle-g" if name == "gaussian.test" else "mle"
        fitted[name] = pybnlearn.fit(
            pybnlearn.model2network(modelstring), data, method=method)
    return fitted


def _records(kind):
    path = FIXTURES / "inference.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


@pytest.mark.parametrize(
    "case", _records("rbn"),
    ids=lambda c: f"{c['network']}-seed{c['seed']:g}-n{c['n']:g}")
def test_rbn_reproduces_r(case, networks):
    pybnlearn.set_seed(int(case["seed"]))
    sample = pybnlearn.rbn(networks[case["network"]], int(case["n"]))

    assert len(sample) == int(case["n"])
    assert set(sample.columns) == set(case["columns"])

    for column, expected in case["columns"].items():
        got = sample[column]
        if isinstance(expected[0], str):
            assert list(got.astype(str)) == expected, f"column {column}"
        else:
            assert np.allclose(got.to_numpy(), expected, rtol=1e-12,
                               atol=1e-12), f"column {column}"


@pytest.mark.parametrize(
    "case", _records("cpquery"),
    ids=lambda c: (f"{c['network']}-{c['method']}-seed{c['seed']:g}-"
                   f"{sorted(c['event'])}-{sorted(c['evidence'] or {})}"))
def test_cpquery_reproduces_r(case, networks):
    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.cpquery(
        networks[case["network"]],
        event=case["event"],
        evidence=case["evidence"],
        method=case["method"],
        n=int(case["n"]))

    assert got == pytest.approx(case["probability"], rel=1e-11, abs=1e-12)


@pytest.mark.parametrize(
    "case", _records("cpdist"),
    ids=lambda c: f"{c['network']}-{c['method']}-seed{c['seed']:g}")
def test_cpdist_reproduces_r(case, networks):
    pybnlearn.set_seed(int(case["seed"]))
    sample, weights = pybnlearn.cpdist(
        networks[case["network"]], nodes=case["nodes"],
        evidence=case["evidence"], method=case["method"], n=int(case["n"]))

    assert list(sample["B"].astype(str)) == case["B"]
    assert list(sample["E"].astype(str)) == case["E"]
    assert np.allclose(weights, case["weights"], rtol=1e-12, atol=1e-12)


def test_the_suite_covers_both_sampling_methods():
    methods = {c["method"] for c in _records("cpquery")}
    assert methods == {"ls", "lw"}, (
        "the query fixtures must exercise both logic sampling and likelihood "
        "weighting; they take entirely different paths")


def test_seeding_is_reproducible(networks):
    pybnlearn.set_seed(3)
    first = pybnlearn.rbn(networks["learning.test"], 20)
    pybnlearn.set_seed(3)
    second = pybnlearn.rbn(networks["learning.test"], 20)

    assert first.equals(second)


def test_unseeded_runs_differ(networks):
    """A guard against the generator being accidentally reset each call."""
    pybnlearn.set_seed(11)
    first = pybnlearn.rbn(networks["learning.test"], 50)
    second = pybnlearn.rbn(networks["learning.test"], 50)

    assert not first.equals(second)


def test_likelihood_weighting_rejects_non_dict_evidence(networks):
    with pytest.raises(ValueError, match="dict"):
        pybnlearn.cpquery(networks["learning.test"], {"B": "a"},
                          lambda frame: frame["A"] == "a", method="lw")


def test_callable_evidence_needs_its_nodes_declared(networks):
    def predicate(frame):
        return (frame["A"] == "a").to_numpy()

    with pytest.raises(ValueError, match="nodes"):
        pybnlearn.cpquery(networks["learning.test"], {"B": "a"}, predicate,
                          method="ls", n=100)


def test_callable_evidence_with_nodes_works(networks):
    def predicate(frame):
        return (frame["A"] == "a").to_numpy()

    predicate.nodes = ["A"]

    pybnlearn.set_seed(1)
    got = pybnlearn.cpquery(networks["learning.test"], {"B": "a"}, predicate,
                            method="ls", n=10000)

    # the same query written as a dict must give the same answer, since it
    # reduces the network the same way and draws the same numbers.
    pybnlearn.set_seed(1)
    expected = pybnlearn.cpquery(networks["learning.test"], {"B": "a"},
                                 {"A": "a"}, method="ls", n=10000)

    assert got == pytest.approx(expected)
