"""Check Gaussian exact inference against R.

Unlike the discrete case in test_exact.py, this is a port and the claim is
the strong one: bnlearn implements the multivariate normal machinery itself,
so agreeing with R means reproducing its choices, not merely getting a right
answer.  Two of those choices show up here.  Conditioning goes through a
pseudoinverse with a hand-rolled cutoff rather than a solve, and factorising
a joint distribution back into a network patches the diagonal of a singular
covariance matrix rather than failing -- which is what lets a network with a
deterministic node round-trip at all.

Fixtures come from tools/gen_r_mvnorm_fixtures.R.

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
    path = FIXTURES / "mvnorm.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


@pytest.fixture(scope="session")
def networks(datasets):
    cache = {}

    def get(dataset, modelstring):
        key = (dataset, modelstring)
        if key not in cache:
            cache[key] = pybnlearn.fit(
                pybnlearn.model2network(modelstring), datasets[dataset],
                method="mle-g")
        return cache[key]

    return get


@pytest.mark.parametrize(
    "case", _records("mvnorm"),
    ids=lambda c: f"{c['dataset']}-{c['modelstring'][:24]}")
def test_global_distribution_matches_r(case, networks):
    mvn = pybnlearn.gbn2mvnorm(networks(case["dataset"], case["modelstring"]))

    assert mvn.variables == case["variables"]
    assert np.allclose(mvn.mean, case["mu"], rtol=1e-12, atol=1e-14)

    n = len(case["variables"])
    expected = np.asarray(case["sigma"]).reshape((n, n), order="F")
    assert np.allclose(mvn.cov, expected, rtol=1e-12, atol=1e-14)

    # a covariance matrix, whatever else it is
    assert np.allclose(mvn.cov, mvn.cov.T)
    assert (np.linalg.eigvalsh(mvn.cov) > -1e-10).all()


@pytest.mark.parametrize(
    "case", _records("mvnorm2gbn"),
    ids=lambda c: f"{c['dataset']}-{c['node']}-{c['modelstring'][:20]}")
def test_factorisation_matches_r(case, networks):
    fitted = networks(case["dataset"], case["modelstring"])
    mvn = pybnlearn.gbn2mvnorm(fitted)

    back = pybnlearn.mvnorm2gbn(
        pybnlearn.model2network(case["modelstring"]), mvn.mean, mvn.cov,
        nodes=mvn.variables)
    node = back[case["node"]]

    assert list(node.parents) == case["parents"]
    assert list(node.coefficients) == case["coefnames"]
    for name, expected in zip(case["coefnames"], case["coefficients"]):
        assert float(node.coefficients[name]) == pytest.approx(
            expected, rel=1e-10, abs=1e-12), name
    assert node.sd == pytest.approx(case["sd"][0], rel=1e-10, abs=1e-12)


def test_the_round_trip_returns_the_network_it_started_from(networks):
    """gbn2mvnorm and mvnorm2gbn are inverses, and composing them is a check
    on both that does not depend on either fixture."""
    fitted = networks("gaussian.test", "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")
    mvn = pybnlearn.gbn2mvnorm(fitted)

    back = pybnlearn.mvnorm2gbn(
        pybnlearn.model2network("[A][B][E][G][C|A:B][D|B][F|A:D:E:G]"),
        mvn.mean, mvn.cov, nodes=mvn.variables)

    for name in fitted.nodes:
        original, recovered = fitted[name], back[name]
        assert original.sd == pytest.approx(recovered.sd, rel=1e-8, abs=1e-10)
        for coefficient in original.coefficients:
            assert float(original.coefficients[coefficient]) == pytest.approx(
                float(recovered.coefficients[coefficient]), rel=1e-8,
                abs=1e-10), f"{name}${coefficient}"


@pytest.mark.parametrize(
    "case", _records("conditional"),
    ids=lambda c: (f"{c['dataset']}-{'.'.join(c['to'])}"
                   f"-given-{'.'.join(c['from'])}"))
def test_conditional_mean_matches_r(case, networks):
    fitted = networks(case["dataset"], case["modelstring"])
    evidence = dict(zip(case["from"], case["value"]))

    result = pybnlearn.query(fitted, case["to"], evidence)

    assert result.variables == case["to"]
    assert np.allclose(result.mean, case["mean"], rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# the conditional covariance, which bnlearn does not compute
# ---------------------------------------------------------------------------

def test_the_conditional_covariance_agrees_with_the_direct_formula(networks):
    """conditional.mvnorm() returns only the mean, so the covariance half of
    query() has nothing in R to compare against.  It is checked against the
    textbook formula instead, computed with a plain solve rather than the
    pseudoinverse the implementation uses."""
    fitted = networks("gaussian.test", "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")
    mvn = pybnlearn.gbn2mvnorm(fitted)

    to, given = ["C", "F"], ["A", "B", "D"]
    index = {v: i for i, v in enumerate(mvn.variables)}
    rows = [index[v] for v in to]
    columns = [index[v] for v in given]

    joint = mvn.cov[np.ix_(rows, rows)]
    cross = mvn.cov[np.ix_(rows, columns)]
    inner = mvn.cov[np.ix_(columns, columns)]
    expected = joint - cross @ np.linalg.solve(inner, cross.T)

    got = pybnlearn.query(fitted, to, {v: 1.0 for v in given})

    assert np.allclose(got.cov, expected, rtol=1e-8, atol=1e-10)


def test_the_conditional_distribution_agrees_with_sampling(networks):
    """cpdist samples where query() computes.  Conditioning on an interval
    is not the same as conditioning on a point, so the interval is made
    narrow and the tolerance is loose; this is a sanity check on the whole
    path, not a precision one."""
    fitted = networks("gaussian.test", "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")

    exact = pybnlearn.query(fitted, ["F"], {"B": 2.0})

    def near_two(frame):
        return (frame["B"] > 1.98) & (frame["B"] < 2.02)
    near_two.nodes = ["B"]

    pybnlearn.set_seed(1)
    sampled = pybnlearn.cpdist(fitted, ["F"], near_two, method="ls",
                               n=2000000)

    assert sampled["F"].mean() == pytest.approx(float(exact.mean[0]), abs=0.05)
    assert sampled["F"].std(ddof=1) == pytest.approx(
        float(np.sqrt(exact.cov[0, 0])), rel=0.02)


def test_conditioning_on_nothing_is_the_marginal(networks):
    fitted = networks("marks",
                      "[MECH][VECT|MECH][ALG|MECH:VECT][ANL|ALG][STAT|ALG:ANL]")
    mvn = pybnlearn.gbn2mvnorm(fitted)

    got = pybnlearn.query(fitted, ["ALG", "STAT"])
    index = [mvn.variables.index(v) for v in ("ALG", "STAT")]

    assert np.allclose(got.mean, mvn.mean[index])
    assert np.allclose(got.cov, mvn.cov[np.ix_(index, index)])


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("predict"),
    ids=lambda c: (f"{c['dataset']}-{c['node']}-{c['modelstring'][:20]}"
                   + ("-from" if c.get("from") else "")))
def test_exact_gaussian_prediction_matches_r(case, datasets, networks):
    fitted = networks(case["dataset"], case["modelstring"])
    data = datasets[case["dataset"]].head(int(case["n"]))

    got = pybnlearn.predict(fitted, case["node"], data, method="exact",
                            predictors=case.get("from"))

    assert np.allclose(np.asarray(got, dtype=float), case["values"],
                       rtol=1e-10, atol=1e-12, equal_nan=True)


def test_prediction_from_nothing_is_the_marginal_mean(networks):
    """A node with no predictors has no evidence to condition on, so every
    row gets the same answer: the marginal expectation."""
    import pandas as pd

    fitted = networks("gaussian.test", "[A][B][C][D][E][F][G]")
    mvn = pybnlearn.gbn2mvnorm(fitted)

    got = pybnlearn.predict(fitted, "A", pd.DataFrame(index=range(4)),
                            method="exact", predictors=[])

    assert np.allclose(np.asarray(got, dtype=float),
                       mvn.mean[mvn.variables.index("A")], rtol=1e-10)


def test_exact_prediction_is_the_conditional_expectation(networks, datasets):
    """The two entry points have to agree: predicting a node row by row is
    querying it with that row as evidence."""
    fitted = networks("gaussian.test", "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")
    data = datasets["gaussian.test"].head(5)
    predictors = [c for c in data.columns if c != "F"]

    got = np.asarray(pybnlearn.predict(fitted, "F", data, method="exact"),
                     dtype=float)

    for position in range(len(data)):
        evidence = {name: float(data.iloc[position][name])
                    for name in predictors}
        expected = pybnlearn.query(fitted, "F", evidence)
        assert got[position] == pytest.approx(float(expected.mean[0]),
                                              rel=1e-8, abs=1e-10)


# ---------------------------------------------------------------------------
# singular covariance matrices
# ---------------------------------------------------------------------------

def _singular():
    case = _records("singular")
    return case[0] if case else None


@pytest.mark.skipif(_singular() is None, reason="fixtures not generated")
def test_a_deterministic_node_factorises_like_r():
    """A node with zero residual variance makes the joint covariance matrix
    singular.  R patches the diagonal, factorises, and undoes the patch; a
    Cholesky decomposition of the matrix as it stands would fail."""
    case = _singular()
    n = len(case["variables"])
    cov = np.asarray(case["sigma"]).reshape((n, n), order="F")

    assert np.linalg.matrix_rank(cov) < n, "the fixture is not singular"

    back = pybnlearn.mvnorm2gbn(
        pybnlearn.model2network(case["modelstring"]), case["mu"], cov,
        nodes=case["variables"])

    for expected in _records("singular.gbn"):
        node = back[expected["node"]]
        assert list(node.coefficients) == expected["coefnames"]
        for name, value in zip(expected["coefnames"],
                               expected["coefficients"]):
            assert float(node.coefficients[name]) == pytest.approx(
                value, rel=1e-10, abs=1e-12), f"{expected['node']}${name}"
        assert node.sd == pytest.approx(expected["sd"][0], rel=1e-10,
                                        abs=1e-12)

    # and the deterministic node comes back deterministic.
    assert back["X"].sd == 0.0


def test_a_deterministic_root_survives_conditioning():
    """The pseudoinverse exists so that evidence on a variable with zero
    variance does not divide by zero."""
    case = _singular()
    n = len(case["variables"])
    cov = np.asarray(case["sigma"]).reshape((n, n), order="F")

    from pybnlearn.mvnorm import MultivariateNormal

    mvn = MultivariateNormal(case["variables"], case["mu"], cov)
    conditioned = mvn.condition({"X": 4.0})

    assert conditioned.variables == ["Y", "Z"]
    # X is constant at its mean, so conditioning on that value tells us
    # nothing: the rest of the distribution is unchanged.
    index = [case["variables"].index(v) for v in ("Y", "Z")]
    assert np.allclose(conditioned.mean, np.asarray(case["mu"])[index])
    assert np.allclose(conditioned.cov, cov[np.ix_(index, index)])


# ---------------------------------------------------------------------------
# argument checking
# ---------------------------------------------------------------------------

def test_a_discrete_network_is_not_a_multivariate_normal(datasets):
    fitted = pybnlearn.fit(pybnlearn.hc(datasets["learning.test"]),
                           datasets["learning.test"])

    with pytest.raises(ValueError, match="Gaussian"):
        pybnlearn.gbn2mvnorm(fitted)


def test_a_node_cannot_be_both_queried_and_given(networks):
    fitted = networks("gaussian.test", "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")

    with pytest.raises(ValueError, match="already given as evidence"):
        pybnlearn.query(fitted, ["A", "F"], {"A": 1.0})


def test_unknown_nodes_are_reported(networks):
    fitted = networks("gaussian.test", "[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")

    with pytest.raises(ValueError, match="unknown node"):
        pybnlearn.query(fitted, "F", {"nonesuch": 1.0})

    with pytest.raises(ValueError, match="unknown node"):
        pybnlearn.query(fitted, "nonesuch")
