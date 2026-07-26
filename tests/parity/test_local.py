"""Check fast.iamb, learn.mb/learn.nbr, entropy and KL divergence against R.

fast.iamb is the one with a real chance of diverging.  It admits every
candidate that looks associated in a single pass rather than one at a time,
and stops speculating when the contingency table would get too sparse for an
asymptotic test to mean anything -- a rule that fires on some data sets and
not others, so both are here.  The arc sets are compared including direction.

H() and KL() over discrete networks need the joint distribution of every
node's parents.  R gets that from gRain; pybnlearn computes it with its own
junction tree, so these cases compare two independent inference engines and
not just two ways of adding logarithms up.

Fixtures come from tools/gen_r_local_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import math
import pathlib

import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

CONTINUOUS = {"gaussian.test", "marks"}


def _records(kind):
    path = FIXTURES / "local.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(pairs):
    return sorted(tuple(a) for a in pairs)


def _number(value):
    if isinstance(value, str):
        return {"Inf": math.inf, "-Inf": -math.inf, "NaN": math.nan}[value]
    return value


# ---------------------------------------------------------------------------
# fast.iamb
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("fast.iamb"),
    ids=lambda c: (f"{c['dataset']}-{c['test'] or 'default'}-{c['alpha']:g}"
                   + ("-wl" if "whitelist" in c else "")
                   + ("-undirected" if "undirected" in c else "")))
def test_fast_iamb_matches_r(case, datasets):
    got = pybnlearn.fast_iamb(
        datasets[case["dataset"]], test=case["test"],
        alpha=float(case["alpha"]),
        whitelist=[tuple(a) for a in case.get("whitelist", [])] or None,
        undirected=case.get("undirected", False))

    assert _arcs(got.arcs) == _arcs(case["arcs"])


def test_fast_iamb_covers_both_data_types():
    datasets_seen = {c["dataset"] for c in _records("fast.iamb")}
    assert datasets_seen & CONTINUOUS
    assert datasets_seen - CONTINUOUS


def test_the_sparsity_guard_is_reachable(datasets):
    """fast.iamb stops admitting nodes once the contingency table would be
    too thin for an asymptotic test, and that rule is the only thing
    separating it from iamb.

    On bnlearn's own data sets it never fires -- the two algorithms agree
    everywhere, which is what they are designed to do -- so the fixtures
    include a deliberately sparse data set where it does.
    """
    from pybnlearn.constraint import _level_counts, _observations_per_cell

    data = datasets["sparse"]
    counts = _level_counts(data)

    # 60 rows over four levels each: thin to begin with and thinner with
    # every node conditioned on, which is exactly the situation the guard
    # exists for.
    unconditioned = _observations_per_cell(counts, len(data), "V1", "V2", [])
    conditioned = _observations_per_cell(counts, len(data), "V1", "V2",
                                         ["V3", "V4"])

    assert conditioned < 5 < _observations_per_cell(
        _level_counts(datasets["learning.test"]),
        len(datasets["learning.test"]), "A", "B", ["C"])
    assert conditioned < unconditioned

    # and continuous data has no cells at all, which is how the guard is
    # kept from firing on it.
    assert _observations_per_cell(None, 100, "A", "B", []) == float("inf")


def test_fast_iamb_agrees_with_iamb_where_the_guard_does_not_fire(datasets):
    """Not a tautology: they are different algorithms, and this is the claim
    that fast.iamb is a faster route to the same answer rather than a
    different one."""
    for name in ("learning.test", "asia", "lizards"):
        data = datasets[name]
        assert (_arcs(pybnlearn.fast_iamb(data).arcs)
                == _arcs(pybnlearn.iamb(data).arcs)), name


# ---------------------------------------------------------------------------
# local structure learning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("mb"),
    ids=lambda c: f"{c['dataset']}-{c['node']}-{c['method']}")
def test_learn_mb_matches_r(case, datasets):
    got = pybnlearn.learn_mb(datasets[case["dataset"]], case["node"],
                             method=case["method"])

    assert sorted(got) == sorted(case["result"])


@pytest.mark.parametrize(
    "case", _records("nbr"),
    ids=lambda c: f"{c['dataset']}-{c['node']}-{c['method']}")
def test_learn_nbr_matches_r(case, datasets):
    got = pybnlearn.learn_nbr(datasets[case["dataset"]], case["node"],
                              method=case["method"])

    assert sorted(got) == sorted(case["result"])


def test_the_blanket_contains_the_neighbourhood(datasets):
    """A node's neighbours are its parents and children; its blanket adds
    the spouses.  So one is always a subset of the other, whichever pair of
    algorithms found them."""
    data = datasets["learning.test"]

    for node in data.columns:
        blanket = set(pybnlearn.learn_mb(data, node))
        neighbours = set(pybnlearn.learn_nbr(data, node))
        assert neighbours <= blanket, node


def test_a_local_result_agrees_with_the_whole_network(datasets):
    """Learning one node's blanket and learning the whole structure are
    different code paths for the same question."""
    data = datasets["learning.test"]
    whole = pybnlearn.gs(data)

    for node in data.columns:
        assert (sorted(pybnlearn.learn_mb(data, node, "gs"))
                == sorted(pybnlearn.mb(whole, node))), node


def test_the_wrong_kind_of_algorithm_is_reported(datasets):
    data = datasets["learning.test"]

    with pytest.raises(ValueError, match="Markov blankets"):
        pybnlearn.learn_mb(data, "A", method="mmpc")

    with pytest.raises(ValueError, match="neighbourhoods"):
        pybnlearn.learn_nbr(data, "A", method="gs")


# ---------------------------------------------------------------------------
# entropy and divergence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted(datasets):
    cache = {}

    def get(dataset, modelstring):
        key = (dataset, modelstring)
        if key not in cache:
            data = datasets[dataset]
            cache[key] = pybnlearn.fit(
                pybnlearn.model2network(modelstring), data,
                method="mle-g" if dataset in CONTINUOUS else "mle")
        return cache[key]

    return get


@pytest.mark.parametrize(
    "case", _records("entropy"),
    ids=lambda c: f"{c['dataset']}-{c['modelstring'][:22]}")
def test_entropy_matches_r(case, fitted):
    got = pybnlearn.H(fitted(case["dataset"], case["modelstring"]))

    assert got == pytest.approx(_number(case["value"]), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize(
    "case", _records("kl"),
    ids=lambda c: f"{c['dataset']}-{c['p'][:16]}-vs-{c['q'][:16]}")
def test_kl_matches_r(case, fitted):
    P = fitted(case["dataset"], case["p"])
    Q = fitted(case["dataset"], case["q"])

    got = pybnlearn.KL(P, Q)
    expected = _number(case["value"])

    if isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(got)
    elif math.isinf(expected):
        assert got == expected
    else:
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-11)


def test_a_network_diverges_from_itself_by_nothing_or_by_nothing_defined():
    """Exactly zero, not nearly: R collapses the two log-likelihood terms
    when they agree, because subtracting them leaves rounding.

    The exception is a network with a deterministic node -- asia's E is a
    logic gate -- where a configuration of probability zero contributes
    0 * log 0.  That is undefined rather than zero, and R propagates the
    NaN rather than quietly calling it zero.
    """
    seen = set()
    for case in _records("kl"):
        if case["p"] != case["q"]:
            continue
        value = _number(case["value"])
        seen.add("nan" if math.isnan(value) else "zero")
        assert math.isnan(value) or value == 0.0

    assert seen == {"zero", "nan"}, seen


def test_divergence_is_not_symmetric():
    """The reason it is not a distance, and worth confirming the fixtures
    actually contain a pair that shows it."""
    by_pair = {(c["dataset"], c["p"], c["q"]): _number(c["value"])
               for c in _records("kl")}

    asymmetric = [k for k in by_pair
                  if (k[0], k[2], k[1]) in by_pair
                  and k[1] != k[2]
                  and by_pair[k] != by_pair[k[0], k[2], k[1]]]

    assert asymmetric


def test_an_impossible_configuration_makes_the_divergence_infinite():
    """If Q gives probability zero to something P does not, no amount of
    data can rescue Q, and the divergence says so."""
    infinite = [c for c in _records("kl") if c["value"] == "Inf"]
    assert infinite


def test_entropy_is_larger_for_a_less_informative_network(fitted):
    """A network with no arcs claims the variables are independent, which
    cannot describe the data as tightly -- so it has more entropy left."""
    learned = fitted("learning.test", "[A][C][F][B|A][D|A:C][E|B:F]")
    empty = fitted("learning.test", "[A][B][C][D][E][F]")

    assert pybnlearn.H(empty) > pybnlearn.H(learned)


def test_mixed_networks_are_reported_as_unsupported(datasets):
    data = datasets["clgaussian.test"]
    mixed = pybnlearn.fit(pybnlearn.hc(data), data)

    with pytest.raises(NotImplementedError, match="conditional Gaussian"):
        pybnlearn.H(mixed)
