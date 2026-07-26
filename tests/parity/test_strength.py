"""Check arc strengths and network averaging against R.

The threshold tests are the load-bearing ones.  Arc strengths are ordinary
arithmetic and would agree with any correct implementation, but the
inclusion threshold is the minimum of a piecewise-linear function located by
R's optimize(), and it is used as a cutoff: an implementation that finds a
minimum a few ulp away can put an arc on the other side of it and return a
different network.  So `_brent_fmin` is a port rather than a substitution,
and test_inclusion_threshold_matches_r is what says so.

Fixtures come from tools/gen_r_strength_fixtures.R.

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


def _records(kind):
    path = FIXTURES / "strength.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _by_arc(frame):
    return {(str(a), str(b)): float(s) for a, b, s
            in zip(frame["from"], frame["to"], frame["strength"])}


# ---------------------------------------------------------------------------
# arc.strength
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("arc.strength"),
    ids=lambda c: (f"{c['dataset']}-{c['criterion']}"
                   + ("-" + "-".join(f"{k}{v:g}" for k, v in c["args"].items())
                      if c["args"] else "")))
def test_arc_strength_matches_r(case, datasets):
    network = pybnlearn.model2network(case["modelstring"])

    got = pybnlearn.arc_strength(network, datasets[case["dataset"]],
                                 criterion=case["criterion"], **case["args"])

    assert got.attrs["method"] == case["method"]
    assert got.attrs["threshold"] == pytest.approx(case["threshold"])

    expected = dict(zip(zip(case["from"], case["to"]), case["strength"]))
    assert set(_by_arc(got)) == set(expected)

    for arc, value in _by_arc(got).items():
        # p-values run down to 1e-320, where a relative comparison is the
        # only meaningful one; score differences are ordinary numbers.
        assert value == pytest.approx(expected[arc], rel=1e-11, abs=1e-300), arc


@pytest.mark.parametrize(
    "case", _records("arc.strength.default"),
    ids=lambda c: f"{c['dataset']}-{c['algorithm']}")
def test_the_default_criterion_comes_from_the_network(case, datasets):
    """A learned network remembers what it was learned with, and
    arc.strength() reuses it -- so the same call means a score after hc() and
    a test after gs()."""
    data = datasets[case["dataset"]]
    learned = getattr(pybnlearn, case["algorithm"])(data)

    got = pybnlearn.arc_strength(learned, data)

    assert got.attrs["method"] == case["method"]
    expected = dict(zip(zip(case["from"], case["to"]), case["strength"]))
    assert set(_by_arc(got)) == set(expected)
    for arc, value in _by_arc(got).items():
        assert value == pytest.approx(expected[arc], rel=1e-11, abs=1e-300), arc


def test_both_kinds_of_criterion_are_covered():
    """The two branches compute completely different things; a fixture set
    that drifted to one of them would still pass every case above."""
    methods = {c["method"] for c in _records("arc.strength")}
    assert methods == {"test", "score"}

    defaults = {c["method"] for c in _records("arc.strength.default")}
    assert defaults == {"test", "score"}


def test_a_partially_directed_graph_has_no_arc_strengths(datasets):
    data = datasets["learning.test"]
    skeleton = pybnlearn.skeleton(pybnlearn.hc(data))

    with pytest.raises(ValueError, match="partially directed"):
        pybnlearn.arc_strength(skeleton, data)


# ---------------------------------------------------------------------------
# bootstrap strengths and the threshold they carry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("boot"),
    ids=lambda c: f"{c['dataset']}-seed{c['seed']}-R{c['replicates']}")
def test_bootstrap_strengths_and_threshold_match_r(case, datasets):
    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.boot_strength(datasets[case["dataset"]], "hc",
                                  replicates=int(case["replicates"]))

    assert list(got["from"]) == case["from"]
    assert list(got["to"]) == case["to"]
    assert np.allclose(got["strength"], case["strength"], rtol=1e-12,
                       atol=1e-14)
    assert np.allclose(got["direction"], case["direction"], rtol=1e-12,
                       atol=1e-14)
    assert got.attrs["threshold"] == pytest.approx(case["threshold"],
                                                   rel=1e-12, abs=1e-14)


@pytest.mark.parametrize(
    "case", _records("averaged"),
    ids=lambda c: (f"{c['dataset']}-seed{c['seed']}-R{c['replicates']}"
                   f"-{'default' if c['threshold'] is None else c['threshold']}"))
def test_averaged_network_matches_r(case, datasets):
    pybnlearn.set_seed(int(case["seed"]))
    strength = pybnlearn.boot_strength(datasets[case["dataset"]], "hc",
                                       replicates=int(case["replicates"]))

    got = pybnlearn.averaged_network(strength, threshold=case["threshold"])

    assert sorted(got.arcs) == sorted(tuple(a) for a in case["arcs"])
    assert set(got.nodes) == set(datasets[case["dataset"]].columns)


def test_averaging_covers_both_the_cyclic_and_acyclic_paths():
    """One branch takes the candidate arcs as they are, the other has to
    break cycles; only the second goes through the C code."""
    cases = _records("averaged")
    sizes = {len(c["arcs"]) for c in cases}
    assert len(sizes) > 1, "every averaged network has the same size"
    assert any(c["arcs"] for c in cases) and any(not c["arcs"] for c in cases)


def test_a_threshold_of_one_still_keeps_the_certain_arcs(datasets):
    """An arc that appeared in every replicate is kept even at threshold 1,
    where a strict comparison would drop everything."""
    pybnlearn.set_seed(42)
    strength = pybnlearn.boot_strength(datasets["learning.test"], "hc",
                                       replicates=20)

    assert (strength["strength"] == 1).any()
    assert pybnlearn.averaged_network(strength, threshold=1).narcs > 0


def test_averaging_refuses_statistics_that_are_not_frequencies(datasets):
    data = datasets["learning.test"]
    scored = pybnlearn.arc_strength(pybnlearn.hc(data), data)

    with pytest.raises(ValueError, match="frequencies"):
        pybnlearn.averaged_network(scored)


# ---------------------------------------------------------------------------
# custom.strength
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("custom"),
    ids=lambda c: f"{c['dataset']}-cpdag{c['cpdag']:d}-{c['weights']}")
def test_custom_strength_matches_r(case):
    networks = [pybnlearn.model2network(m) for m in case["networks"]]

    n = len(networks)
    weights = (None if case["weights"] == "equal"
               else [(i + 1) / (n * (n + 1) / 2) * n for i in range(n)])

    got = pybnlearn.custom_strength(networks, nodes=case["nodes"],
                                    weights=weights, cpdag=case["cpdag"])

    assert list(got["from"]) == case["from"]
    assert list(got["to"]) == case["to"]
    assert np.allclose(got["strength"], case["strength"], rtol=1e-12,
                       atol=1e-14)
    assert np.allclose(got["direction"], case["direction"], rtol=1e-12,
                       atol=1e-14)
    assert got.attrs["threshold"] == pytest.approx(case["threshold"],
                                                   rel=1e-12, abs=1e-14)


def test_custom_strength_accepts_the_shapes_a_network_comes_in(datasets):
    """A graph, a fitted network and a bare arc list all describe the same
    structure, and counting them must give the same answer."""
    data = datasets["learning.test"]
    learned = pybnlearn.hc(data)
    fitted = pybnlearn.fit(learned, data)

    nodes = [str(c) for c in data.columns]
    from_graph = pybnlearn.custom_strength([learned], nodes=nodes)
    from_fit = pybnlearn.custom_strength([fitted], nodes=nodes)
    from_arcs = pybnlearn.custom_strength([learned.arcs], nodes=nodes)

    assert np.allclose(from_graph["strength"], from_fit["strength"])
    assert np.allclose(from_graph["strength"], from_arcs["strength"])


def test_weights_have_to_match_the_networks(datasets):
    learned = pybnlearn.hc(datasets["learning.test"])

    with pytest.raises(ValueError, match="weights"):
        pybnlearn.custom_strength([learned, learned], weights=[1.0])


# ---------------------------------------------------------------------------
# the inclusion threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("threshold"),
                         ids=lambda c: c["name"])
def test_inclusion_threshold_matches_r(case):
    """R's optimize() is Brent's fmin, ported in strength.py rather than
    replaced: the result is a cutoff, so being close is not the same as
    agreeing."""
    strength = pd.DataFrame({"from": ["x"] * len(case["strength"]),
                             "to": ["y"] * len(case["strength"]),
                             "strength": case["strength"]})

    got = pybnlearn.inclusion_threshold(strength)

    assert got == pytest.approx(case["threshold"], rel=1e-12, abs=1e-14)


def test_the_threshold_is_always_one_of_the_observed_strengths():
    """quantile(type = 1) is the inverse empirical distribution, so it
    returns a value that was actually observed rather than interpolating
    between two."""
    for case in _records("threshold"):
        assert case["threshold"] in case["strength"], case["name"]


def test_degenerate_strength_vectors_are_covered():
    names = {c["name"] for c in _records("threshold")}
    assert {"all-zero", "all-one", "single"} <= names
