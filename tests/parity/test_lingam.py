"""Check the direct LiNGAM causal ordering against R.

Only the ordering is compared, because only the ordering is ported: R turns
it into arcs with glmnet's adaptive lasso, and glmnet is not vendored here.
`direct_lingam` says so in its docstring and the test below says so again.

The ordering is greedy and deterministic, so a single wrong decision
cascades into every later one -- which is why the whole sequence is compared
rather than a summary of it, and why the pieces it is built from are also
checked separately, so a disagreement can be localised.

One data set has genuinely non-Gaussian noise, which is the assumption
LiNGAM rests on.  On Gaussian data the ordering is arbitrary in theory; that
it still agrees with R is a test of the arithmetic, which is what a port
needs.

Fixtures come from tools/gen_r_lingam_fixtures.R.

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
    path = FIXTURES / "lingam.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _identify(case):
    parts = [case["dataset"], str(case["rows"])]
    if "whitelist" in case:
        parts.append("wl")
    if "blacklist" in case:
        parts.append("bl")
    return "-".join(parts)


@pytest.mark.parametrize("case", _records("ordering"), ids=_identify)
def test_lingam_ordering_matches_r(case, datasets):
    data = datasets[case["dataset"]].head(int(case["rows"]))

    got = pybnlearn.lingam_ordering(
        data,
        whitelist=[tuple(a) for a in case.get("whitelist", [])] or None,
        blacklist=[tuple(a) for a in case.get("blacklist", [])] or None)

    assert got == case["ordering"]


def test_the_ordering_is_a_permutation():
    for case in _records("ordering"):
        assert sorted(case["ordering"]) == sorted(case["nodes"])


def test_non_gaussian_data_is_covered():
    """LiNGAM's whole claim rests on non-Gaussian noise; on Gaussian data
    the ordering is arbitrary in theory, so a fixture set made only of
    bnlearn's Gaussian data would not be testing the method at all."""
    assert any(c["dataset"] == "nongaussian" for c in _records("ordering"))


def test_the_ordering_finds_the_causal_direction(datasets):
    """On data generated from a known ordering with non-Gaussian noise, the
    recovered ordering should put causes before effects.  This is the claim
    the algorithm makes, checked against the truth rather than against R."""
    data = datasets["nongaussian"]

    got = pybnlearn.lingam_ordering(data)
    position = {node: i for i, node in enumerate(got)}

    # V1 -> V2 -> V3 -> V4, and V2, V4 -> V5 by construction.
    assert position["V1"] < position["V2"]
    assert position["V2"] < position["V3"]
    assert position["V3"] < position["V4"]
    assert position["V4"] < position["V5"]


def test_constraints_change_the_ordering():
    """A blacklist can fix part of the ordering before the search starts, so
    if the constrained and unconstrained runs agreed everywhere the
    constraint handling would be untested."""
    by_key = {}
    for case in _records("ordering"):
        kind = ("wl" if "whitelist" in case
                else "bl" if "blacklist" in case else "plain")
        by_key.setdefault(case["dataset"], {}).setdefault(
            kind, []).append(tuple(case["ordering"]))

    differ = [d for d, v in by_key.items()
              if len({o for orderings in v.values() for o in orderings}) > 1]
    assert differ, by_key


def test_a_blacklisted_source_becomes_a_leaf(datasets):
    """Forbidding every arc out of a node means nothing can follow it, so it
    goes last."""
    data = datasets["nongaussian"]
    nodes = list(data.columns)
    blacklist = [(nodes[0], other) for other in nodes[1:]]

    got = pybnlearn.lingam_ordering(data, blacklist=blacklist)

    assert got[-1] == nodes[0]


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("pairwise"),
                         ids=lambda c: f"{c['x']}-{c['y']}")
def test_the_pairwise_measure_matches_r(case, datasets):
    """Checked on its own so that a disagreement in the ordering can be
    localised to the measure or to the search around it."""
    from pybnlearn.lingam import (_pairwise_mutual_information, _remove_effect,
                                  _scale, _sd)

    data = datasets["gaussian.test"].head(int(case["rows"]))
    x = _scale(data[case["x"]])
    y = _scale(data[case["y"]])

    assert _pairwise_mutual_information(x, y) == pytest.approx(
        case["value"], rel=1e-10, abs=1e-12)
    assert _sd(_remove_effect(x, y)) == pytest.approx(
        case["residual.sd"], rel=1e-10, abs=1e-12)


def test_the_measure_is_antisymmetric(datasets):
    """It answers which of the two is the cause, so swapping them has to
    flip the sign."""
    from pybnlearn.lingam import _pairwise_mutual_information, _scale

    data = datasets["nongaussian"]

    for a, b in (("V1", "V2"), ("V2", "V3"), ("V1", "V5")):
        forward = _pairwise_mutual_information(_scale(data[a]),
                                               _scale(data[b]))
        backward = _pairwise_mutual_information(_scale(data[b]),
                                                _scale(data[a]))
        assert forward == pytest.approx(-backward, rel=1e-10, abs=1e-12)


# ---------------------------------------------------------------------------
# what is not ported, and why
# ---------------------------------------------------------------------------

def test_the_kernel_estimator_refuses_rather_than_guesses(datasets):
    """bnlearn's mi = "gkernel" builds its Gram matrix with a matrix product
    where it means an elementwise one.  That leaves the matrix with a
    condition number around 1e20, so its singular values are rounding noise
    and R's own answer is not reproducible -- by this package or by anything
    else.  Returning a number that merely looked like an answer would be
    worse than refusing."""
    with pytest.raises(NotImplementedError, match="gkernel"):
        pybnlearn.lingam_ordering(datasets["gaussian.test"], mi="gkernel")


def test_direct_lingam_uses_the_ordering(datasets):
    """The arcs are not R's -- bnlearn picks them with glmnet's adaptive
    lasso, which is not vendored -- but they must respect the ordering,
    which is."""
    data = datasets["nongaussian"]

    learned = pybnlearn.direct_lingam(data)
    ordering = learned.learning["ordering"]

    assert ordering == pybnlearn.lingam_ordering(data)

    rank = {node: i for i, node in enumerate(ordering)}
    for a, b in learned.arcs:
        assert rank[a] < rank[b], (a, b)

    assert pybnlearn.directed(learned)
    assert pybnlearn.acyclic(learned, directed=True)


def test_direct_lingam_accepts_both_searches(datasets):
    data = datasets["nongaussian"]

    for maximize in ("hc", "tabu"):
        learned = pybnlearn.direct_lingam(data, maximize=maximize)
        assert learned.learning["maximize"] == maximize
        assert learned.narcs > 0


def test_discrete_data_is_refused(datasets):
    with pytest.raises(ValueError, match="continuous"):
        pybnlearn.lingam_ordering(datasets["learning.test"])
