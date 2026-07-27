"""Measure the one structural divergence from R, rather than assert it.

`direct_lingam` reproduces R's causal *ordering* exactly -- that is checked
arc-for-arc in test_lingam.py -- but selects each node's parents with a
score-based search where R uses glmnet's adaptive lasso, because glmnet is
not vendored.  The README says so.  What it could not say, until this file
existed, was how far apart the two actually are.

The answer is: six of the eight cases here agree exactly, and the two that
do not are both on the same data set, both differing only by arcs that this
keeps and R prunes -- two at one sample size, three at another.  So the
honest description is not "returns a different structure" but "is sometimes
less sparse", and these tests hold that description to account: if a change
made the divergence wider, or narrower, this notices.  Writing them found
the second case, which sampling by hand had missed.

Why not simply port glmnet and remove the divergence?  Because its lambda
path cannot be reproduced from the algorithm alone.  glmnet stops when the
gain in deviance falls below 1e-5 times the deviance, and on the first data
set tried that comparison came down to 7.865e-06 against 7.634e-06 -- a
three percent margin.  Tightening the coordinate-descent tolerance flips the
number of lambdas returned from 61 to 60, which moves which lambda is
selected.  Matching it needs glmnet's arithmetic, not its method, and would
still drift across platforms for the same reason the rest of this suite
compares at 1e-11 rather than exactly.

Fixtures come from tools/gen_r_lingam_arcs_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# The cases where the two disagree, and by exactly what.  Listed rather than
# tolerated in bulk: a divergence nobody has looked at is indistinguishable
# from a bug, and this is the whole documented extent of it.
KNOWN_DIFFERENCES = {
    ("gaussian.test", 1000): {
        "extra": {("F", "B"), ("F", "D"), ("G", "B")}, "missing": set()},
    ("gaussian.test", 5000): {
        "extra": {("A", "B"), ("C", "B")}, "missing": set()},
}


def _records():
    path = FIXTURES / "lingam_arcs.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load(case):
    return pd.read_csv(FIXTURES / case["file"],
                       dtype="float64").head(int(case["rows"]))


@pytest.mark.parametrize(
    "case", _records(), ids=lambda c: f"{c['dataset']}-{c['rows']}")
def test_the_ordering_matches_r_exactly(case):
    """The ordering is the part that *is* ported, and it has no exceptions."""
    got = pybnlearn.direct_lingam(_load(case))

    assert got.learning["ordering"] == case["ordering"]


@pytest.mark.parametrize(
    "case", _records(), ids=lambda c: f"{c['dataset']}-{c['rows']}")
def test_the_arcs_differ_from_r_only_where_recorded(case):
    """Agreement where it is expected, and the exact difference where it is
    not.  An unexpected difference fails; so does an expected one that has
    quietly gone away, because that would mean this file is describing a
    package that no longer exists."""
    expected = {tuple(a) for a in case["arcs"]}
    got = set(pybnlearn.direct_lingam(_load(case)).arcs)

    known = KNOWN_DIFFERENCES.get((case["dataset"], int(case["rows"])))

    if known is None:
        assert got == expected, (
            "a new divergence from R's adaptive lasso: "
            f"extra {sorted(got - expected)}, missing {sorted(expected - got)}")
        return

    assert got - expected == known["extra"]
    assert expected - got == known["missing"]


def test_the_divergence_is_as_small_as_the_readme_says():
    """The README claims this is rare and one-directional.  Both halves of
    that are measured here, so the claim cannot rot."""
    cases = _records()
    assert len(cases) >= 8

    differing = [c for c in cases
                 if (c["dataset"], int(c["rows"])) in KNOWN_DIFFERENCES]

    assert len(differing) == 2, "the divergence is no longer these two cases"
    assert {c["dataset"] for c in differing} == {"gaussian.test"}, (
        "the divergence has spread beyond the one data set it was measured on")

    # one-directional: this keeps arcs R prunes, never the reverse
    for spec in KNOWN_DIFFERENCES.values():
        assert spec["extra"], "expected extra arcs, not missing ones"
        assert not spec["missing"], (
            "this is now dropping arcs R keeps, which is a different and "
            "worse failure than being less sparse")


def test_being_less_sparse_is_what_the_difference_looks_like():
    """Not a restatement: this checks the arcs this adds are ones R *could*
    have added, meaning they respect the shared causal ordering.  An extra
    arc pointing against the ordering would be a genuine defect rather than
    a sparsity difference."""
    for case in _records():
        key = (case["dataset"], int(case["rows"]))
        if key not in KNOWN_DIFFERENCES:
            continue

        rank = {node: i for i, node in enumerate(case["ordering"])}
        for parent, child in KNOWN_DIFFERENCES[key]["extra"]:
            assert rank[parent] < rank[child], (parent, child)
