"""Check tabu search against networks learned by R.

Tabu search only diverges from hill climbing once it is forced to accept a move
that makes the score worse, so each fixture records whether R's tabu actually
ended up somewhere different from R's hc.  `test_suite_exercises_tabu` asserts
that enough of them do: a suite where every case agreed with hc would pass
without ever touching the tabu list, the loss-iteration counter or the
best-network bookkeeping.

Fixtures come from tools/gen_r_tabu_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _cases():
    path = FIXTURES / "tabu.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _case_id(case):
    bits = [case["dataset"], case["score"], f"tabu={case['tabu']:g}"]
    if case["extra"]:
        bits += [f"{k}={v:g}" for k, v in case["extra"].items()]
    if case["whitelist"]:
        bits.append("wl")
    if case["blacklist"]:
        bits.append("bl")
    if case["maxp"] is not None:
        bits.append(f"maxp={case['maxp']:g}")
    return "-".join(bits)


def _arcs(encoded):
    return [tuple(a.split(">")) for a in encoded]


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_tabu_matches_r(case, datasets):
    data = datasets[case["dataset"]]

    learned = pybnlearn.tabu(
        data,
        score=case["score"],
        tabu=int(case["tabu"]),
        whitelist=_arcs(case["whitelist"]) or None,
        blacklist=_arcs(case["blacklist"]) or None,
        maxp=case["maxp"] if case["maxp"] is not None else float("inf"),
        **case["extra"],
    )

    expected = set(_arcs(case["arcs"]))
    assert set(learned.arcs) == expected, (
        f"arc sets differ\n"
        f"  only in R:         {sorted(expected - set(learned.arcs))}\n"
        f"  only in pybnlearn: {sorted(set(learned.arcs) - expected)}")

    assert learned.modelstring() == case["modelstring"]

    got = pybnlearn.score(learned, data, by_node=True)
    for node, expected_score in zip(case["nodes"], case["node.scores"]):
        assert got[node] == pytest.approx(expected_score, rel=1e-11, abs=1e-12)


def test_suite_exercises_tabu():
    """Guard against the suite quietly degenerating into a second hc test."""
    cases = _cases()
    if not cases:
        pytest.skip("fixtures not generated")

    diverged = [c for c in cases if not c["same.as.hc"]]
    assert len(diverged) >= 10, (
        f"only {len(diverged)} of {len(cases)} cases differ from hill "
        "climbing; the tabu-specific paths are barely covered")


def test_tabu_list_size_is_validated(datasets):
    """A tabu list with no slots is a tabu search that cannot remember
    anything, so it walks straight back into the local maximum it just left;
    R refuses it, in check.tabu()'s wording, which is what is matched here."""
    with pytest.raises(ValueError, match="tabu list must be a positive"):
        pybnlearn.tabu(datasets["learning.test"], tabu=0)
