"""Check hill climbing against networks learned by R.

Structure learning fails quietly: a search that diverges from R by a single arc
still returns a well-formed network, and only a comparison against R shows it.
So each case asserts three things -- the arc set, the model string, and the
per-node scores -- because any one of them alone can agree by luck.

Fixtures come from tools/gen_r_hc_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# The data sets whose columns are all numeric; everything else is categorical.
CONTINUOUS = {"gaussian.test", "marks"}


@pytest.fixture(scope="session")
def datasets():
    loaded = {}
    for path in FIXTURES.glob("*.csv"):
        name = path.stem
        if name in CONTINUOUS:
            loaded[name] = pd.read_csv(path, dtype="float64")
        else:
            # keep_default_na is essential: insurance has a category literally
            # called "None", and pandas would otherwise read 14681 of its rows
            # as missing, silently changing the level counts and the scores.
            loaded[name] = pd.read_csv(path, dtype="category",
                                       keep_default_na=False, na_values=[])
    return loaded


def _cases():
    path = FIXTURES / "hc.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _case_id(case):
    bits = [case["dataset"], case["score"]]
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
def test_hc_matches_r(case, datasets):
    data = datasets[case["dataset"]]

    learned = pybnlearn.hc(
        data,
        score=case["score"],
        whitelist=_arcs(case["whitelist"]) or None,
        blacklist=_arcs(case["blacklist"]) or None,
        maxp=case["maxp"] if case["maxp"] is not None else float("inf"),
        **case["extra"],
    )

    expected_arcs = set(_arcs(case["arcs"]))
    assert set(learned.arcs) == expected_arcs, (
        f"arc sets differ\n"
        f"  only in R:         {sorted(expected_arcs - set(learned.arcs))}\n"
        f"  only in pybnlearn: {sorted(set(learned.arcs) - expected_arcs)}")

    assert learned.modelstring() == case["modelstring"]

    # The scores are compared as well as the structure: two searches can land
    # on the same graph while disagreeing about what it is worth, which would
    # mean the score functions are wrong even though the search is not.
    got = pybnlearn.score(learned, data, by_node=True)
    for node, expected in zip(case["nodes"], case["node.scores"]):
        assert got[node] == pytest.approx(expected, rel=1e-12, abs=1e-12), (
            f"score of node {node}: R gave {expected!r}, "
            f"pybnlearn gave {got[node]!r}")
