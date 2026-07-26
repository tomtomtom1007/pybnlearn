"""Check the constraint-based algorithms against networks learned by R.

The arc set is compared as a set but *including direction*: these algorithms
return partially directed graphs, where an undirected edge is two opposing
arcs, so dropping direction would hide exactly the disagreements that matter.

Fixtures come from tools/gen_r_constraint_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CONTINUOUS = {"gaussian.test", "marks"}

ALGORITHMS = {
    "gs": pybnlearn.gs,
    "iamb": pybnlearn.iamb,
    "inter.iamb": pybnlearn.inter_iamb,
}


@pytest.fixture(scope="session")
def datasets():
    loaded = {}
    for path in FIXTURES.glob("*.csv"):
        name = path.stem
        if name in CONTINUOUS:
            loaded[name] = pd.read_csv(path, dtype="float64")
        else:
            loaded[name] = pd.read_csv(path, dtype="category",
                                       keep_default_na=False, na_values=[])
    return loaded


def _cases():
    path = FIXTURES / "constraint.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _case_id(case):
    bits = [case["dataset"], case["algorithm"], case["test"],
            f"alpha={case['alpha']:g}"]
    if case["whitelist"]:
        bits.append("wl")
    if case["blacklist"]:
        bits.append("bl")
    if case["undirected"]:
        bits.append("undirected")
    return "-".join(bits)


def _arcs(encoded):
    return [tuple(a.split(">")) for a in encoded]


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_matches_r(case, datasets):
    data = datasets[case["dataset"]]

    learned = ALGORITHMS[case["algorithm"]](
        data,
        test=case["test"],
        alpha=case["alpha"],
        whitelist=_arcs(case["whitelist"]) or None,
        blacklist=_arcs(case["blacklist"]) or None,
        undirected=case["undirected"],
    )

    expected = set(_arcs(case["arcs"]))
    got = set(learned.arcs)

    assert got == expected, (
        f"arc sets differ\n"
        f"  only in R:         {sorted(expected - got)}\n"
        f"  only in pybnlearn: {sorted(got - expected)}")
