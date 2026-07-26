"""Check the hybrid algorithms against networks learned by R.

Fixtures come from tools/gen_r_hybrid_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

ALGORITHMS = {"mmhc": pybnlearn.mmhc, "rsmax2": pybnlearn.rsmax2}




def _cases():
    path = FIXTURES / "hybrid.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _case_id(case):
    bits = [case["dataset"], case["algorithm"]]
    if case["restrict"]:
        bits.append(case["restrict"])
    if case["maximize"]:
        bits.append(case["maximize"])
    for key in ("restrict.args", "maximize.args"):
        bits += [f"{k}={v}" for k, v in case[key].items()]
    if case["whitelist"]:
        bits.append("wl")
    if case["blacklist"]:
        bits.append("bl")
    return "-".join(str(b) for b in bits)


def _arcs(encoded):
    return [tuple(a.split(">")) for a in encoded]


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_matches_r(case, datasets):
    data = datasets[case["dataset"]]

    kwargs = {
        "whitelist": _arcs(case["whitelist"]) or None,
        "blacklist": _arcs(case["blacklist"]) or None,
        "restrict_args": case["restrict.args"] or None,
        "maximize_args": case["maximize.args"] or None,
    }
    if case["algorithm"] == "rsmax2":
        kwargs["restrict"] = case["restrict"]
        kwargs["maximize"] = case["maximize"]

    learned = ALGORITHMS[case["algorithm"]](data, **kwargs)

    expected = set(_arcs(case["arcs"]))
    assert set(learned.arcs) == expected, (
        f"arc sets differ\n"
        f"  only in R:         {sorted(expected - set(learned.arcs))}\n"
        f"  only in pybnlearn: {sorted(set(learned.arcs) - expected)}")
    assert learned.modelstring() == case["modelstring"]


def test_h2pc_says_what_is_missing(datasets):
    with pytest.raises(NotImplementedError, match="hpc"):
        pybnlearn.h2pc(datasets["learning.test"])
