"""Check conditional independence tests against values produced by R.

The fixtures in fixtures/ci_test.json come from tools/gen_r_fixtures.R, which
runs R's own bnlearn.  Comparing against them is the whole point of the port:
the claim is that pybnlearn *matches* R, and that claim is only worth making if
something checks it.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import math
import pathlib

import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _scalar(value):
    """R's htest fields are named vectors, which arrive as one-item dicts."""
    if isinstance(value, dict):
        return float(next(iter(value.values())))
    return None if value is None else float(value)


@pytest.fixture(scope="session")
def datasets():
    return {
        # discrete networks are factors in R, so categorical here
        # keep_default_na=False so that category labels which happen to look
        # like missing-value markers ("None", "NA") survive the round trip.
        "learning.test": pd.read_csv(
            FIXTURES / "learning.test.csv", dtype="category",
            keep_default_na=False, na_values=[]),
        "gaussian.test": pd.read_csv(
            FIXTURES / "gaussian.test.csv", dtype="float64"),
    }


def _cases():
    path = FIXTURES / "ci_test.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _case_id(case):
    sx = "|" + ",".join(case["sx"]) if case["sx"] else ""
    return f"{case['dataset']}:{case['x']}~{case['y']}{sx}:{case['test']}"


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_matches_r(case, datasets):
    data = datasets[case["dataset"]]

    got = pybnlearn.ci_test(
        data, case["x"], case["y"],
        sx=case["sx"] or None,
        test=case["test"],
    )

    # Tolerances are tight on purpose.  nmath and R's RNG are vendored so that
    # the arithmetic is the same arithmetic, not merely equivalent; a
    # disagreement beyond floating-point noise means something is actually
    # wrong, and loosening the tolerance would hide it.
    for field in ("statistic", "parameter", "p.value"):
        expected = case[field]
        actual = _scalar(got[field])

        if expected is None:
            assert actual is None or math.isnan(actual), field
            continue

        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), (
            f"{field}: R gave {expected!r}, pybnlearn gave {actual!r}")


def test_unknown_test_raises(datasets):
    with pytest.raises(pybnlearn.BNLearnError, match="unknown test"):
        pybnlearn.ci_test(datasets["learning.test"], "A", "B", test="nonesuch")
