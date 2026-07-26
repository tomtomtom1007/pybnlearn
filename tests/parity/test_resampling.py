"""Check the bootstrap and cross-validation against R.

Both resample the data, so these are exact comparisons: two implementations
only agree on a bootstrap replicate if they drew the same rows in the same
order.  `test_sample_matches_r` checks that primitive directly, because
everything else here rests on it and a mismatch there would show up as a
puzzling disagreement much further downstream.

Fixtures come from tools/gen_r_resampling_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import math
import pathlib

import numpy as np
import pytest

import pybnlearn
from pybnlearn._core import sample_indices

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _records(kind):
    path = FIXTURES / "resampling.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _number(value):
    """The fixtures write Inf and NaN as strings, since JSON has no literal
    for either and a singular fold genuinely produces one."""
    if isinstance(value, str):
        return {"Inf": math.inf, "-Inf": -math.inf, "NaN": math.nan}[value]
    return value


def _same(got, expected):
    expected = _number(expected)
    if isinstance(expected, float) and math.isnan(expected):
        return math.isnan(got)
    if isinstance(expected, float) and math.isinf(expected):
        return got == expected
    return got == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "case", _records("sample"),
    ids=lambda c: (f"seed{c['seed']:g}-n{c['n']:g}-k{c['k']:g}"
                   f"{'-replace' if c['replace'] else ''}"))
def test_sample_matches_r(case):
    pybnlearn.set_seed(int(case["seed"]))
    got = sample_indices(int(case["n"]), int(case["k"]),
                         replace=case["replace"])

    assert list(map(int, got)) == [int(v) for v in case["values"]]


@pytest.mark.parametrize(
    "case", _records("boot"),
    ids=lambda c: (f"{c['dataset']}-{c['algorithm']}-seed{c['seed']:g}"
                   f"{'' if c['cpdag'] else '-nocpdag'}"
                   f"{'' if c['shuffle'] else '-noshuffle'}"))
def test_boot_strength_matches_r(case, datasets):
    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.boot_strength(
        datasets[case["dataset"]], algorithm=case["algorithm"],
        replicates=int(case["R"]), cpdag=case["cpdag"],
        shuffle=case["shuffle"])

    assert list(got["from"]) == case["from"]
    assert list(got["to"]) == case["to"]
    assert np.allclose(got["strength"], case["strength"],
                       rtol=1e-12, atol=1e-12)
    assert np.allclose(got["direction"], case["direction"],
                       rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "case", _records("cv"),
    ids=lambda c: (f"{c['dataset']}-{c['bn']}-{c['method']}-"
                   f"k{c['k']:g}-seed{c['seed']:g}"))
def test_bn_cv_matches_r(case, datasets):
    spec = case["bn"]
    if spec.startswith("["):
        spec = pybnlearn.model2network(spec)

    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.bn_cv(
        datasets[case["dataset"]], spec, k=int(case["k"]),
        method=case["method"],
        m=int(case["m"]) if case["m"] is not None else None)

    assert len(got) == len(case["losses"])
    for i, (fold, expected) in enumerate(zip(got, case["losses"])):
        assert _same(fold["loss"], expected), f"fold {i}"

    assert _same(got.mean, case["mean"])


def test_the_bootstrap_fixtures_vary_cpdag_and_shuffle():
    """Both flags change what gets counted, so leaving them at their defaults
    would test only one of four code paths."""
    combinations = {(c["cpdag"], c["shuffle"]) for c in _records("boot")}
    assert combinations == {(True, True), (True, False),
                            (False, True), (False, False)}


def test_cross_validation_covers_singular_folds():
    """A Gaussian fold can be singular and score an infinite loss; R reports
    it rather than hiding it, and so must this."""
    infinite = [c for c in _records("cv")
                if any(v == "Inf" for v in c["losses"])]
    assert infinite, "no fixture exercises an infinite fold loss"


def test_unknown_losses_are_reported(datasets):
    with pytest.raises(ValueError, match="unknown loss"):
        pybnlearn.bn_cv(datasets["learning.test"], "hc", loss="nonesuch", k=2)


def test_custom_folds(datasets):
    data = datasets["learning.test"]
    folds = [np.arange(1, 2501), np.arange(2501, 5001)]

    got = pybnlearn.bn_cv(data, "hc", method="custom-folds", folds=folds)

    assert len(got) == 2
    assert all(math.isfinite(fold["loss"]) for fold in got)
