"""Check parameter learning against R's bn.fit().

Every cell of every conditional probability table is compared, and every
regression coefficient, so a table that is right in aggregate but wrong in one
configuration still fails.

Fixtures come from tools/gen_r_fit_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _records(kind):
    path = FIXTURES / "fit.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _case_id(case):
    bits = [case["dataset"], case.get("method", "mle-g")]
    if case.get("iss") is not None:
        bits.append(f"iss={case['iss']:g}")
    bits.append(case["modelstring"])
    return "-".join(bits)


def _close(got, expected):
    """R writes unidentifiable parameters as NA; those must match as NaN."""
    if expected is None:
        return got is None or (isinstance(got, float) and math.isnan(got))
    return got == pytest.approx(expected, rel=1e-11, abs=1e-12, nan_ok=True)


@pytest.mark.parametrize("case", _records("discrete"), ids=_case_id)
def test_discrete_parameters(case, datasets):
    data = datasets[case["dataset"]]
    network = pybnlearn.model2network(case["modelstring"])

    fitted = pybnlearn.fit(network, data, method=case["method"],
                           iss=case["iss"] if case["iss"] is not None else 1)

    for expected in case["nodes"]:
        node = fitted[expected["node"]]
        assert node.parents == expected["parents"]

        assert list(node.probabilities.shape) == expected["dim"], (
            f"table shape for {expected['node']}")

        # R's as.vector() unrolls column-major, which is the order the values
        # are stored in on the C side too.
        got = node.probabilities.reshape(-1, order="F")
        assert len(got) == len(expected["values"])
        for i, (g, e) in enumerate(zip(got, expected["values"])):
            assert _close(float(g), e), (
                f"{expected['node']} cell {i}: R gave {e!r}, "
                f"pybnlearn gave {float(g)!r}")


@pytest.mark.parametrize("case", _records("gaussian"), ids=_case_id)
def test_gaussian_parameters(case, datasets):
    data = datasets[case["dataset"]]
    network = pybnlearn.model2network(case["modelstring"])

    fitted = pybnlearn.fit(network, data, method="mle-g")

    for expected in case["nodes"]:
        node = fitted[expected["node"]]
        assert node.parents == expected["parents"]

        assert list(node.coefficients) == expected["coefnames"]
        for name, e in zip(expected["coefnames"], expected["coefficients"]):
            assert _close(float(node.coefficients[name]), e), (
                f"{expected['node']} coefficient {name}")

        assert _close(node.sd, expected["sd"]), f"{expected['node']} sd"


def test_partially_directed_networks_are_rejected(datasets):
    """bn.fit needs a DAG: an undirected edge does not say which variable is
    conditioned on which."""
    data = datasets["learning.test"]
    skeleton = pybnlearn.gs(data, undirected=True)

    with pytest.raises(ValueError, match="partially directed"):
        pybnlearn.fit(skeleton, data)


def test_pdag2dag_makes_a_learned_cpdag_fittable(datasets):
    data = datasets["learning.test"]
    learned = pybnlearn.gs(data)
    fitted = pybnlearn.fit(pybnlearn.pdag2dag(learned), data)

    assert set(fitted.nodes) == set(data.columns)


def test_probabilities_sum_to_one(datasets):
    data = datasets["learning.test"]
    fitted = pybnlearn.fit(pybnlearn.hc(data), data)

    for node in fitted:
        # the node's own level is the first axis, so each column of the table
        # is one conditional distribution.
        totals = node.probabilities.sum(axis=0)
        assert np.allclose(totals, 1.0)
