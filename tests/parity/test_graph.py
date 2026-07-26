"""Check the graph utilities and pairwise learners against R.

Fixtures come from tools/gen_r_graph_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _records(kind):
    path = FIXTURES / "graph.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(encoded):
    return [tuple(a.split(">")) for a in encoded]


@pytest.mark.parametrize("case", _records("transform"),
                         ids=lambda c: c["dataset"])
def test_transformations(case, datasets):
    data = datasets[case["dataset"]]
    net = pybnlearn.hc(data)

    assert set(net.arcs) == set(_arcs(case["arcs"])), "hc disagreed"
    assert net.modelstring() == case["modelstring"]

    assert set(pybnlearn.cpdag(net).arcs) == set(_arcs(case["cpdag"]))
    assert set(pybnlearn.skeleton(net).arcs) == set(_arcs(case["skeleton"]))
    assert set(pybnlearn.moral(net).arcs) == set(_arcs(case["moral"]))
    assert pybnlearn.nparams(net, data) == pytest.approx(case["nparams"])


@pytest.mark.parametrize("case", _records("compare"),
                         ids=lambda c: f"{c['dataset']}:{c['target']}")
def test_comparisons(case):
    target = pybnlearn.model2network(case["target"])
    current = pybnlearn.model2network(case["current"])

    assert pybnlearn.shd(current, target) == case["shd"]
    assert pybnlearn.shd(current, target, use_cpdag=False) \
        == case["shd.nocpdag"]
    assert pybnlearn.hamming(current, target) == case["hamming"]

    got = pybnlearn.compare(target, current)
    assert got == {"tp": case["tp"], "fp": case["fp"], "fn": case["fn"]}


@pytest.mark.parametrize("case", _records("modelstring"),
                         ids=lambda c: c["input"])
def test_model_string_round_trip(case):
    net = pybnlearn.model2network(case["input"])

    assert net.nodes == case["nodes"], "node order differs from R"
    assert set(net.arcs) == set(_arcs(case["arcs"]))
    # R's model2network sorts the nodes, and the node order decides how
    # modelstring() lists parents, so this catches an ordering mismatch that
    # the arc set alone would not.
    assert net.modelstring() == case["output"]


@pytest.mark.parametrize("case", _records("chow.liu"),
                         ids=lambda c: f"{c['dataset']}-{c['mi']}")
def test_chow_liu(case, datasets):
    learned = pybnlearn.chow_liu(datasets[case["dataset"]], mi=case["mi"])
    assert set(learned.arcs) == set(_arcs(case["arcs"]))


@pytest.mark.parametrize("case", _records("aracne"),
                         ids=lambda c: f"{c['dataset']}-{c['mi']}")
def test_aracne(case, datasets):
    learned = pybnlearn.aracne(datasets[case["dataset"]], mi=case["mi"])
    assert set(learned.arcs) == set(_arcs(case["arcs"]))
