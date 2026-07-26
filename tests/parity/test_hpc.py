"""Check hpc() and h2pc() against R.

HPC is three algorithms stacked on each other, and a mistake in any of them
shows up as a missing arc rather than as an error.  The cheap first pass
finds a superset of the neighbours using only tests of order zero and one;
the separating sets it produces drive the search for spouses; and only then
do the expensive tests run, inside that superset rather than over every
variable.  Get the first pass wrong and the rest never sees the node.

So the arc sets below are compared including direction, over data sets with
different shapes -- alarm has 37 nodes and large in-degrees, coronary has
variable names with spaces and dots, lizards is tiny -- and with the
conditioning-set limit that changes both supersets.

Fixtures come from tools/gen_r_hpc_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _records(kind):
    path = FIXTURES / "hpc.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(pairs):
    return sorted(tuple(a) for a in pairs)


def _identify(case):
    parts = [case["dataset"], case.get("test") or "default",
             f"{case['alpha']:g}"]
    if not case.get("undirected", True):
        parts.append("directed")
    if "max.sx" in case:
        parts.append(f"maxsx{case['max.sx']}")
    if "whitelist" in case:
        parts.append("wl")
    if "blacklist" in case:
        parts.append("bl")
    return "-".join(parts)


# ---------------------------------------------------------------------------
# hpc
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("hpc"), ids=_identify)
def test_hpc_matches_r(case, datasets):
    got = pybnlearn.hpc(
        datasets[case["dataset"]], test=case["test"],
        alpha=float(case["alpha"]),
        undirected=case.get("undirected", True),
        max_sx=case.get("max.sx"),
        whitelist=[tuple(a) for a in case.get("whitelist", [])] or None,
        blacklist=[tuple(a) for a in case.get("blacklist", [])] or None)

    assert _arcs(got.arcs) == _arcs(case["arcs"])


def test_hpc_is_undirected_by_default(datasets):
    """Like the other neighbourhood algorithms: it learns which nodes are
    adjacent without claiming which way the arcs point, and bnlearn defaults
    the same way."""
    learned = pybnlearn.hpc(datasets["learning.test"])

    assert not pybnlearn.directed(learned)
    directed = pybnlearn.hpc(datasets["learning.test"], undirected=False)
    assert len(directed.arcs) < len(learned.arcs)


def test_directing_the_result_is_covered():
    assert {c.get("undirected", True) for c in _records("hpc")} == {True,
                                                                    False}


def test_limiting_the_conditioning_set_changes_the_answer():
    """max_sx feeds into both supersets and into the filtering, so if it
    made no difference anywhere the fixtures would not be testing it."""
    by_key = {}
    for case in _records("hpc"):
        if case.get("test") is not None or "whitelist" in case \
                or "blacklist" in case or not case.get("undirected", True):
            continue
        by_key.setdefault(case["dataset"], {})[case.get("max.sx")] = _arcs(
            case["arcs"])

    differ = [d for d, v in by_key.items()
              if len({tuple(a) for a in v.values()}) > 1]
    assert differ, by_key


def test_hpc_is_not_a_rename_of_mmpc(datasets):
    """They look for the same thing, but not by the same tests, and they do
    not always agree -- on asia, mmpc keeps an adjacency hpc rules out.  If
    they agreed everywhere, none of these fixtures would be exercising hpc's
    own code."""
    agree = disagree = 0
    for name in ("learning.test", "asia", "coronary", "lizards", "alarm"):
        data = datasets[name]
        if _arcs(pybnlearn.hpc(data).arcs) == _arcs(pybnlearn.mmpc(data).arcs):
            agree += 1
        else:
            disagree += 1

    assert agree and disagree, (agree, disagree)


def test_a_large_network_is_covered():
    """alarm has 37 nodes and nodes with four parents, which is where the
    superset stages earn their keep -- and where an off-by-one in them would
    be least visible."""
    assert any(c["dataset"] == "alarm" for c in _records("hpc"))


# ---------------------------------------------------------------------------
# learn.nbr with hpc
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("nbr"),
                         ids=lambda c: f"{c['dataset']}-{c['node']}")
def test_learn_nbr_with_hpc_matches_r(case, datasets):
    got = pybnlearn.learn_nbr(datasets[case["dataset"]], case["node"],
                              method="hpc")

    assert sorted(got) == sorted(case["result"])


def test_the_local_and_global_results_agree(datasets):
    """Learning one node's neighbourhood and learning the whole skeleton are
    different code paths for the same question; the second symmetrises the
    first, so it can only add."""
    data = datasets["learning.test"]
    whole = pybnlearn.hpc(data)

    for node in data.columns:
        local = set(pybnlearn.learn_nbr(data, node, method="hpc"))
        assert local <= set(pybnlearn.nbr(whole, node)), node


# ---------------------------------------------------------------------------
# h2pc
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("h2pc"),
    ids=lambda c: f"{c['dataset']}" + ("-args" if c["args"] else ""))
def test_h2pc_matches_r(case, datasets):
    restrict = maximize = None
    if case["args"]:
        restrict = {"alpha": case["args"]["alpha"]}
        maximize = {"maxp": case["args"]["maxp"]}

    got = pybnlearn.h2pc(datasets[case["dataset"]], restrict_args=restrict,
                         maximize_args=maximize)

    assert _arcs(got.arcs) == _arcs(case["arcs"])
    assert got.modelstring() == case["modelstring"]


def test_h2pc_arguments_reach_the_right_phase():
    """alpha belongs to the restrict phase and maxp to the maximize phase;
    if they were swapped or dropped the two sets of cases would coincide."""
    by_dataset = {}
    for case in _records("h2pc"):
        by_dataset.setdefault(case["dataset"], []).append(case["modelstring"])

    assert any(len(set(v)) > 1 for v in by_dataset.values()), by_dataset


def test_h2pc_respects_the_parent_limit(datasets):
    learned = pybnlearn.h2pc(datasets["alarm"], maximize_args={"maxp": 2})

    for node in learned.nodes:
        assert len(learned.parents(node)) <= 2, node


def test_h2pc_is_no_longer_unimplemented(datasets):
    """It used to raise; the point of porting hpc was to make this work."""
    learned = pybnlearn.h2pc(datasets["learning.test"])

    assert learned.narcs > 0
    assert pybnlearn.directed(learned)
