"""Check the BIF, DSC and NET readers and writers against R.

The files under fixtures/foreign/ were written by R from a network R fitted,
and the tables recorded alongside them are what R's own reader gets back.
So reading is compared against the network the file came from, not against
another parser's opinion of the syntax.

The order of the conditional distributions is the thing under test.  BIF
names each parent configuration, DSC gives numeric coordinates, and NET
gives nothing at all and relies on position -- so a reader that assumes the
wrong order still produces a well-formed network with the rows permuted, and
only a cell-by-cell comparison notices.  insurance is in the set because it
has 27 nodes, up to four parents and levels containing spaces and
punctuation, which is where a permutation would be least visible.

Fixtures come from tools/gen_r_foreign_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
FILES = FIXTURES / "foreign"

READERS = {"bif": pybnlearn.read_bif, "dsc": pybnlearn.read_dsc,
           "net": pybnlearn.read_net}
WRITERS = {"bif": pybnlearn.write_bif, "dsc": pybnlearn.write_dsc,
           "net": pybnlearn.write_net}


def _records(kind):
    path = FIXTURES / "foreign.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


@pytest.fixture(scope="session")
def loaded():
    cache = {}

    def get(fmt, name):
        if (fmt, name) not in cache:
            cache[fmt, name] = READERS[fmt](FILES / f"{name}.{fmt}")
        return cache[fmt, name]

    return get


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("read"),
    ids=lambda c: f"{c['format']}-{c['dataset']}-{c['node']}")
def test_reading_matches_r(case, loaded):
    fitted = loaded(case["format"], case["dataset"])
    node = fitted[case["node"]]

    assert list(node.parents) == case["parents"]
    assert list(node.probabilities.shape) == case["dim"]
    assert [list(l) for l in node.levels] == case["dimnames"]
    # R unrolls column-major, which is numpy's order="F".
    assert np.allclose(node.probabilities.reshape(-1, order="F"),
                       case["prob"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "case", _records("network"),
    ids=lambda c: f"{c['format']}-{c['dataset']}")
def test_the_structure_survives_the_file(case, loaded):
    fitted = loaded(case["format"], case["dataset"])

    assert fitted.nodes == case["nodes"]
    assert pybnlearn.bn_net(fitted).modelstring() == case["modelstring"]


def test_all_three_formats_are_read():
    assert {c["format"] for c in _records("read")} == {"bif", "dsc", "net"}


def test_the_three_formats_agree_with_each_other(loaded):
    """The same network written three ways has to read back the same three
    times.  This catches an ordering mistake in one format that the fixture
    for that format would have to be wrong in the same way to hide."""
    for dataset in {c["dataset"] for c in _records("network")}:
        reference = loaded("bif", dataset)

        for fmt in ("dsc", "net"):
            other = loaded(fmt, dataset)
            assert other.nodes == reference.nodes, (fmt, dataset)

            for node in reference.nodes:
                assert np.allclose(other[node].probabilities,
                                   reference[node].probabilities,
                                   rtol=1e-9, atol=1e-11), (fmt, dataset, node)


def test_a_non_alphabetical_level_order_survives(loaded):
    """lizards' Species is Sagrei then Distichus in R.  A reader that sorted
    the levels would transpose every table that involves it."""
    for fmt in READERS:
        fitted = loaded(fmt, "lizards")
        assert fitted["Species"].levels[0] == ["Sagrei", "Distichus"]


def test_a_read_network_can_be_used(loaded):
    """The point of reading a file is to do something with it, and the C
    code has its own opinion of what a fitted network looks like."""
    fitted = loaded("bif", "asia")

    pybnlearn.set_seed(1)
    generated = pybnlearn.rbn(fitted, 100)
    assert list(generated.columns) == fitted.nodes
    assert len(generated) == 100

    marginal = pybnlearn.query(fitted, "D")
    assert marginal.values.sum() == pytest.approx(1.0)


def test_a_file_of_the_wrong_format_is_reported():
    with pytest.raises(ValueError, match="conform"):
        pybnlearn.read_dsc(FILES / "learning.test.bif")

    with pytest.raises(ValueError, match="conform"):
        pybnlearn.read_bif(FILES / "learning.test.net")


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", sorted(WRITERS))
@pytest.mark.parametrize("dataset", ["learning.test", "asia", "lizards",
                                     "insurance"])
def test_writing_then_reading_gives_the_network_back(fmt, dataset, tmp_path,
                                                     loaded):
    """The writer is checked through the reader, which the cases above have
    already checked against R -- so a writer that put the rows in the wrong
    order would have to be wrong in exactly the way the reader is."""
    original = loaded("bif", dataset)

    path = tmp_path / f"{dataset}.{fmt}"
    WRITERS[fmt](path, original)
    back = READERS[fmt](path)

    assert back.nodes == original.nodes

    for node in original.nodes:
        assert list(back[node].parents) == list(original[node].parents), node
        assert back[node].levels == original[node].levels, node
        assert np.allclose(back[node].probabilities,
                           original[node].probabilities,
                           rtol=1e-8, atol=1e-10), node


@pytest.mark.parametrize("fmt", sorted(WRITERS))
def test_a_file_written_here_reads_the_same_as_the_one_r_wrote(fmt, tmp_path,
                                                              loaded):
    """Stronger than the round trip: the file R wrote and the file written
    here have to mean the same thing, so the two writers agree even though
    they lay the text out differently."""
    original = loaded(fmt, "learning.test")

    path = tmp_path / f"ours.{fmt}"
    WRITERS[fmt](path, original)
    ours = READERS[fmt](path)

    for node in original.nodes:
        assert np.allclose(ours[node].probabilities,
                           original[node].probabilities,
                           rtol=1e-8, atol=1e-10), node


def test_only_discrete_networks_can_be_written(datasets, tmp_path):
    data = datasets["gaussian.test"]
    fitted = pybnlearn.fit(pybnlearn.hc(data), data)

    for fmt, writer in WRITERS.items():
        with pytest.raises(ValueError, match="discrete"):
            writer(tmp_path / f"x.{fmt}", fitted)


def test_levels_with_punctuation_are_sanitised(tmp_path):
    """The grammars use commas and braces, so a level containing one has to
    be rewritten or the file cannot be parsed back."""
    dag = pybnlearn.model2network("[A]")
    fitted = pybnlearn.custom_fit(dag, {
        "A": {"prob": np.array([0.5, 0.5]),
              "levels": {"A": ["a,b", "c{d}"]}}})

    path = tmp_path / "x.bif"
    pybnlearn.write_bif(path, fitted)

    assert pybnlearn.read_bif(path)["A"].levels[0] == ["a_b", "c_d_"]


# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("dot"), ids=lambda c: c["graph"])
def test_dot_output_matches_r(case, tmp_path):
    """R's own file is compared line by line, except for the empty graph --
    see below."""
    net = pybnlearn.BayesianNetwork(case["nodes"],
                                    [tuple(a) for a in case["arcs"]])

    path = tmp_path / f"{case['graph']}.dot"
    pybnlearn.write_dot(path, net)
    ours = [line.strip() for line in path.read_text().splitlines() if
            line.strip()]

    theirs = [line.strip() for line in
              (FILES / case["file"]).read_text().splitlines() if line.strip()]

    if case["graph"] == "empty":
        # bnlearn 5.2.1's write.dot() omits `file = fd` when it writes the
        # node list of an arcless graph, so R's file is an empty digraph and
        # the nodes go to the console instead.  Reproducing that would mean
        # writing a file that silently drops every node, so it is not
        # reproduced; the nodes are written.
        assert theirs == ["digraph {", "}"]
        assert ours[0] == "digraph {" and ours[-1] == "}"
        assert [line for line in ours[1:-1]] == [f'"{n}" ;'
                                                 for n in case["nodes"]]
        return

    assert ours == theirs


def test_dot_covers_directed_undirected_and_mixed():
    graphs = {c["graph"] for c in _records("dot")}
    assert {"dag", "cpdag", "skeleton", "empty"} <= graphs


def test_dot_accepts_a_fitted_network(datasets, tmp_path):
    data = datasets["learning.test"]
    fitted = pybnlearn.fit(pybnlearn.hc(data), data)

    path = tmp_path / "fitted.dot"
    pybnlearn.write_dot(path, fitted)

    text = path.read_text()
    assert text.startswith("digraph {")
    for arc in pybnlearn.bn_net(fitted).arcs:
        assert f'"{arc[0]}" -> "{arc[1]}"' in text
