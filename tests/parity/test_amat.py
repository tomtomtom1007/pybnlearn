"""Check the adjacency-matrix accessor against R.

`amat` and `set_amat` are the only place bnlearn describes a whole graph at
once rather than an arc at a time, which makes them the way in and out of
anything that already speaks adjacency matrices.

The matrix is lossy in one specific way, and most of what is here exists to
pin that down: an undirected arc is stored as both directions, so it becomes
a symmetric pair of ones and is indistinguishable from two opposed directed
arcs.  Reading a partially directed graph and writing it back is therefore
*not* the identity on the arc set's meaning, only on the arc set itself, and
the fixtures record the round trip rather than assuming it.

Two conventions have to agree exactly with R or everything downstream is
transposed: row *i* column *j* means an arc from *i* to *j*, and the arc
order that comes out of the matrix decides how a model string reads.  Both
are compared, and the matrix is compared row by row rather than as a flat
buffer so that a transposition cannot hide.

Fixtures come from tools/gen_r_amat_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _records(kind):
    path = FIXTURES / "amat.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(pairs):
    return [tuple(a) for a in pairs]


def _network(case):
    """Rebuild the fixture's graph from its nodes and arcs, so that the node
    order is R's rather than whatever sorting would give."""
    return pybnlearn.BayesianNetwork(case["nodes"], _arcs(case["arcs"]))


def _matrix(case):
    """The fixture's matrix with its dimnames, when it had any.  Passing it
    bare would be a different input: the labelled cases are exactly the ones
    where positional and labelled reading disagree."""
    values = np.array(case["amat"])
    if case["labels"] is None:
        return values
    return pd.DataFrame(values, index=case["labels"], columns=case["labels"])


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("amat"), ids=lambda c: c["name"])
def test_the_adjacency_matrix_matches_r(case):
    got = pybnlearn.amat(_network(case))

    assert got.to_numpy().tolist() == case["amat"]
    assert list(got.index) == case["rownames"]
    assert list(got.columns) == case["colnames"]


@pytest.mark.parametrize("case", _records("amat"), ids=lambda c: c["name"])
def test_the_matrix_gives_the_arcs_back_in_rs_order(case):
    """The arc order is not cosmetic: it decides how a model string reads and
    how a conditional probability table's axes come out."""
    network = _network(case)

    got = pybnlearn.set_amat(network, pybnlearn.amat(network),
                             check_cycles=False)

    assert got.arcs == _arcs(case["roundtrip"])


def test_a_row_is_a_parent_and_a_column_is_a_child():
    """Transposing the convention would leave every fixture above still
    square and still binary, so it is worth saying out loud."""
    network = pybnlearn.model2network("[A][B|A]")

    matrix = pybnlearn.amat(network)

    assert matrix.loc["A", "B"] == 1
    assert matrix.loc["B", "A"] == 0


def test_the_matrix_is_labelled_in_the_networks_node_order():
    """R returns a bare matrix whose dimnames carry the node order; a frame
    carries the same thing where pandas can see it."""
    network = pybnlearn.hc(pd.read_csv(FIXTURES / "learning.test.csv",
                                       dtype="category"))

    matrix = pybnlearn.amat(network)

    assert list(matrix.index) == network.nodes
    assert list(matrix.columns) == network.nodes


def test_a_fitted_network_has_an_adjacency_matrix_too(datasets):
    data = datasets["learning.test"]
    network = pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]")

    fitted = pybnlearn.fit(network, data)

    assert (pybnlearn.amat(fitted).to_numpy()
            == pybnlearn.amat(network).to_numpy()).all()


def test_an_undirected_arc_is_a_symmetric_pair():
    """Which is also the reason the matrix cannot represent a partially
    directed graph faithfully -- `undirected_arcs` can, and this cannot."""
    network = pybnlearn.model2network("[A][B|A][C|B][D|C]")
    equivalence_class = pybnlearn.cpdag(network)

    matrix = pybnlearn.amat(equivalence_class).to_numpy()

    assert (matrix == matrix.T).all()
    assert pybnlearn.amat(network).to_numpy().tolist() != matrix.tolist()


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("assign"), ids=lambda c: c["name"])
def test_assigning_an_adjacency_matrix_matches_r(case):
    """The arcs, the model string when there is one, and every node's
    parents: the last of these is what the score functions actually read, and
    it is derived from the arcs rather than stored, so a graph can have the
    right arcs and the wrong parents."""
    network = pybnlearn.BayesianNetwork(case["nodes"], [])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # see the reordering test
        got = pybnlearn.set_amat(
            network, _matrix(case),
            check_cycles=(case["name"] != "cyclic-unchecked"))

    assert got.arcs == _arcs(case["arcs"])
    if case["modelstring"] is not None:
        assert got.modelstring() == case["modelstring"]
    for node, expected in case["parents"].items():
        assert got.parents(node) == expected


def test_reading_and_writing_a_directed_graph_is_a_round_trip():
    for modelstring in ("[A][C][F][B|A][D|A:C][E|B:F]",
                        "[A][B|A][C|A:B][D|A:B:C]",
                        "[A][B][C][D]"):
        network = pybnlearn.model2network(modelstring)

        back = pybnlearn.set_amat(network, pybnlearn.amat(network))

        assert back.modelstring() == modelstring
        assert sorted(back.arcs) == sorted(network.arcs)


def test_the_matrix_replaces_the_arcs_rather_than_adding_to_them():
    network = pybnlearn.model2network("[A][B|A][C|B][D|C]")
    empty = np.zeros((4, 4), dtype=int)

    assert pybnlearn.set_amat(network, empty).arcs == []


def test_the_original_network_is_left_alone():
    network = pybnlearn.model2network("[A][B|A][C|B][D|C]")
    before = list(network.arcs)

    pybnlearn.set_amat(network, np.zeros((4, 4), dtype=int))

    assert network.arcs == before


def test_the_learning_metadata_survives():
    data = pd.read_csv(FIXTURES / "learning.test.csv", dtype="category")
    network = pybnlearn.hc(data, blacklist=[("A", "B")])

    updated = pybnlearn.set_amat(network, pybnlearn.amat(network))

    assert updated.learning == network.learning


def test_labels_in_a_different_order_are_rearranged_not_read_positionally():
    """Reading them positionally would build a different graph from the one
    the matrix describes, so R warns and reorders.  A test that only checked
    the warning would pass on either behaviour."""
    network = pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]")
    matrix = pybnlearn.amat(network)
    shuffled = matrix.loc[matrix.index[::-1], matrix.columns[::-1]]

    with pytest.warns(UserWarning, match="rearranging"):
        got = pybnlearn.set_amat(network, shuffled)

    assert got.modelstring() == network.modelstring()

    # what reading it positionally would have given, for contrast.
    positional = pybnlearn.set_amat(network, shuffled.to_numpy())
    assert positional.modelstring() != network.modelstring()


def test_an_unlabelled_frame_is_read_positionally():
    """A frame that was never given labels has a RangeIndex, which is not a
    claim about node names and must not be treated as one."""
    network = pybnlearn.model2network("[A][B|A][C|B][D|C]")
    matrix = pd.DataFrame(pybnlearn.amat(network).to_numpy())

    got = pybnlearn.set_amat(network, matrix)

    assert got.modelstring() == network.modelstring()


def test_a_nested_sequence_works_as_well_as_an_array():
    network = pybnlearn.model2network("[A][B][C][D]")

    got = pybnlearn.set_amat(network, [[0, 1, 0, 0], [0, 0, 1, 0],
                                       [0, 0, 0, 1], [0, 0, 0, 0]])

    assert got.modelstring() == "[A][B|A][C|B][D|C]"


# ---------------------------------------------------------------------------
# what is rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("rejected"), ids=lambda c: c["name"])
def test_r_rejects_these_matrices_and_so_does_this(case):
    """The messages differ in wording -- these are not ports of R's strings --
    but every matrix R refuses is refused here, and for the same reason."""
    network = pybnlearn.BayesianNetwork(case["nodes"], [])

    assert case["error"] is not None
    with pytest.raises(ValueError):
        pybnlearn.set_amat(network, _matrix(case))


def test_the_rejections_are_distinguishable():
    """One catch-all message would satisfy the test above and tell a caller
    nothing about which of five things went wrong."""
    network = pybnlearn.model2network("[A][B|A][C|B][D|C]")
    expected = {
        "cyclic": "cycle",
        "diagonal": "diagonal",
        "not-binary": "0 or 1",
        "wrong-size": "nodes",
        "unknown-labels": "not present",
    }

    for case in _records("rejected"):
        with pytest.raises(ValueError, match=expected[case["name"]]):
            pybnlearn.set_amat(network, _matrix(case))


def test_a_non_square_matrix_is_refused():
    network = pybnlearn.model2network("[A][B][C][D]")

    with pytest.raises(ValueError, match="square"):
        pybnlearn.set_amat(network, np.zeros((4, 3), dtype=int))


def test_row_and_column_labels_have_to_agree():
    network = pybnlearn.model2network("[A][B][C][D]")
    matrix = pd.DataFrame(np.zeros((4, 4), dtype=int),
                          index=["A", "B", "C", "D"],
                          columns=["D", "C", "B", "A"])

    with pytest.raises(ValueError, match="mismatch"):
        pybnlearn.set_amat(network, matrix)


def test_the_cycle_check_can_be_turned_off():
    network = pybnlearn.model2network("[A][B|A][C|B][D|C]")
    cyclic = np.array([[0, 1, 0, 0], [0, 0, 1, 0],
                       [0, 0, 0, 1], [1, 0, 0, 0]])

    with pytest.raises(ValueError, match="cycle"):
        pybnlearn.set_amat(network, cyclic)

    got = pybnlearn.set_amat(network, cyclic, check_cycles=False)
    assert len(got.arcs) == 4


def test_an_illegal_arc_is_refused():
    """A network learned under parametric constraints records the arcs those
    constraints rule out; a matrix that reintroduces one is refused rather
    than quietly building a network the parameters cannot be fitted to."""
    network = pybnlearn.BayesianNetwork(
        ["A", "B", "C"], [], {"illegal": [("A", "B")]})
    matrix = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    with pytest.raises(ValueError, match="not valid"):
        pybnlearn.set_amat(network, matrix)

    assert pybnlearn.set_amat(network, matrix, check_illegal=False).arcs == [
        ("A", "B")]


def test_a_fitted_network_cannot_be_assigned_to():
    """`amat` reads one; `set_amat` would have to invent parameters for the
    arcs it added."""
    data = pd.read_csv(FIXTURES / "learning.test.csv", dtype="category")
    fitted = pybnlearn.fit(
        pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]"), data)

    with pytest.raises(TypeError):
        pybnlearn.set_amat(fitted, np.zeros((6, 6), dtype=int))
