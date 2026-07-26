"""Check the Bayesian network classifiers against R.

Both the structures and the class posteriors are compared.  The structure of a
naive Bayes classifier is fixed by definition, so agreeing on it proves little;
the TAN cases are the ones that exercise anything, since its feature tree comes
from mutual information measured *given the class*, and the root decides how
that tree is oriented.

Fixtures come from tools/gen_r_classifier_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

BUILDERS = {
    "naive.bayes": pybnlearn.naive_bayes,
    "tree.bayes": pybnlearn.tree_bayes,
}


def _records(kind):
    path = FIXTURES / "classifiers.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(encoded):
    return [tuple(a.split(">")) for a in encoded]


def _build(case, data):
    kwargs = {}
    if case.get("explanatory"):
        kwargs["explanatory"] = case["explanatory"]
    if case.get("root"):
        kwargs["root"] = case["root"]
    return BUILDERS[case["algorithm"]](data, case["training"], **kwargs)


@pytest.mark.parametrize(
    "case", _records("structure"),
    ids=lambda c: (f"{c['dataset']}-{c['algorithm']}-{c['training']}"
                   f"{'-root' + c['root'] if c['root'] else ''}"
                   f"{'-subset' if c['explanatory'] else ''}"))
def test_structure_matches_r(case, datasets):
    learned = _build(case, datasets[case["dataset"]])

    assert learned.nodes == case["nodes"], "node order differs from R"
    assert set(learned.arcs) == set(_arcs(case["arcs"]))
    assert learned.modelstring() == case["modelstring"]


@pytest.mark.parametrize(
    "case", _records("predict"),
    ids=lambda c: f"{c['dataset']}-{c['algorithm']}-{c['training']}")
def test_classification_matches_r(case, datasets):
    data = datasets[case["dataset"]]
    fitted = pybnlearn.fit(_build(case, data), data)

    subset = data.head(int(case["n"]))
    predicted, probabilities = pybnlearn.classify(fitted, subset, prob=True)

    assert list(np.asarray(predicted).astype(str)) == case["predicted"]
    assert list(probabilities.columns) == case["levels"]

    for level, expected in case["probabilities"].items():
        assert np.allclose(probabilities[level].to_numpy(), expected,
                           rtol=1e-12, atol=1e-12), f"level {level}"

    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_the_suite_covers_more_than_naive_bayes():
    """A naive Bayes structure is fixed by definition, so a suite that only
    covered it would prove almost nothing about the port."""
    tan = [c for c in _records("structure") if c["algorithm"] == "tree.bayes"]
    assert len(tan) >= 10
    assert any(c["root"] for c in tan), "no case varies the tree's root"


def test_the_root_changes_the_tree(datasets):
    data = datasets["learning.test"]
    first = pybnlearn.tree_bayes(data, "F", root="A")
    second = pybnlearn.tree_bayes(data, "F", root="C")

    assert set(first.arcs) != set(second.arcs)


def test_continuous_data_is_rejected(datasets):
    with pytest.raises(ValueError, match="discrete"):
        pybnlearn.naive_bayes(datasets["gaussian.test"], "A")


def test_the_class_cannot_also_be_a_feature(datasets):
    with pytest.raises(ValueError, match="both"):
        pybnlearn.naive_bayes(datasets["learning.test"], "A",
                              explanatory=["A", "B"])


def test_arcs_touching_the_class_cannot_be_constrained(datasets):
    """Those arcs are fixed by the model, so constraining them is a mistake
    rather than something to silently ignore."""
    with pytest.raises(ValueError, match="training node"):
        pybnlearn.tree_bayes(datasets["learning.test"], "A",
                             blacklist=[("A", "B")])


def test_classify_needs_to_know_the_class(datasets):
    data = datasets["learning.test"]
    plain = pybnlearn.fit(pybnlearn.hc(data), data)

    with pytest.raises(ValueError, match="which node is the class"):
        pybnlearn.classify(plain, data.head(5))


def test_classify_rejects_networks_that_are_not_classifiers(datasets):
    """The C routine indexes the parameters by the classifier's shape and
    reads past them on anything else, so this has to be caught in Python.  R
    is protected by its class system instead."""
    data = datasets["learning.test"]
    plain = pybnlearn.fit(pybnlearn.hc(data), data)

    with pytest.raises(ValueError, match="not a naive Bayes or TAN"):
        pybnlearn.classify(plain, data.head(5), training="A")


def test_classify_rejects_a_class_node_with_parents(datasets):
    """A classifier's class node is a root; one with parents is a different
    model and the posterior would not be the product the C routine computes."""
    data = datasets["learning.test"][["A", "B", "C"]]
    fitted = pybnlearn.fit(pybnlearn.model2network("[C][A|C][B|A]"), data)

    with pytest.raises(ValueError, match="has parents"):
        pybnlearn.classify(fitted, data.head(5), training="A")
