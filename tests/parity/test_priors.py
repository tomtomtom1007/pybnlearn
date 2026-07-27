"""Check the non-uniform graph priors against R.

A graph prior says how likely a *structure* is before the data are seen,
which is a different thing from the parameter priors the Bayesian scores
already carry.  bnlearn has four, and they differ in more than their values:

* `uniform` leaves the score decomposable and score equivalent;
* `vsp` puts an independent probability on each arc -- still both;
* `marginal` puts one on every *pair* of nodes, adjacent or not, so the
  score stops being decomposable, but stays score equivalent because it does
  not care which way an arc points;
* `cs` -- Castelo and Siebes -- puts one on each arc *direction*, so the
  score is neither.

Those two flags are what these tests are really about.  They decide how much
of the score cache a search may reuse, and getting either wrong gives a
network that is wrong in a way no single score comparison would reveal.  So
`hc` and `tabu` are both here -- and they are *expected* to disagree with
each other under the marginal prior, because only `hc` recomputes the
endpoints' reference scores after a move.

Fixtures come from tools/gen_r_priors_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# the same informative priors the generator uses.
CS_BETA = {
    "learning.test": pd.DataFrame(
        {"from": ["A", "B", "C"], "to": ["B", "C", "E"],
         "prob": [0.9, 0.05, 0.9]}),
    "asia": pd.DataFrame({"from": ["A", "S"], "to": ["T", "L"],
                          "prob": [0.85, 0.7]}),
    "gaussian.test": pd.DataFrame({"from": ["A", "B"], "to": ["C", "D"],
                                   "prob": [0.75, 0.2]}),
}


def _records(kind):
    path = FIXTURES / "priors.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(pairs):
    return sorted(tuple(a) for a in pairs)


def _beta(case):
    """The fixture records a frame as the string "frame" rather than
    inlining it; the frames themselves are above."""
    if case["beta"] is None:
        return {}
    if case["beta"] == "frame":
        return {"beta": CS_BETA[case["dataset"]]}
    return {"beta": float(case["beta"])}


def _label(case):
    beta = ("" if case["beta"] is None
            else "-frame" if case["beta"] == "frame"
            else f"-{float(case['beta']):g}")
    return case["prior"] + beta


# ---------------------------------------------------------------------------
# completing a Castelo & Siebes prior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("completion"),
                         ids=lambda c: c["name"])
def test_the_castelo_completion_matches_r(case):
    """The user gives a probability for some arcs; the completion works out
    what that implies for both directions of every pair, and numbers the
    pairs the way the scoring code looks them up.  The numbering is as much
    a part of the answer as the probabilities."""
    from pybnlearn._core import complete_castelo_prior

    got = complete_castelo_prior(case["from"], case["to"], case["prob"],
                                 case["nodes"])

    assert got.frm == case["out.from"]
    assert got.to == case["out.to"]
    assert got.aid == [int(a) for a in case["aid"]]
    assert got.forward == pytest.approx(case["fwd"], rel=1e-12, abs=1e-14)
    assert got.backward == pytest.approx(case["bkwd"], rel=1e-12, abs=1e-14)


def test_an_unspecified_arc_gets_the_non_informative_share():
    """With nothing said about a pair, each of the three possibilities --
    one direction, the other, or no arc -- is equally likely, and the
    completion has nothing to record."""
    from pybnlearn._core import complete_castelo_prior

    empty = complete_castelo_prior([], [], [], list("ABCD"))
    assert len(empty) == 0

    one = complete_castelo_prior(["A"], ["B"], [0.8], list("ABCD"))
    assert one.forward == [0.8]
    # what is left over is split between the reverse arc and no arc at all.
    assert one.backward == pytest.approx([0.1])


def test_specifying_both_directions_is_respected():
    for case in _records("completion"):
        if case["name"] != "both-directions":
            continue
        # A -> B at 0.6 and B -> A at 0.3 leaves 0.1 for no arc.
        assert case["fwd"] == pytest.approx([0.6])
        assert case["bkwd"] == pytest.approx([0.3])


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("score"),
    ids=lambda c: (f"{c['dataset']}-{c['score']}-{_label(c)}"
                   f"-{c['modelstring'][:18]}"))
def test_scores_under_each_prior_match_r(case, datasets):
    network = pybnlearn.model2network(case["modelstring"])

    got = pybnlearn.score(network, datasets[case["dataset"]],
                          type=case["score"], prior=case["prior"],
                          **_beta(case))

    assert got == pytest.approx(case["value"], rel=1e-12, abs=1e-12)


def _score_table():
    table = {}
    for case in _records("score"):
        key = (case["dataset"], case["score"], case["modelstring"])
        table.setdefault(key, {})[_label(case)] = case["value"]
    return table


def test_the_priors_actually_change_the_score():
    """A prior that made no difference would let every comparison above pass
    on the uniform value."""
    changed = False
    for key, values in _score_table().items():
        if key[1] == "bdj":
            continue                      # see the next test
        assert len(set(values.values())) > 1, key
        changed = True
    assert changed


def test_the_jeffreys_score_ignores_the_graph_prior():
    """Not an oversight here: bnlearn's bdj accepts `prior` and `beta` --
    they are in score.extra.args, so passing them is not even a warning --
    and then scores as if the prior were uniform.  It is still reported as
    non-decomposable under `cs` and `marginal`, which is why the searches
    below do move even though the scores do not.  Reproducing the
    inconsistency is the point; smoothing it over would be a divergence.
    """
    seen = False
    for key, values in _score_table().items():
        if key[1] != "bdj":
            continue
        assert len(set(values.values())) == 1, key
        seen = True
    assert seen


def test_the_castelo_prior_distinguishes_directions():
    """Two networks in the same equivalence class score the same under every
    prior except an informative Castelo & Siebes one, which is the only one
    with an opinion about which way an arc points.  Left to its default it
    has no opinion about anything, and reduces to the uniform prior."""
    forward = "[A][C][F][B|A][D|A:C][E|B:F]"
    backward = "[B][C][F][A|B][E|B:F][D|A:C]"

    scores = {}
    for case in _records("score"):
        if case["dataset"] != "learning.test" or case["score"] != "bde":
            continue
        if case["modelstring"] in (forward, backward):
            scores[_label(case), case["modelstring"]] = case["value"]

    for label in {k[0] for k in scores}:
        pair = (scores.get((label, forward)), scores.get((label, backward)))
        if None in pair:
            continue
        differ = abs(pair[0] - pair[1]) > 1e-6
        assert differ == (label == "cs-frame"), label

    assert scores["cs", forward] == scores["uniform", forward]


# ---------------------------------------------------------------------------
# structure learning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("learn"),
    ids=lambda c: f"{c['dataset']}-{c['algorithm']}-{c['score']}-{_label(c)}")
def test_structure_learning_under_each_prior_matches_r(case, datasets):
    algorithm = getattr(pybnlearn, case["algorithm"])

    learned = algorithm(datasets[case["dataset"]], score=case["score"],
                        prior=case["prior"], **_beta(case))

    assert _arcs(learned.arcs) == _arcs(case["arcs"])
    assert learned.modelstring() == case["modelstring"]


def test_hc_and_tabu_disagree_under_the_marginal_prior():
    """Not a defect in either.  The marginal prior leaves the score
    equivalent, so the two networks below score the same and the search is
    picking between ties -- and only hc() recomputes the endpoints'
    reference scores after a move, which is what breaks the tie differently.
    Reproducing that is the point.
    """
    by_key = {}
    for case in _records("learn"):
        if case["dataset"] != "learning.test" or case["score"] != "bde":
            continue
        by_key.setdefault(_label(case), {})[case["algorithm"]] = (
            case["modelstring"])

    disagreeing = [label for label, v in by_key.items()
                   if len(set(v.values())) > 1]

    assert any(label.startswith("marginal") for label in disagreeing), by_key


def test_an_informative_prior_moves_the_search():
    """The prior is meant to be able to change the answer, not merely the
    number attached to it.  It does not manage that everywhere -- on
    learning.test with bde the data are decisive enough that every prior in
    the set lands on the same network -- so this asks that it happen
    somewhere, and names where."""
    by_key = {}
    for case in _records("learn"):
        key = (case["dataset"], case["algorithm"], case["score"])
        by_key.setdefault(key, {})[_label(case)] = case["modelstring"]

    moved = {key for key, v in by_key.items()
             if any(ms != v["uniform"] for ms in v.values())}

    assert ("learning.test", "hc", "bdj") in moved
    assert ("learning.test", "tabu", "bde") in moved


# ---------------------------------------------------------------------------
# the two flags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("flags"),
                         ids=lambda c: f"{c['score']}-{c['prior']}")
def test_the_equivalence_and_decomposability_flags_match_r(case):
    """These are not cosmetic: they decide how much of the score cache the
    search may reuse between iterations."""
    from pybnlearn.structure import (_is_score_decomposable,
                                     _is_score_equivalent)

    extra = {"prior": case["prior"]}

    assert _is_score_equivalent(case["score"], extra) is case["equivalent"]
    assert _is_score_decomposable(case["score"], extra) is case["decomposable"]


def test_both_flags_take_both_values_in_the_fixtures():
    cases = _records("flags")
    assert {c["equivalent"] for c in cases} == {True, False}
    assert {c["decomposable"] for c in cases} == {True, False}


def test_only_the_castelo_prior_breaks_score_equivalence():
    for case in _records("flags"):
        if case["score"] not in ("bde", "bge"):
            continue
        assert case["equivalent"] == (case["prior"] != "cs"), case


def test_only_the_pairwise_priors_break_decomposability():
    for case in _records("flags"):
        if case["score"] in ("bic", "loglik"):
            assert case["decomposable"], case
        else:
            assert case["decomposable"] == (
                case["prior"] not in ("cs", "marginal")), case


# ---------------------------------------------------------------------------
# argument checking
# ---------------------------------------------------------------------------

def test_an_unknown_prior_is_reported(datasets):
    with pytest.raises(ValueError, match="graph prior"):
        pybnlearn.score(pybnlearn.hc(datasets["learning.test"]),
                        datasets["learning.test"], type="bde",
                        prior="nonesuch")


def test_which_scores_accept_a_prior_at_all(datasets):
    """bdj takes the arguments and ignores them; k2 does not take them.  Both
    are R's, and the difference is in score.extra.args rather than in
    anything the scores do."""
    data = datasets["learning.test"]
    network = pybnlearn.hc(data)

    pybnlearn.score(network, data, type="bdj", prior="marginal", beta=0.9)

    with pytest.raises(ValueError, match="does not take"):
        pybnlearn.score(network, data, type="k2", prior="marginal", beta=0.9)


def test_the_uniform_prior_takes_no_hyperparameter(datasets):
    with pytest.raises(ValueError, match="no beta"):
        pybnlearn.score(pybnlearn.hc(datasets["learning.test"]),
                        datasets["learning.test"], type="bde",
                        prior="uniform", beta=0.5)


def test_beta_has_to_be_a_probability(datasets):
    for prior in ("vsp", "marginal"):
        with pytest.raises(ValueError, match="probability"):
            pybnlearn.score(pybnlearn.hc(datasets["learning.test"]),
                            datasets["learning.test"], type="bde",
                            prior=prior, beta=1.5)


def test_the_castelo_prior_checks_its_frame(datasets):
    data = datasets["learning.test"]
    network = pybnlearn.hc(data)

    with pytest.raises(ValueError, match="from, to and prob"):
        pybnlearn.score(network, data, type="bde", prior="cs",
                        beta=pd.DataFrame({"a": [1], "b": [2], "c": [3]}))

    with pytest.raises(ValueError, match="unknown node"):
        pybnlearn.score(network, data, type="bde", prior="cs",
                        beta=pd.DataFrame({"from": ["Z"], "to": ["A"],
                                           "prob": [0.5]}))

    with pytest.raises(ValueError, match="probabilities"):
        pybnlearn.score(network, data, type="bde", prior="cs",
                        beta=pd.DataFrame({"from": ["A"], "to": ["B"],
                                           "prob": [2.0]}))


def test_the_castelo_prior_accepts_triples(datasets):
    """A DataFrame is R's shape; a list of triples is the obvious Python
    one, and they must give the same answer."""
    data = datasets["learning.test"]
    network = pybnlearn.hc(data)

    frame = pybnlearn.score(
        network, data, type="bde", prior="cs",
        beta=pd.DataFrame({"from": ["A"], "to": ["B"], "prob": [0.9]}))
    triples = pybnlearn.score(network, data, type="bde", prior="cs",
                              beta=[("A", "B", 0.9)])

    assert frame == triples
