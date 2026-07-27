"""Check the remaining utilities against R.

perturb() and the hill-climbing restarts built on it, cextend_all(),
count_graphs(), alst(), dedup(), bn_boot(), loss() and bf_strength().

Two of these draw from R's generator and are therefore compared exactly
rather than statistically: perturb picks its operation and its arc, and
bn_boot picks its resamples.  The restarts only matter where hill climbing
has somewhere to get stuck, so alarm and insurance are in the set and not
just the toy networks.

bf_strength needs care for a different reason.  A Bayes factor between two
networks routinely runs past what a double holds, so the three weights are
accumulated in extended precision and only their normalised ratio comes back
as a double -- and the precision has to be *installed*, not merely used to
build the numbers, or 1 + 4e-30 rounds to 1 and every improbable arc's
direction collapses to a tie.

Fixtures come from tools/gen_r_misc_fixtures.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

GRAPHS = {
    "learning.test": "[A][C][F][B|A][D|A:C][E|B:F]",
    "chain": "[A][B|A][C|B][D|C][E|D]",
    "empty": "[A][B][C][D]",
}


def _records(kind):
    path = FIXTURES / "misc.json"
    if not path.exists():
        return []
    return [r for r in json.loads(path.read_text()) if r["kind"] == kind]


def _arcs(pairs):
    return sorted(tuple(a) for a in pairs)


# ---------------------------------------------------------------------------
# perturb
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("perturb"),
    ids=lambda c: (f"{c['graph']}-seed{c['seed']}-{c['nops']}"
                   + ("-" + "".join(o[0] for o in c["ops"])
                      if "ops" in c else "")))
def test_perturb_matches_r(case):
    """Exact, not statistical: every choice -- which operation, which arc --
    is a draw from R's generator."""
    network = pybnlearn.model2network(case["modelstring"])

    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.perturb(network, int(case["nops"]),
                            ops=case.get("ops",
                                         ("set", "drop", "reverse")))

    assert _arcs(got.arcs) == _arcs(case["arcs"])
    assert got.nodes == network.nodes


def test_perturb_actually_changes_something():
    """A perturbation that never moved an arc would pass every comparison
    above on a network that was already the answer."""
    changed = 0
    for case in _records("perturb"):
        original = pybnlearn.model2network(case["modelstring"])
        changed += _arcs(case["arcs"]) != _arcs(original.arcs)

    assert changed > len(_records("perturb")) // 2


def test_restricting_the_operations_restricts_the_result():
    """With only "set" allowed nothing can be removed, and with "drop" and
    "reverse" nothing can be added."""
    for case in _records("perturb"):
        if case.get("ops") != ["set"]:
            continue
        original = set(pybnlearn.model2network(case["modelstring"]).arcs)
        after = {tuple(a) for a in case["arcs"]}
        # setting an undirected arc removes a row, so only pairs may vanish
        assert {frozenset(a) for a in original} <= {frozenset(a)
                                                    for a in after}


# ---------------------------------------------------------------------------
# hill climbing with random restarts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("restart"),
    ids=lambda c: (f"{c['dataset']}-seed{c['seed']}"
                   f"-r{c['restart']}p{c['perturb']}"))
def test_hill_climbing_restarts_match_r(case, datasets):
    data = datasets[case["dataset"]].head(300)

    pybnlearn.set_seed(int(case["seed"]))
    learned = pybnlearn.hc(data, restart=int(case["restart"]),
                           perturb=int(case["perturb"]))

    assert _arcs(learned.arcs) == _arcs(case["arcs"])
    assert learned.modelstring() == case["modelstring"]


def test_restarts_change_the_answer_somewhere(datasets):
    """The point of restarting is to escape a local optimum.  On data where
    there is no local optimum to escape it changes nothing, so these cases
    have to include data where it does."""
    differ = 0
    for case in _records("restart"):
        plain = pybnlearn.hc(datasets[case["dataset"]].head(300))
        differ += _arcs(case["arcs"]) != _arcs(plain.arcs)

    assert differ, "restarting never changed the result"


def test_restarts_do_not_make_the_network_worse(datasets):
    """The best network found is kept, so restarting can only help."""
    for name in ("alarm", "insurance"):
        data = datasets[name].head(300)
        plain = pybnlearn.hc(data)

        pybnlearn.set_seed(1)
        restarted = pybnlearn.hc(data, restart=5, perturb=3)

        assert (pybnlearn.score(restarted, data)
                >= pybnlearn.score(plain, data) - 1e-8), name


def test_perturb_must_be_positive_when_restarting(datasets):
    with pytest.raises(ValueError, match="changes at each random restart"):
        pybnlearn.hc(datasets["learning.test"], restart=2, perturb=0)


# ---------------------------------------------------------------------------
# all the extensions of an equivalence class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("cextend.all"),
                         ids=lambda c: c["modelstring"][:24])
def test_cextend_all_matches_r(case):
    equivalence = pybnlearn.cpdag(
        pybnlearn.model2network(case["modelstring"]))

    got = pybnlearn.cextend_all(equivalence)

    assert len(got) == case["count"]
    assert (sorted(g.modelstring() for g in got)
            == sorted(case["extensions"]))


def test_every_extension_is_in_the_class():
    """The definition: each one is a DAG whose equivalence class is the one
    it came from."""
    for case in _records("cextend.all"):
        equivalence = pybnlearn.cpdag(
            pybnlearn.model2network(case["modelstring"]))

        for extension in pybnlearn.cextend_all(equivalence):
            assert pybnlearn.directed(extension)
            assert (set(pybnlearn.cpdag(extension).arcs)
                    == set(equivalence.arcs))


def test_a_fully_directed_class_has_exactly_one_extension():
    counts = {c["modelstring"]: c["count"] for c in _records("cextend.all")}
    assert 1 in counts.values()
    assert max(counts.values()) > 1


# ---------------------------------------------------------------------------
# counting graphs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("count"),
    ids=lambda c: (f"{c['type']}-" + (c["modelstring"][:16]
                                      if "modelstring" in c
                                      else f"n{c['nodes']}"
                                           + (f"k{c['k']}" if "k" in c else "")
                                           + (f"r{c['r']}" if "r" in c
                                              else ""))))
def test_count_graphs_matches_r(case):
    """Exact integers, and they outrun a double at about nine nodes -- R
    uses gmp for this and Python's own integers are already unbounded."""
    if case["type"] == "dags-with-r-arcs" and int(case.get("r", 1)) == 0:
        pytest.skip("R miscounts this corner; see the test below")

    kwargs = {}
    if "nodes" in case:
        kwargs["nodes"] = int(case["nodes"])
    if "k" in case:
        kwargs["k"] = int(case["k"])
    if "r" in case:
        kwargs["r"] = int(case["r"])
    if "modelstring" in case:
        kwargs["eqclass"] = pybnlearn.cpdag(
            pybnlearn.model2network(case["modelstring"]))

    assert pybnlearn.count_graphs(case["type"], **kwargs) == int(case["value"])


def test_a_graph_with_no_arcs_is_counted_once():
    """There is exactly one labelled DAG on n nodes with no arcs: the empty
    one.  R returns n instead.

    Its inner loop is `for (j in seq(from = 1, to = min(r, ...)))`, and
    R's seq() counts *downwards* when the end is below the start -- so with
    r = 0 the loop runs for j = 1 and then j = 0, and the second pass
    overwrites the count for zero arcs, which had been initialised to one.
    The rest of the function is right, and this is a corner rather than a
    systematic error, so the correct answer is returned rather than R's.

    The partition test below is what makes this more than an opinion: with
    R's value the counts for a given n do not add up to the number of DAGs.
    """
    for n in range(1, 7):
        assert pybnlearn.count_graphs("dags-with-r-arcs", nodes=n, r=0) == 1


def test_the_counts_are_bigger_than_a_double_can_hold():
    """Ten nodes is already past 2^53, which is why this is done with
    integers rather than with floating point."""
    assert pybnlearn.count_graphs("all-dags", nodes=10) > 2 ** 53
    assert pybnlearn.count_graphs("all-dags", nodes=20) > 10 ** 70


def test_the_roots_and_the_arcs_partition_the_dags():
    """Summing over every possible number of roots, or of arcs, has to give
    the total -- which is a check on all three recursions at once."""
    for n in range(1, 7):
        total = pybnlearn.count_graphs("all-dags", nodes=n)

        by_roots = sum(pybnlearn.count_graphs("dags-with-k-roots", nodes=n,
                                              k=k)
                       for k in range(1, n + 1))
        by_arcs = sum(pybnlearn.count_graphs("dags-with-r-arcs", nodes=n, r=r)
                      for r in range(0, n * (n - 1) // 2 + 1))

        assert by_roots == total, n
        assert by_arcs == total, n


# ---------------------------------------------------------------------------
# adjacency lists and deduplication
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("alst"),
                         ids=lambda c: c["modelstring"][:24])
def test_alst_matches_r(case):
    got = pybnlearn.alst(pybnlearn.model2network(case["modelstring"]))

    assert list(got) == case["nodes"]
    assert {k: list(v) for k, v in got.items()} == case["alst"]


def test_alst_lists_children_not_parents():
    """R's arcs2alist() goes the way the arcs point, which is the opposite
    of how a conditional probability table is indexed."""
    network = pybnlearn.model2network("[A][B|A]")

    assert pybnlearn.alst(network) == {"A": ["B"], "B": []}


@pytest.mark.parametrize("case", _records("dedup"),
                         ids=lambda c: f"{c['threshold']:g}")
def test_dedup_matches_r(case, datasets):
    got = pybnlearn.dedup(datasets["redundant"],
                          threshold=float(case["threshold"]))

    assert list(got.columns) == case["kept"]


def test_the_threshold_decides_how_close_is_too_close():
    """H is a multiple of A and I is a noisy copy of B; a strict threshold
    drops only the first, a loose one drops both."""
    kept = {c["threshold"]: c["kept"] for c in _records("dedup")}

    assert all("H" not in v for v in kept.values())
    assert len(set(len(v) for v in kept.values())) > 1


# ---------------------------------------------------------------------------
# bootstrapping a statistic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("boot"),
    ids=lambda c: f"{c['dataset']}-seed{c['seed']}-{c['statistic']}")
def test_bn_boot_matches_r(case, datasets):
    """Exact: the resamples come from R's generator, so the same seed draws
    the same rows in the same order."""
    data = datasets[case["dataset"]].head(500)

    statistic = ({"narcs": lambda net: net.narcs,
                  "modelstring": lambda net: net.modelstring()}
                 [case["statistic"]])

    pybnlearn.set_seed(int(case["seed"]))
    got = pybnlearn.bn_boot(data, statistic, replicates=10)

    if case["statistic"] == "narcs":
        assert got == [int(v) for v in case["values"]]
    else:
        assert got == case["strings"]


def test_bn_boot_and_boot_strength_draw_the_same_samples(datasets):
    """They are the same resampling loop with a different thing done to each
    network, so seeded identically they must see identical replicates."""
    data = datasets["learning.test"].head(300)

    pybnlearn.set_seed(9)
    networks = pybnlearn.bn_boot(data, lambda net: net, replicates=6)

    pybnlearn.set_seed(9)
    strength = pybnlearn.boot_strength(data, "hc", replicates=6, shuffle=False)

    counted = pybnlearn.custom_strength(networks,
                                        nodes=[str(c) for c in data.columns])

    assert np.allclose(counted["strength"], strength["strength"])


def test_loss_reads_a_cross_validation_result(datasets):
    result = pybnlearn.bn_cv(datasets["learning.test"], "hc", loss="pred",
                             target="B", k=3)

    assert pybnlearn.loss(result) == result.mean
    assert pybnlearn.loss([result, result]) == [result.mean, result.mean]


# ---------------------------------------------------------------------------
# Bayes factor arc strengths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("bf"),
                         ids=lambda c: f"{c['dataset']}-{c['score']}")
def test_bf_strength_matches_r(case, datasets):
    data = datasets[case["dataset"]].head(int(case["rows"]))
    network = pybnlearn.model2network(case["modelstring"])

    got = pybnlearn.bf_strength(network, data, type=case["score"])

    assert list(got["from"]) == case["from"]
    assert list(got["to"]) == case["to"]
    assert np.allclose(got["strength"], case["strength"], rtol=1e-9,
                       atol=1e-11)
    assert np.allclose(got["direction"], case["direction"], rtol=1e-7,
                       atol=1e-11)
    # the threshold is one of the strengths, so it cannot agree any more
    # closely than they do.
    assert got.attrs["threshold"] == pytest.approx(case["threshold"],
                                                   rel=1e-9, abs=1e-11)


def test_the_extended_precision_is_actually_needed():
    """A Bayes factor of exp(1152) is normal on this data.  Left in double
    precision the weights overflow and the strengths come out as NaN; left
    at Decimal's default 28 digits, 1 + 4e-30 rounds to 1 and every
    improbable arc's direction collapses to a tie."""
    cases = _records("bf")
    tiny = [v for c in cases for v in c["strength"] if 0 < v < 1e-20]
    assert tiny, "no arc improbable enough to need the precision"

    decisive = [c for c in cases
                for s, d in zip(c["strength"], c["direction"])
                if s < 1e-20 and d in (0.0, 1.0)]
    assert decisive, "no improbable arc with a decided direction"


def test_bf_strength_feeds_averaged_network(datasets):
    """It carries strengths and directions in the same shape boot_strength
    does, so the same consensus network can be built from it."""
    data = datasets["learning.test"].head(500)
    strength = pybnlearn.bf_strength(
        pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]"), data)

    averaged = pybnlearn.averaged_network(strength, threshold=0.5)

    assert averaged.narcs > 0
    assert pybnlearn.acyclic(averaged, directed=True)


# ---------------------------------------------------------------------------
# the test counter
# ---------------------------------------------------------------------------

def test_the_counter_records_what_a_search_did(datasets):
    """It is a global in the C core, so it has to be reset before it means
    anything -- which is what every learning algorithm does on entry."""
    pybnlearn.reset_test_counter()
    assert pybnlearn.test_counter() == 0

    pybnlearn.increment_test_counter(5)
    assert pybnlearn.test_counter() == 5

    pybnlearn.increment_test_counter()
    assert pybnlearn.test_counter() == 6

    pybnlearn.reset_test_counter()
    assert pybnlearn.test_counter() == 0
