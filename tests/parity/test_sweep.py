"""An exhaustive sweep of the learning surface against R.

The other parity files are collections of cases chosen by hand, and
hand-chosen cases share the blind spots of the hand that chose them.  Three
in particular, which this file exists to cover:

* **Arguments are varied one at a time elsewhere.**  Here they are crossed,
  so a pair that interacts is reached -- a whitelist together with a parent
  limit that has no room for the arc it wants, for instance.
* **The data are always one of bnlearn's own sets elsewhere**, and those are
  comfortable: thousands of rows, two to five levels, nothing degenerate.
  The ten data sets here are awkward on purpose -- two observations per
  contingency-table cell, eleven levels against two, 98% of the mass on one
  level, near-determinism, more variables than the sample supports, nearly
  collinear columns, values spanning ten orders of magnitude, Cauchy tails,
  and thirty rows.
* **A few exported functions were never called at all**, which a runtime
  count of the suite established rather than a reading of it.  `subgraph`,
  `connected_components` and `reversible_arcs` are here for that reason.

Failures are part of the comparison.  Some of these combinations have no
solution and R raises; the fixture records that it raised, and the test
asserts this raises too.  Skipping them would leave the most interesting
cases untested, since an argument combination that cannot be satisfied is
exactly where two implementations are most likely to diverge.

Fixtures come from tools/gen_r_sweep_fixtures.R, which also writes the data.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# The sweep deliberately includes combinations with no solution.  R's message
# and this package's differ in wording -- these are not ports of R's strings
# -- so what is compared is that both refuse, not how they say so.
TOLERANCE = dict(rel=1e-10, abs=1e-10)

# Two of the data sets are ill-conditioned on purpose, and on those the
# Gaussian quantities cannot agree to 1e-10 with anything -- including with R
# itself run twice against different linear algebra.
#
# The C is bnlearn's own and the input bits are identical, so the difference
# enters below the C: R here links OpenBLAS and its own reference LAPACK,
# this package links whatever the platform provides (Accelerate on macOS,
# OpenBLAS on the Linux and Windows wheels).  Different summation orders,
# amplified by the conditioning.
#
# "collinear" has a correlation matrix with a condition number near 7e12, so
# the forward error bound for a partial correlation on it is about
# 7e12 * 2.2e-16 = 1.5e-3.  Measured against exact rational arithmetic, R is
# wrong by 6.6e-5 and this package by 2.9e-4: both well inside the bound, and
# neither meaningfully closer to the truth.  See
# test_the_ill_conditioned_disagreement_is_not_a_defect, which does that
# arithmetic rather than asserting it.
ILL_CONDITIONED = {"collinear", "scaled"}
LOOSE = dict(rel=2e-3, abs=1e-8)


def _tolerance(dataset, test=None):
    if dataset in ILL_CONDITIONED and (test is None or test != "mi"):
        return LOOSE
    return TOLERANCE


def _all_records():
    path = FIXTURES / "sweep.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _records(kind):
    return [r for r in _all_records() if r["kind"] == kind]


@pytest.fixture(scope="module")
def sweep_data():
    """The sweep's own data sets, typed from the fixture rather than guessed.

    The "wide" set's levels are "0" and "1", which read back as numbers and
    would send a discrete data set to a continuous score -- so which columns
    are factors is recorded by the generator, not inferred here.
    """
    loaded = {}
    for spec in _records("dataset"):
        frame = pd.read_csv(FIXTURES / f"sweep.{spec['name']}.csv",
                            dtype=str, keep_default_na=False, na_values=[])
        for column in frame.columns:
            if column in spec["discrete"]:
                frame[column] = pd.Categorical(
                    frame[column], categories=spec["levels"][column])
            else:
                frame[column] = frame[column].astype("float64")
        assert list(frame.columns) == spec["nodes"]
        assert len(frame) == int(spec["rows"])
        loaded[spec["name"]] = frame
    return loaded


def _network(nodes, arcs):
    return pybnlearn.BayesianNetwork(nodes, [tuple(a) for a in arcs])


def _structure(name, nodes):
    """The three structures the generator scores, rebuilt here."""
    if name == "empty":
        return _network(nodes, [])
    if name == "chain":
        return _network(nodes, list(zip(nodes, nodes[1:])))
    if name == "star":
        return _network(nodes, [(n, nodes[-1]) for n in nodes[:-1]])
    raise AssertionError(name)


def _arcs(pairs):
    return sorted(tuple(a) for a in pairs)


def _float(value):
    if value is None:
        return None
    if value == "inf":
        return np.inf
    if value == "-inf":
        return -np.inf
    return float(value)


# ---------------------------------------------------------------------------
# the data themselves
# ---------------------------------------------------------------------------

def test_the_sweep_data_round_tripped_through_csv(sweep_data):
    """Everything below compares against numbers R computed from these exact
    frames.  If a column came back as the wrong type or lost a digit, every
    comparison after this one is measuring the wrong thing."""
    assert len(sweep_data) == 10

    for spec in _records("dataset"):
        frame = sweep_data[spec["name"]]
        for column in spec["discrete"]:
            assert isinstance(frame[column].dtype, pd.CategoricalDtype)
            assert list(frame[column].cat.categories) == spec["levels"][column]
        for column in set(spec["nodes"]) - set(spec["discrete"]):
            assert frame[column].dtype == np.float64
            assert np.isfinite(frame[column]).all()


def test_the_sweep_data_are_actually_awkward(sweep_data):
    """The point of these sets is that they are not comfortable.  If a later
    edit made them tame, the sweep would still pass and stop testing
    anything."""
    counts = sweep_data["sparse-cells"].groupby(
        list(sweep_data["sparse-cells"].columns[:3]), observed=True).size()
    assert counts.mean() < 5, "sparse-cells is no longer sparse"

    assert sweep_data["many-levels"]["A"].nunique() >= 10
    assert sweep_data["unbalanced"]["A"].value_counts(normalize=True).max() > 0.9
    assert len(sweep_data["small-n"]) <= 40
    assert sweep_data["wide"].shape[1] > sweep_data["wide"].shape[0] / 5

    collinear = sweep_data["collinear"]
    assert abs(np.corrcoef(collinear["A"], collinear["B"])[0, 1]) > 0.999

    scaled = sweep_data["scaled"]
    assert scaled["B"].abs().mean() / scaled["A"].abs().mean() > 1e6


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("score"),
    ids=lambda c: f"{c['dataset']}-{c['structure']}-{c['score']}")
def test_every_score_on_every_dataset(case, sweep_data):
    data = sweep_data[case["dataset"]]
    network = _structure(case["structure"], case["nodes"])

    if not case["ok"]:
        with pytest.raises((ValueError, pybnlearn.BNLearnError)):
            pybnlearn.score(network, data, type=case["score"])
        return

    got = pybnlearn.score(network, data, type=case["score"])
    assert got == pytest.approx(_float(case["value"]),
                                **_tolerance(case["dataset"]))


@pytest.mark.parametrize(
    "case", [c for c in _records("score") if c["by.node"] is not None],
    ids=lambda c: f"{c['dataset']}-{c['structure']}-{c['score']}")
def test_the_per_node_breakdown_matches_too(case, sweep_data):
    """A total can be right while the parts making it up are wrong in ways
    that cancel -- which is exactly what happens when two nodes' parent sets
    are swapped."""
    data = sweep_data[case["dataset"]]
    network = _structure(case["structure"], case["nodes"])

    got = pybnlearn.score(network, data, type=case["score"], by_node=True)

    tolerance = _tolerance(case["dataset"])
    for node, expected in zip(case["nodes"], case["by.node"]):
        assert got[node] == pytest.approx(_float(expected), **tolerance), node


# ---------------------------------------------------------------------------
# conditional independence tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("citest"),
    ids=lambda c: (f"{c['dataset']}-{c['test']}-{c['x']}{c['y']}"
                   f"-{len(c['sx'])}"))
def test_every_test_over_every_pair(case, sweep_data):
    data = sweep_data[case["dataset"]]

    if not case["ok"]:
        with pytest.raises((ValueError, pybnlearn.BNLearnError)):
            pybnlearn.ci_test(data, case["x"], case["y"],
                              sx=case["sx"] or None, test=case["test"])
        return

    got = pybnlearn.ci_test(data, case["x"], case["y"],
                            sx=case["sx"] or None, test=case["test"])

    tolerance = _tolerance(case["dataset"], case["test"])
    statistic = list(got["statistic"].values())[0]
    assert statistic == pytest.approx(_float(case["statistic"]), **tolerance)
    assert got["p.value"] == pytest.approx(
        _float(case["p.value"]),
        rel=max(tolerance["rel"], 1e-9), abs=1e-9)
    # the degrees of freedom are counted, not computed, so they are exact on
    # every data set however badly conditioned it is.
    if case["df"] is not None and got.get("parameter"):
        assert list(got["parameter"].values())[0] == pytest.approx(
            _float(case["df"]), **TOLERANCE)


# ---------------------------------------------------------------------------
# score-based learning, arguments crossed
# ---------------------------------------------------------------------------

def _constraints(name, nodes):
    whitelist = [(nodes[0], nodes[1])]
    blacklist = [(nodes[1], nodes[2]), (nodes[2], nodes[0])]
    return {
        "none": {},
        "whitelist": {"whitelist": whitelist},
        "blacklist": {"blacklist": blacklist},
        "both": {"whitelist": whitelist, "blacklist": blacklist},
    }[name]


@pytest.mark.parametrize(
    "case", _records("learn"),
    ids=lambda c: (f"{c['dataset']}-{c['algorithm']}-{c['score']}"
                   f"-{c['constraint']}-maxp{c['maxp']}"))
def test_score_based_learning_with_arguments_crossed(case, sweep_data):
    data = sweep_data[case["dataset"]]
    search = getattr(pybnlearn, case["algorithm"])
    kwargs = dict(_constraints(case["constraint"], list(data.columns)))
    if case["maxp"] is not None:
        kwargs["maxp"] = int(case["maxp"])

    if not case["ok"]:
        with pytest.raises((ValueError, pybnlearn.BNLearnError)):
            search(data, score=case["score"], **kwargs)
        return

    learned = search(data, score=case["score"], **kwargs)

    if case["dataset"] not in ILL_CONDITIONED:
        assert _arcs(learned.arcs) == _arcs(case["arcs"])
        return

    # On the ill-conditioned sets the scores differ in the fourth digit --
    # see the module comment -- so ties between equally-scoring networks
    # break differently, and demanding the same arcs would be demanding that
    # rounding noise agree.  What has to agree is the answer: the same
    # equivalence class, at the same score.  Anything weaker would pass on
    # two unrelated networks.
    expected = pybnlearn.BayesianNetwork(list(data.columns),
                                         [tuple(a) for a in case["arcs"]])
    assert pybnlearn.shd(learned, expected) == 0
    assert pybnlearn.score(learned, data, type=case["score"]) == pytest.approx(
        pybnlearn.score(expected, data, type=case["score"]), **LOOSE)


def test_the_crossed_arguments_actually_conflict():
    """The whitelist wants an arc; maxp=1 sometimes leaves no room for it.
    If no case in the sweep ever hit that, crossing the arguments would not
    be testing anything the one-at-a-time fixtures do not."""
    by_key = {}
    for case in _records("learn"):
        key = (case["dataset"], case["algorithm"], case["score"],
               case["constraint"])
        by_key.setdefault(key, {})[case["maxp"]] = tuple(
            _arcs(case["arcs"]))

    constrained = [k for k, v in by_key.items()
                   if k[3] != "none" and len(set(v.values())) > 1]

    assert constrained, "no crossing of constraint and maxp changed anything"


def test_the_ill_conditioned_disagreement_is_not_a_defect(sweep_data):
    """Why "collinear" gets a loose tolerance, worked out rather than asserted.

    The C is bnlearn's own and the input bits are identical, so a difference
    can only enter below the C -- in the BLAS and LAPACK, which are not the
    same library on both sides.  The question is whether that explains the
    size of it.

    So compute the partial correlation the two disagree about in exact
    rational arithmetic and use it as a referee.  If one of them were simply
    wrong it would sit far outside the forward error bound while the other
    sat inside; instead both are inside it, and neither is meaningfully
    closer to the truth.  The data do not determine those digits.
    """
    from fractions import Fraction

    data = sweep_data["collinear"]
    columns = list(data.columns)
    n = len(data)

    exact = {c: [Fraction(float(v)) for v in data[c]] for c in columns}

    def covariance(a, b):
        mean_a = sum(exact[a]) / n
        mean_b = sum(exact[b]) / n
        return sum((x - mean_a) * (y - mean_b)
                   for x, y in zip(exact[a], exact[b])) / (n - 1)

    matrix = [[covariance(a, b) for b in columns] for a in columns]

    # Gauss-Jordan over the rationals: no rounding anywhere.
    size = len(matrix)
    augmented = [row[:] + [Fraction(int(i == j)) for j in range(size)]
                 for i, row in enumerate(matrix)]
    for i in range(size):
        pivot_row = max(range(i, size), key=lambda r: abs(augmented[r][i]))
        augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]
        pivot = augmented[i][i]
        augmented[i] = [v / pivot for v in augmented[i]]
        for r in range(size):
            if r != i and augmented[r][i]:
                factor = augmented[r][i]
                augmented[r] = [v - factor * w
                                for v, w in zip(augmented[r], augmented[i])]
    inverse = [row[size:] for row in augmented]

    a, c = columns.index("A"), columns.index("C")
    truth = float(-inverse[a][c]) / float(
        inverse[a][a] * inverse[c][c]) ** 0.5

    r_value = [case for case in _records("citest")
               if case["dataset"] == "collinear" and case["test"] == "cor"
               and case["x"] == "A" and case["y"] == "C"
               and len(case["sx"]) == 2][0]
    from_r = _float(r_value["statistic"])
    from_here = list(pybnlearn.ci_test(
        data, "A", "C", sx=r_value["sx"], test="cor")["statistic"].values())[0]

    # kappa * eps, the most any correctly implemented method can promise.
    condition = np.linalg.cond(np.corrcoef(data.to_numpy().T))
    bound = condition * np.finfo(float).eps

    assert condition > 1e11, "the collinear set is no longer ill-conditioned"
    assert abs(from_r - truth) / abs(truth) < bound
    assert abs(from_here - truth) / abs(truth) < bound

    # and the two differ by no more than that bound either.
    assert abs(from_here - from_r) / abs(truth) < bound


def test_the_well_conditioned_data_agree_to_full_precision(sweep_data):
    """The loose tolerance above must not be quietly covering everything.

    Every Gaussian comparison on a data set that is *not* ill-conditioned is
    held to 1e-10, and this says so independently of the parametrisation, so
    that widening ILL_CONDITIONED later would fail here.
    """
    checked = 0
    for case in _records("citest"):
        if case["dataset"] in ILL_CONDITIONED or not case["ok"]:
            continue
        if case["test"] not in ("cor", "zf", "mi-g", "mi-g-sh"):
            continue
        got = pybnlearn.ci_test(sweep_data[case["dataset"]], case["x"],
                                case["y"], sx=case["sx"] or None,
                                test=case["test"])
        assert list(got["statistic"].values())[0] == pytest.approx(
            _float(case["statistic"]), rel=1e-10, abs=1e-10)
        checked += 1

    assert checked >= 70, "too few well-conditioned Gaussian cases to matter"


@pytest.mark.parametrize(
    "case", _records("impossible"),
    ids=lambda c: f"{c['dataset']}-{c['name']}")
def test_combinations_with_no_solution(case, sweep_data):
    """Six ways for a call to be unsatisfiable, each for a different reason.

    Without these the failure paths in this file would be dead code: every
    other combination the sweep generates is satisfiable, so nothing would
    check that this refuses what R refuses -- or, in one case, that it
    *accepts* what R accepts.
    """
    data = sweep_data[case["dataset"]]
    nodes = list(data.columns)
    search = getattr(pybnlearn, case["algorithm"])

    kwargs = {
        "cyclic-whitelist": {"whitelist": [(nodes[0], nodes[1]),
                                           (nodes[1], nodes[0])]},
        "contradiction": {"whitelist": [(nodes[0], nodes[1])],
                          "blacklist": [(nodes[0], nodes[1])]},
        "wrong-score": {"score": "bge" if _is_discrete(data) else "bde"},
        "wrong-test": {"test": "cor" if _is_discrete(data) else "mi"},
        "alpha-too-big": {"alpha": 1.5},
        "unknown-node": {"whitelist": [("nonesuch", nodes[0])]},
    }[case["name"]]

    if case["ok"]:
        # Whitelisting and blacklisting the same arc is *not* a contradiction
        # in bnlearn: the whitelist wins and the blacklist entry is dropped,
        # so the call succeeds and the arc is present.  Refusing it here
        # would be stricter than R rather than a port of it.
        learned = search(data, **kwargs)
        assert _arcs(learned.arcs) == _arcs(case["arcs"])
        return

    with pytest.raises((ValueError, TypeError, pybnlearn.BNLearnError)):
        search(data, **kwargs)


def _is_discrete(data):
    return all(isinstance(data[c].dtype, pd.CategoricalDtype)
               for c in data.columns)


def test_the_impossible_cases_are_mostly_impossible():
    """If a later change made every one of them succeed, the test above
    would still pass and check nothing."""
    cases = _records("impossible")
    refused = [c for c in cases if not c["ok"]]

    assert len(cases) == 60
    assert len(refused) == 50, "R's refusals changed; the fixture is stale"
    assert {c["name"] for c in cases if c["ok"]} == {"contradiction"}


# ---------------------------------------------------------------------------
# constraint-based learning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("constraint"),
    ids=lambda c: (f"{c['dataset']}-{c['algorithm']}-{c['test']}"
                   f"-a{c['alpha']}-{'und' if c['undirected'] else 'dir'}"))
def test_every_constraint_algorithm_against_every_test(case, sweep_data):
    data = sweep_data[case["dataset"]]
    algorithm = getattr(pybnlearn, case["algorithm"].replace(".", "_"))

    kwargs = dict(test=case["test"], alpha=_float(case["alpha"]),
                  undirected=case["undirected"])

    if not case["ok"]:
        with pytest.raises((ValueError, pybnlearn.BNLearnError)):
            algorithm(data, **kwargs)
        return

    learned = algorithm(data, **kwargs)
    assert _arcs(learned.arcs) == _arcs(case["arcs"])


# ---------------------------------------------------------------------------
# parameter learning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("fit"),
    ids=lambda c: f"{c['dataset']}-{c['method']}-iss{c['iss']}")
def test_parameter_learning_over_awkward_data(case, sweep_data):
    """Every number in every node, in R's own storage order: a table with an
    axis in the wrong place reads back correctly and samples from a
    different distribution."""
    data = sweep_data[case["dataset"]]
    network = _structure("star", case["nodes"])

    kwargs = {"method": case["method"]}
    if case["iss"] is not None:
        kwargs["iss"] = _float(case["iss"])

    if not case["ok"]:
        with pytest.raises((ValueError, pybnlearn.BNLearnError)):
            pybnlearn.fit(network, data, **kwargs)
        return

    fitted = pybnlearn.fit(network, data, **kwargs)

    # A contingency-table cell no observation reaches gives 0/0.  R stores
    # that as NA, the generator writes it as null, and here it is a NaN --
    # the same fact in three notations, and on data this sparse it is most
    # of the table rather than a corner case.
    def expected_values(values):
        return [np.nan if v is None else _float(v) for v in values]

    for node, expected in case["params"].items():
        parameters = fitted[node]
        if hasattr(parameters, "probabilities"):
            got = np.asarray(parameters.probabilities).ravel(order="F")
        else:
            # A Gaussian node's coefficients are a dict keyed by parent name,
            # the intercept first, which is the order R stores the vector in.
            # A conditional Gaussian node's are an array, one column per
            # configuration of the discrete parents.
            coefficients = parameters.coefficients
            coefficients = (
                np.array(list(coefficients.values()), dtype=float)
                if isinstance(coefficients, dict)
                else np.asarray(coefficients, dtype=float).ravel(order="F"))
            got = np.concatenate([
                coefficients,
                np.atleast_1d(np.asarray(parameters.sd, dtype=float))
                  .ravel(order="F")])
        assert got == pytest.approx(
            expected_values(expected), nan_ok=True,
            **_tolerance(case["dataset"])), node

    assert pybnlearn.nparams(fitted, data) == int(case["nparams"])


# ---------------------------------------------------------------------------
# the graph utilities the runtime coverage count found unreached
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _records("graph"), ids=lambda c: c["name"])
def test_the_graph_utilities(case):
    network = _network(case["nodes"], case["arcs"])

    assert pybnlearn.narcs(network) == int(case["narcs"])
    assert pybnlearn.nnodes(network) == int(case["nnodes"])
    assert pybnlearn.directed(network) is case["directed"]
    assert pybnlearn.acyclic(network) is case["acyclic"]
    assert _arcs(pybnlearn.arcs(network)) == _arcs(case["arcs"])
    assert pybnlearn.node_ordering(network) == case["ordering"]

    assert _arcs(pybnlearn.cpdag(network).arcs) == _arcs(case["cpdag"])
    assert _arcs(pybnlearn.moral(network).arcs) == _arcs(case["moral"])
    assert _arcs(pybnlearn.skeleton(network).arcs) == _arcs(case["skeleton"])


@pytest.mark.parametrize("case", _records("graph"), ids=lambda c: c["name"])
def test_compelled_and_reversible_arcs(case):
    """An arc is compelled when every member of the equivalence class points
    it the same way, and reversible otherwise -- so the two partition the
    equivalence class's arcs and neither can be checked alone."""
    equivalence_class = pybnlearn.cpdag(_network(case["nodes"], case["arcs"]))

    compelled = _arcs(pybnlearn.compelled_arcs(equivalence_class))
    reversible = _arcs(pybnlearn.reversible_arcs(equivalence_class))

    assert compelled == _arcs(case["compelled"])
    assert reversible == _arcs(case["reversible"])
    assert sorted(compelled + reversible) == _arcs(equivalence_class.arcs)


@pytest.mark.parametrize("case", _records("graph"), ids=lambda c: c["name"])
def test_connected_components(case):
    network = _network(case["nodes"], case["arcs"])

    components = pybnlearn.connected_components(
        pybnlearn.skeleton(network))

    assert len(components) == int(case["components"])
    assert [part for part, _chordal in components] == case["component.nodes"]

    # A one-node component has to be a list holding one name rather than the
    # bare name, or iterating a component gives letters.  R gets away with
    # the bare name because a length-one character vector is still a vector.
    assert all(isinstance(part, list) for part, _chordal in components)


def test_the_components_are_not_a_partition():
    """Worth saying, because it is the obvious assumption and it is wrong.

    For the chain A-B-C-D-E, bnlearn returns two *overlapping* sets rather
    than the single component the name suggests -- so an assertion that every
    node appears exactly once would fail against R itself.  This package
    reproduces what bnlearn does, not what the name implies.
    """
    chain = [c for c in _records("graph") if c["name"] == "chain"][0]

    parts = chain["component.nodes"]
    flattened = [node for part in parts for node in part]

    assert len(parts) > 1
    assert len(flattened) > len(set(flattened)), (
        "the chain's components no longer overlap; if bnlearn changed, this "
        "test and the docstring above both need revisiting")


@pytest.mark.parametrize(
    "case", _records("subgraph"),
    ids=lambda c: f"{c['name']}-{len(c['keep'])}")
def test_subgraph(case):
    """Dropping a node has to drop every arc touching it, including the ones
    whose other end is kept."""
    graph = {g["name"]: g for g in _records("graph")}[case["name"]]
    network = _network(graph["nodes"], graph["arcs"])

    if not case["ok"]:
        with pytest.raises((ValueError, pybnlearn.BNLearnError)):
            pybnlearn.subgraph(network, case["keep"])
        return

    got = pybnlearn.subgraph(network, case["keep"])

    assert got.nodes == case["keep"]
    assert _arcs(got.arcs) == _arcs(case["arcs"])


# ---------------------------------------------------------------------------
# what a learned network remembers about how it was learned
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _records("metadata"),
    ids=lambda c: f"{c['dataset']}-{c['name']}")
def test_the_learning_metadata_matches_r(case, sweep_data):
    """`blacklist` is the interesting one: it is not what was passed in.
    Whitelisting an arc implicitly blacklists its reverse, and that implied
    entry is what fixes the direction during orientation -- so a network that
    reported only the caller's own blacklist would be reporting something
    that never constrained the search.
    """
    data = sweep_data[case["dataset"]]
    nodes = list(data.columns)
    search = getattr(pybnlearn, case["algorithm"].replace(".", "_"))

    kwargs = ({"whitelist": [(nodes[0], nodes[1])],
               "blacklist": [(nodes[2], nodes[0])]}
              if case["constrained"] else {})
    learned = search(data, **kwargs)

    assert _arcs(pybnlearn.whitelist(learned) or []) == _arcs(case["whitelist"])
    assert _arcs(pybnlearn.blacklist(learned) or []) == _arcs(case["blacklist"])

    assert pybnlearn.ntests(learned) == int(case["ntests"])


def test_the_work_counter_covers_every_search():
    """The comparison above is only worth as much as its coverage: every
    search has to be in the fixtures, or one of them could quietly start
    doing a different amount of work than R."""
    covered = {c["algorithm"] for c in _records("metadata")}
    assert len(covered) >= 11, "the work counter no longer covers every search"


def test_whitelisting_an_arc_blacklists_its_reverse():
    """Stated on its own so the parametrised comparison above cannot pass by
    agreeing with R that both are empty."""
    constrained = [c for c in _records("metadata") if c["constrained"]]
    assert constrained

    for case in constrained:
        forward = [tuple(a) for a in case["whitelist"]]
        backward = {(b, a) for a, b in forward}
        assert backward <= {tuple(a) for a in case["blacklist"]}, case["name"]
        # ... and the caller's own entry survives alongside the implied one.
        assert len(case["blacklist"]) > len(backward)


def test_the_work_counter_moves():
    """ntests() counts the scores or tests the search evaluated, so it only
    agrees if the search took the same path -- which makes it a sharper
    check on the search than the arc set, where two paths can meet."""
    counts = {(c["dataset"], c["name"]): int(c["ntests"])
              for c in _records("metadata")}

    assert all(n > 0 for n in counts.values())
    assert len(set(counts.values())) > 5, "the counter is not discriminating"
