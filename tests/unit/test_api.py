"""Behaviour of the Python API that is not about matching R's numbers.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import numpy as np
import pandas as pd
import pytest

import pybnlearn


@pytest.fixture
def discrete():
    rng = np.random.default_rng(0)
    a = rng.choice(["x", "y"], size=300)
    b = np.where(rng.random(300) < 0.8, a, np.where(a == "x", "y", "x"))
    c = rng.choice(["p", "q", "r"], size=300)
    return pd.DataFrame({"A": a, "B": b, "C": c}, dtype="category")


def test_missing_values_are_rejected(discrete):
    """A missing value reaches the discrete scores as INT_MIN and is used as a
    contingency-table index, so it must never get that far."""
    holed = discrete.copy()
    holed["A"] = holed["A"].cat.add_categories(["z"])
    holed.loc[0, "A"] = None

    with pytest.raises(ValueError, match="missing values"):
        pybnlearn.hc(holed)


def test_missing_value_message_mentions_the_pandas_trap(discrete):
    """Categories that look like NA markers are the usual cause, and the error
    should say so -- this bit us on bnlearn's own insurance data set, where a
    category is literally called "None"."""
    holed = discrete.copy()
    holed["B"] = holed["B"].cat.add_categories(["z"])
    holed.loc[5, "B"] = None

    with pytest.raises(ValueError, match="keep_default_na"):
        pybnlearn.hc(holed)


def test_category_named_none_survives_a_csv_round_trip(tmp_path, discrete):
    frame = discrete.copy()
    frame["A"] = frame["A"].cat.rename_categories({"x": "None"})
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)

    reloaded = pd.read_csv(path, dtype="category",
                           keep_default_na=False, na_values=[])

    assert "None" in list(reloaded["A"].cat.categories)
    assert not reloaded.isna().any().any()
    pybnlearn.hc(reloaded)          # must not raise


def test_unknown_score_is_reported_clearly(discrete):
    with pytest.raises(ValueError, match="not implemented"):
        pybnlearn.hc(discrete, score="nonesuch")


def test_score_rejects_arguments_it_does_not_take(discrete):
    with pytest.raises(ValueError, match="does not take"):
        pybnlearn.hc(discrete, score="bic", iss=10)


def test_continuous_score_on_discrete_data(discrete):
    with pytest.raises(ValueError, match="continuous data"):
        pybnlearn.hc(discrete, score="bic-g")


def test_errors_from_the_c_core_raise_rather_than_crash(discrete):
    """error() in the C core unwinds with longjmp; it has to arrive as a Python
    exception, not as a signal."""
    with pytest.raises(pybnlearn.BNLearnError, match="unknown test"):
        pybnlearn.ci_test(discrete, "A", "B", test="nonesuch")


def test_whitelisted_arc_is_present(discrete):
    learned = pybnlearn.hc(discrete, whitelist=[("C", "A")])
    assert ("C", "A") in learned.arcs


def test_blacklisted_arc_is_absent(discrete):
    learned = pybnlearn.hc(discrete, blacklist=[("A", "B")])
    assert ("A", "B") not in learned.arcs


def test_maxp_limits_the_number_of_parents(discrete):
    learned = pybnlearn.hc(discrete, maxp=1)
    for node in learned.nodes:
        assert len(learned.parents(node)) <= 1


def test_repeated_runs_do_not_grow_memory(discrete):
    """The search allocates into an arena that is freed when it finishes; if
    that stopped happening, this is where it would show."""
    import resource

    def rss_mb():                      # ru_maxrss is bytes on macOS, KiB on Linux
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1e6 if raw > 1e7 else raw / 1e3

    for _ in range(20):
        pybnlearn.hc(discrete)
    baseline = rss_mb()
    for _ in range(200):
        pybnlearn.hc(discrete)

    assert rss_mb() - baseline < 50, "memory grew across repeated searches"


def test_modelstring_round_trips_through_arcs(discrete):
    learned = pybnlearn.hc(discrete)
    assert learned.modelstring().count("[") == len(learned.nodes)
    assert set(learned.topological_order()) == set(learned.nodes)


def test_tabu_returns_the_best_network_not_the_last(discrete):
    """Tabu keeps walking after the score stops improving, so what it returns
    has to be the best network it saw, never wherever the walk ended."""
    learned = pybnlearn.tabu(discrete, tabu=5)
    walked = pybnlearn.hc(discrete)

    assert pybnlearn.score(learned, discrete) >= \
        pybnlearn.score(walked, discrete) - 1e-9


def test_tabu_memory_is_flat_across_runs(discrete):
    """The tabu list holds hashes for the whole search rather than per call,
    which is exactly the kind of thing that leaks if it is built wrong."""
    import resource

    def rss_mb():
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1e6 if raw > 1e7 else raw / 1e3

    for _ in range(20):
        pybnlearn.tabu(discrete, tabu=10)
    baseline = rss_mb()
    for _ in range(200):
        pybnlearn.tabu(discrete, tabu=10)

    assert rss_mb() - baseline < 50, "memory grew across repeated searches"


def _is_undirected(net):
    return all((b, a) in net.arcs for a, b in net.arcs)


@pytest.mark.parametrize("algorithm,expected", [
    ("mmpc", True), ("si_hiton_pc", True),
    ("gs", False), ("iamb", False), ("inter_iamb", False),
    ("iamb_fdr", False),
])
def test_undirected_defaults_match_bnlearn(algorithm, expected, discrete):
    """mmpc and si.hiton.pc learn parents and children without telling them
    apart, so bnlearn returns an undirected graph for those unless asked
    otherwise; the Markov-blanket algorithms orient what they can.  The parity
    fixtures all pass `undirected` explicitly, which is why the default needs
    a test of its own.

    This checks the recorded flag rather than the shape of the result: a CPDAG
    is legitimately all-undirected when the data contain no v-structure, so
    inspecting the arcs would make the test depend on the data.
    """
    net = getattr(pybnlearn, algorithm)(discrete)
    assert net.learning["undirected"] is expected


@pytest.mark.parametrize("algorithm",
                         ["gs", "iamb", "inter_iamb", "iamb_fdr",
                          "mmpc", "si_hiton_pc"])
def test_undirected_true_leaves_nothing_oriented(algorithm, discrete):
    net = getattr(pybnlearn, algorithm)(discrete, undirected=True)
    assert _is_undirected(net)


def test_iamb_fdr_warns_rather_than_looping(discrete):
    """iamb.fdr can revisit a blanket it has already seen; it must break out
    and say so instead of spinning."""
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")          # any infinite-loop warning becomes an
        pybnlearn.iamb_fdr(discrete)     # error, but it must still terminate


def test_every_public_name_is_exported():
    """__all__ is the documented API surface, and it is edited by hand.

    Two functions shipped importable but missing from it before this test
    existed, which meant they were absent from `from pybnlearn import *`, from
    the docs generated off __all__, and from anything else that trusts it.
    """
    public = {
        name for name in dir(pybnlearn)
        if not name.startswith("_")
        and getattr(getattr(pybnlearn, name), "__module__", "")
        .startswith("pybnlearn")
    }
    missing = public - set(pybnlearn.__all__)

    assert not missing, (
        "importable but not in __all__: " + ", ".join(sorted(missing)))


def test_everything_exported_exists():
    missing = [name for name in pybnlearn.__all__
               if not hasattr(pybnlearn, name)]
    assert not missing, "in __all__ but not importable: " + ", ".join(missing)
