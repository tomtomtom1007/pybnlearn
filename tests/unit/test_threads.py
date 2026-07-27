"""Calling the C core from several threads at once.

bnlearn's C keeps its state in process-wide statics -- the arena results are
allocated in, the preserved-object list, the interned symbol table, the
random number generator, and the jmp_buf error() unwinds to.  None of it is
per-thread, and the GIL does not make it safe: the GIL is released at
bytecode boundaries, so a second thread can open an arena frame between this
thread's push and its pop and the two frames interleave.  Before the core
lock that was a segfault, which is the worst way for a library to fail --
the process dies with no traceback and nothing to attribute it to.

Serialising is bnlearn's own model rather than a compromise.  R cannot be
called from two threads at all, the vendored C has no OpenMP and no
pthreads, and bnlearn's `cluster` argument takes a parallel::makeCluster()
cluster, which is separate R *processes*.  ProcessPoolExecutor is the Python
equivalent and needs no lock at all.

These tests are worth their runtime because the failure they guard against
cannot be caught any other way: a crashed interpreter produces no test
failure to read, only an exit code, so the regression has to be reproduced
deliberately.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import concurrent.futures as futures
import pathlib
import threading

import numpy as np
import pandas as pd
import pytest

import pybnlearn

FIXTURES = pathlib.Path(__file__).parent.parent / "parity" / "fixtures"


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIXTURES / "learning.test.csv", dtype="category")


@pytest.fixture(scope="module")
def network(data):
    return pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]")


def _concurrently(fn, repeats=24, workers=4):
    """Run `fn` on several threads and give back every result.

    Any exception is re-raised by .result(); a crash cannot be caught here at
    all, which is the point -- if the lock regresses, the whole test process
    dies rather than this failing.
    """
    with futures.ThreadPoolExecutor(workers) as pool:
        return [f.result()
                for f in [pool.submit(fn) for _ in range(repeats)]]


# ---------------------------------------------------------------------------
# the crash
# ---------------------------------------------------------------------------

def test_scoring_from_several_threads(data, network):
    """The original reproducer: four threads scoring the same network killed
    the interpreter with a heap-corruption abort inside 24 calls."""
    got = _concurrently(lambda: pybnlearn.score(network, data))

    assert len(set(got)) == 1
    assert got[0] == pytest.approx(pybnlearn.score(network, data))


def test_structure_learning_from_several_threads(data):
    got = _concurrently(lambda: pybnlearn.hc(data).modelstring())

    assert set(got) == {pybnlearn.hc(data).modelstring()}


def test_a_search_session_holds_the_lock_for_its_lifetime(data):
    """hc() keeps one arena frame open across the whole search, so a second
    thread must not open one in the middle of it.  Running hc and tabu
    together is the case where that would show: they interleave differently
    every time."""
    def mixed(i):
        return (pybnlearn.hc(data) if i % 2 else pybnlearn.tabu(data)).narcs

    with futures.ThreadPoolExecutor(4) as pool:
        got = list(pool.map(mixed, range(24)))

    assert set(got) == {pybnlearn.hc(data).narcs}


def test_the_whole_surface_survives_concurrent_use(data, network):
    """Different entry points at once, rather than the same one repeatedly:
    the arena frames they open are different shapes and sizes."""
    fitted = pybnlearn.fit(network, data)
    calls = [
        lambda: pybnlearn.score(network, data),
        lambda: pybnlearn.hc(data).narcs,
        lambda: pybnlearn.tabu(data).narcs,
        lambda: pybnlearn.gs(data).narcs,
        lambda: pybnlearn.pc_stable(data).narcs,
        lambda: len(pybnlearn.fit(network, data)["B"].probabilities),
        lambda: pybnlearn.cpdag(network).narcs,
        lambda: float(pybnlearn.query(fitted, "B", {"A": "a"}).values.sum()),
        lambda: float(pybnlearn.arc_strength(
            network, data, criterion="bic")["strength"].sum()),
        lambda: int(pybnlearn.amat(network).to_numpy().sum()),
    ]

    with futures.ThreadPoolExecutor(8) as pool:
        results = [pool.submit(calls[i % len(calls)]) for i in range(80)]
        got = [f.result() for f in results]

    assert len(got) == 80


def test_an_error_on_one_thread_does_not_disturb_another(data, network):
    """error() unwinds through a single process-wide jmp_buf.  A thread
    taking that path while another is mid-call was the second way to corrupt
    the arena, and it is not covered by the success paths above."""
    def failing():
        with pytest.raises(ValueError):
            pybnlearn.score(network, data, type="nonesuch")
        return "raised"

    def succeeding():
        return pybnlearn.score(network, data)

    with futures.ThreadPoolExecutor(4) as pool:
        mixed = [pool.submit(failing if i % 2 else succeeding)
                 for i in range(40)]
        got = [f.result() for f in mixed]

    assert got.count("raised") == 20
    assert set(got) - {"raised"} == {pybnlearn.score(network, data)}


# ---------------------------------------------------------------------------
# what serialising has to preserve
# ---------------------------------------------------------------------------

def test_the_lock_is_released_when_a_call_raises(data, network):
    """A lock leaked on the error path would not crash -- it would hang the
    next caller forever, which is harder to diagnose than a crash."""
    for _ in range(5):
        with pytest.raises(ValueError):
            pybnlearn.score(network, data, type="nonesuch")

    done = threading.Event()

    def later():
        pybnlearn.score(network, data)
        done.set()

    thread = threading.Thread(target=later, daemon=True)
    thread.start()
    thread.join(timeout=30)

    assert done.is_set(), "the core lock was not released on the error path"


def test_the_lock_is_released_when_a_session_raises(data):
    """The same thing one level up: hc() holds a session frame open, and a
    failure inside the search must still close it."""
    for _ in range(3):
        with pytest.raises(ValueError):
            pybnlearn.hc(data, start=pybnlearn.model2network("[A][B|A]"))

    done = threading.Event()

    def later():
        pybnlearn.hc(data)
        done.set()

    thread = threading.Thread(target=later, daemon=True)
    thread.start()
    thread.join(timeout=60)

    assert done.is_set(), "the core lock was not released by a failed session"


def test_nested_calls_do_not_deadlock(data):
    """The lock has to be re-entrant: bn_cv() takes it, and every fold it
    runs takes it again."""
    assert pybnlearn.bn_cv(data.head(300), "hc", k=2).mean is not None


def test_seeded_draws_are_not_interleaved(data):
    """The generator is process-wide.  A bootstrap whose draws were spliced
    with another thread's would still return a sample -- just not the sample
    R's generator produces, which is the entire basis of the parity suite."""
    from pybnlearn._core import sample_indices

    pybnlearn.set_seed(1)
    expected = sample_indices(100, 20, replace=True).tolist()

    def draw():
        pybnlearn.set_seed(1)
        return sample_indices(100, 20, replace=True).tolist()

    got = _concurrently(draw, repeats=16)

    assert all(sequence == expected for sequence in got)


def test_single_threaded_use_is_unaffected(data, network):
    """The lock is uncontended in the ordinary case and must not change any
    answer."""
    assert pybnlearn.hc(data).modelstring() == "[A][C][F][B|A][D|A:C][E|B:F]"
    assert pybnlearn.score(network, data) == pytest.approx(-24006.73423249815)
