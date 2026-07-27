"""Check what pybnlearn *refuses* against what R refuses.

Every other file in this directory compares an answer with R's answer, which
means every one of them is built from inputs R was willing to accept.  That
leaves a blind spot exactly the size of this file: when pybnlearn accepts
something R rejects there is no R answer to compare against, so no amount of
parity fixtures can see it.

What lives in that blind spot is not crashes.  A malformed argument that
crashes announces itself.  The ones that cost you a week are the arguments
that come back as a *result*: `maxp = 0` returning the empty network, `Inf`
in a column making every Gaussian score NaN so the search accepts no move
and returns the empty network, a `start=` network over other variables
learned quietly over the overlap.  None of those look like failures.  All of
them were found by hand before this file existed, which is the argument for
having it.

The grid is generated rather than written: tools/gen_r_rejection_fixtures.R
walks a few hundred deliberately malformed calls, records whether R stopped,
and writes fixtures/rejections.json.  This file maps each of those ids to
the equivalent pybnlearn call and requires the two to agree about whether
there is an answer at all.

Both directions are checked.  Over-rejecting is also a parity bug, and a
nastier one to diagnose, because the traceback names an argument that was
fine: R's `is.probability()` is a *closed* interval, so `alpha = 0` is legal
and refusing it would break working code.

The mapping is one dict of thunks plus two small annotations on it.
NOT_PORTED lists the behaviours R has that this does not, and so has no call
to make; STRICTER lists the places R accepts something and returns nonsense
for it, where refusing is the better answer and the expectation is inverted
rather than dropped.  `test_every_case_is_accounted_for` requires every
fixture id to be covered by one of the three, so coverage cannot rot quietly
when the generator grows a new row.

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


def _records():
    path = FIXTURES / "rejections.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# behaviours R has that this does not, and so cannot be asked to reproduce
#
# Kept as data rather than deleted from the generator: the R side of each is
# a real answer, and knowing which of R's answers are missing is worth more
# than a shorter fixture file.
# ---------------------------------------------------------------------------

NOT_PORTED = {
    "runs/bn.cv/zero": "bn_cv() has no runs= argument; a run is one call",
    "runs/bn.cv/negative": "bn_cv() has no runs= argument",
    "runs/bn.cv/fractional": "bn_cv() has no runs= argument",
    "data/hc/missing-values":
        "R falls back to the node-average-likelihood scores (pnal, pnal-g) "
        "for incomplete data; none of those are wired up here, so the data "
        "are refused instead of scored by a score that does not exist",
    "data/hc/nan":
        "a NaN counts as missing to R too, so this is the same gap as "
        "data/hc/missing-values",
}

# ---------------------------------------------------------------------------
# where R accepts and this does not, on purpose
#
# Each of these is a place bnlearn forgets a check it makes everywhere else,
# and returns something malformed rather than an error.  Reproducing the
# omission would mean reproducing the malformed output, which is worse than
# diverging.
# ---------------------------------------------------------------------------

STRICTER = {
    "model2network/duplicate-node":
        "R accepts '[A][A]' and returns a graph whose node list contains A "
        "twice; its adjacency matrix then cannot be indexed by name",
    "explanatory/naive.bayes/unknown":
        "R builds the classifier over a node the data do not have, and only "
        "fails later when something tries to read that column",
    "whitelist/hc/self-loop":
        "bnlearn refuses loops everywhere it calls check.arcs() and misses "
        "the whitelist path; R carries the loop into the search, where it "
        "does not change the arcs but does shift the topological ordering "
        "the model string is written in",
    "whitelist/gs/self-loop": "as whitelist/hc/self-loop",
}


@pytest.fixture(scope="module")
def cases(datasets):
    """id -> a thunk making the same call the generator made in R.

    Built once: several entries need a fitted network or a set of bootstrap
    strengths, and rebuilding those per test would dominate the runtime.
    """
    lt = datasets["learning.test"]                # discrete,   5000 x 6
    gt = datasets["gaussian.test"]                # continuous, 5000 x 7
    short = lt.head(20)                           # for the fold arithmetic

    dag = pybnlearn.model2network("[A][C][F][B|A][D|A:C][E|B:F]")
    fitted = pybnlearn.fit(dag, lt)
    gdag = pybnlearn.model2network("[A][B][E][G][C|A:B][D|B][F|A:D:E:G]")

    foreign = pybnlearn.empty_graph(["X", "Y", "Z"])
    subset = pybnlearn.empty_graph(["A", "B", "C"])
    tri = pybnlearn.model2network("[A][B|A][C|A:B]")

    strength = pybnlearn.boot_strength(short, algorithm="hc", replicates=5)

    def spoil(frame, value, column="A"):
        out = frame.copy()
        out.loc[out.index[0], column] = value
        return out

    def amat_with(net, edits):
        matrix = pybnlearn.amat(net).copy()
        for (row, column), value in edits.items():
            matrix.loc[row, column] = value
        return matrix

    loop, unknown = [("A", "A")], [("Z", "A")]
    cycle = [("A", "B"), ("B", "C"), ("C", "A")]
    both = [("A", "B"), ("B", "A")]

    return {
        # -- alpha: is.probability(), a closed interval ---------------------
        "alpha/gs/zero": lambda: pybnlearn.gs(lt, alpha=0),
        "alpha/gs/one": lambda: pybnlearn.gs(lt, alpha=1),
        "alpha/gs/negative": lambda: pybnlearn.gs(lt, alpha=-0.1),
        "alpha/gs/above-one": lambda: pybnlearn.gs(lt, alpha=1.5),
        "alpha/gs/nan": lambda: pybnlearn.gs(lt, alpha=float("nan")),
        "alpha/gs/infinite": lambda: pybnlearn.gs(lt, alpha=float("inf")),
        "alpha/gs/missing": lambda: pybnlearn.gs(lt, alpha=None),
        "alpha/gs/string": lambda: pybnlearn.gs(lt, alpha="0.05"),
        "alpha/gs/vector": lambda: pybnlearn.gs(lt, alpha=[0.01, 0.05]),
        "alpha/iamb/negative": lambda: pybnlearn.iamb(lt, alpha=-0.1),
        "alpha/iamb.fdr/negative": lambda: pybnlearn.iamb_fdr(lt, alpha=-0.1),
        "alpha/inter.iamb/negative":
            lambda: pybnlearn.inter_iamb(lt, alpha=-0.1),
        "alpha/fast.iamb/negative":
            lambda: pybnlearn.fast_iamb(lt, alpha=-0.1),
        "alpha/mmpc/negative": lambda: pybnlearn.mmpc(lt, alpha=-0.1),
        "alpha/pc.stable/negative":
            lambda: pybnlearn.pc_stable(lt, alpha=-0.1),
        "alpha/si.hiton.pc/negative":
            lambda: pybnlearn.si_hiton_pc(lt, alpha=-0.1),
        "alpha/hpc/negative": lambda: pybnlearn.hpc(lt, alpha=-0.1),
        "alpha/learn.mb/negative":
            lambda: pybnlearn.learn_mb(lt, node="A", method="gs", alpha=-0.1),
        "alpha/learn.nbr/negative":
            lambda: pybnlearn.learn_nbr(lt, node="A", method="mmpc",
                                        alpha=-0.1),
        "alpha/arc.strength/negative":
            lambda: pybnlearn.arc_strength(dag, lt, criterion="mi",
                                           alpha=-0.1),
        "alpha/arc.strength/above-one":
            lambda: pybnlearn.arc_strength(dag, lt, criterion="mi", alpha=1.5),
        "alpha/mmhc/negative":
            lambda: pybnlearn.mmhc(lt, restrict_args={"alpha": -0.1}),
        "alpha/rsmax2/negative":
            lambda: pybnlearn.rsmax2(lt, restrict_args={"alpha": -0.1}),
        "alpha/h2pc/negative":
            lambda: pybnlearn.h2pc(lt, restrict_args={"alpha": -0.1}),

        # -- max.sx: the bound on the conditioning set ----------------------
        "max.sx/gs/zero": lambda: pybnlearn.gs(lt, max_sx=0),
        "max.sx/gs/negative": lambda: pybnlearn.gs(lt, max_sx=-1),
        "max.sx/gs/fractional": lambda: pybnlearn.gs(lt, max_sx=1.5),
        "max.sx/gs/infinite": lambda: pybnlearn.gs(lt, max_sx=float("inf")),
        "max.sx/gs/above-nnodes": lambda: pybnlearn.gs(lt, max_sx=100),
        "max.sx/pc.stable/zero": lambda: pybnlearn.pc_stable(lt, max_sx=0),
        "max.sx/si.hiton.pc/zero":
            lambda: pybnlearn.si_hiton_pc(lt, max_sx=0),
        "max.sx/learn.nbr/zero":
            lambda: pybnlearn.learn_nbr(lt, node="A", method="si.hiton.pc",
                                        max_sx=0),

        # -- the counters that steer the search -----------------------------
        "maxp/hc/zero": lambda: pybnlearn.hc(lt, maxp=0),
        "maxp/hc/negative": lambda: pybnlearn.hc(lt, maxp=-1),
        "maxp/hc/fractional": lambda: pybnlearn.hc(lt, maxp=1.5),
        "maxp/hc/infinite": lambda: pybnlearn.hc(lt, maxp=float("inf")),
        "maxp/hc/above-nnodes": lambda: pybnlearn.hc(lt, maxp=100),
        "maxp/tabu/zero": lambda: pybnlearn.tabu(lt, maxp=0),
        "maxp/tabu/negative": lambda: pybnlearn.tabu(lt, maxp=-1),
        "maxp/tabu/fractional": lambda: pybnlearn.tabu(lt, maxp=1.5),
        "max.iter/hc/zero": lambda: pybnlearn.hc(lt, max_iter=0),
        "max.iter/hc/negative": lambda: pybnlearn.hc(lt, max_iter=-1),
        "max.iter/hc/fractional": lambda: pybnlearn.hc(lt, max_iter=1.5),
        "max.iter/hc/infinite":
            lambda: pybnlearn.hc(lt, max_iter=float("inf")),
        "max.iter/tabu/zero": lambda: pybnlearn.tabu(lt, max_iter=0),
        "max.iter/tabu/negative": lambda: pybnlearn.tabu(lt, max_iter=-1),
        "restart/hc/zero": lambda: pybnlearn.hc(lt, restart=0),
        "restart/hc/negative": lambda: pybnlearn.hc(lt, restart=-1),
        "restart/hc/fractional": lambda: pybnlearn.hc(lt, restart=1.5),
        "perturb/hc/zero": lambda: pybnlearn.hc(lt, restart=2, perturb=0),
        "perturb/hc/negative": lambda: pybnlearn.hc(lt, restart=2, perturb=-1),
        "perturb/hc/fractional":
            lambda: pybnlearn.hc(lt, restart=2, perturb=1.5),
        "tabu/tabu/zero": lambda: pybnlearn.tabu(lt, tabu=0),
        "tabu/tabu/negative": lambda: pybnlearn.tabu(lt, tabu=-1),
        "tabu/tabu/fractional": lambda: pybnlearn.tabu(lt, tabu=1.5),

        # -- whitelists and blacklists ---------------------------------------
        "whitelist/hc/self-loop": lambda: pybnlearn.hc(lt, whitelist=loop),
        "blacklist/hc/self-loop": lambda: pybnlearn.hc(lt, blacklist=loop),
        "whitelist/gs/self-loop": lambda: pybnlearn.gs(lt, whitelist=loop),
        "blacklist/gs/self-loop": lambda: pybnlearn.gs(lt, blacklist=loop),
        "whitelist/hc/unknown-node":
            lambda: pybnlearn.hc(lt, whitelist=unknown),
        "blacklist/hc/unknown-node":
            lambda: pybnlearn.hc(lt, blacklist=[("A", "Z")]),
        "whitelist/gs/unknown-node":
            lambda: pybnlearn.gs(lt, whitelist=unknown),
        "blacklist/gs/unknown-node":
            lambda: pybnlearn.gs(lt, blacklist=[("A", "Z")]),
        "whitelist/tabu/unknown-node":
            lambda: pybnlearn.tabu(lt, whitelist=unknown),
        "whitelist/pc.stable/unknown-node":
            lambda: pybnlearn.pc_stable(lt, whitelist=unknown),
        "whitelist/mmhc/unknown-node":
            lambda: pybnlearn.mmhc(lt, whitelist=unknown),
        "whitelist/chow.liu/unknown-node":
            lambda: pybnlearn.chow_liu(lt, whitelist=unknown),
        "whitelist/aracne/unknown-node":
            lambda: pybnlearn.aracne(lt, whitelist=unknown),
        "whitelist/hc/cycle": lambda: pybnlearn.hc(lt, whitelist=cycle),
        "whitelist/tabu/cycle": lambda: pybnlearn.tabu(lt, whitelist=cycle),
        "whitelist/gs/cycle": lambda: pybnlearn.gs(lt, whitelist=cycle),
        "whitelist/hc/both-directions":
            lambda: pybnlearn.hc(lt, whitelist=both),
        "whitelist/tabu/both-directions":
            lambda: pybnlearn.tabu(lt, whitelist=both),

        # -- start= networks over the wrong variables -------------------------
        "start/hc/foreign-nodes": lambda: pybnlearn.hc(lt, start=foreign),
        "start/hc/subset-nodes": lambda: pybnlearn.hc(lt, start=subset),
        "start/tabu/foreign-nodes": lambda: pybnlearn.tabu(lt, start=foreign),
        "start/tabu/subset-nodes": lambda: pybnlearn.tabu(lt, start=subset),
        "start/hc/matching-nodes": lambda: pybnlearn.hc(lt, start=dag),

        # -- the data themselves ----------------------------------------------
        "data/hc/no-rows": lambda: pybnlearn.hc(lt.head(0)),
        "data/gs/no-rows": lambda: pybnlearn.gs(lt.head(0)),
        "data/hc/single-column": lambda: pybnlearn.hc(lt[["A"]]),
        "data/hc/single-level": lambda: pybnlearn.hc(pd.DataFrame({
            "A": pd.Categorical(["a"] * 100),
            "B": lt["B"].iloc[:100].reset_index(drop=True)})),
        # data/hc/missing-values and data/hc/nan have no entry: they are in
        # NOT_PORTED, and a thunk that is never called is a thunk that rots.
        "data/hc/infinite": lambda: pybnlearn.hc(spoil(gt.head(100), np.inf)),
        "data/hc/negative-infinite":
            lambda: pybnlearn.hc(spoil(gt.head(100), -np.inf)),
        "data/hc/large-but-finite":
            lambda: pybnlearn.hc(spoil(gt.head(100), 1e300)),
        "data/score/infinite":
            lambda: pybnlearn.score(gdag, spoil(gt.head(100), np.inf),
                                    type="bic-g"),
        "data/bn.fit/infinite":
            lambda: pybnlearn.fit(gdag, spoil(gt.head(100), np.inf)),

        # -- labels, and the data types they apply to --------------------------
        "score/hc/unknown": lambda: pybnlearn.hc(lt, score="nosuch"),
        "score/hc/discrete-on-continuous":
            lambda: pybnlearn.hc(gt, score="bde"),
        "score/hc/continuous-on-discrete":
            lambda: pybnlearn.hc(lt, score="bge"),
        "score/score/discrete-on-continuous":
            lambda: pybnlearn.score(gdag, gt, type="bde"),
        "score/score/continuous-on-discrete":
            lambda: pybnlearn.score(dag, lt, type="bic-g"),
        "test/gs/unknown": lambda: pybnlearn.gs(lt, test="nosuch"),
        "test/gs/continuous-on-discrete": lambda: pybnlearn.gs(lt, test="cor"),
        "test/gs/discrete-on-continuous": lambda: pybnlearn.gs(gt, test="mi"),
        "criterion/arc.strength/unknown":
            lambda: pybnlearn.arc_strength(dag, lt, criterion="nosuch"),
        "method/learn.mb/unknown":
            lambda: pybnlearn.learn_mb(lt, node="A", method="nosuch"),
        "node/learn.mb/unknown":
            lambda: pybnlearn.learn_mb(lt, node="Z", method="gs"),
        "method/bn.fit/unknown":
            lambda: pybnlearn.fit(dag, lt, method="nosuch"),
        "method/discretize/unknown":
            lambda: pybnlearn.discretize(gt, method="nosuch"),
        "method/predict/unknown":
            lambda: pybnlearn.predict(fitted, node="A", data=lt,
                                      method="nosuch"),

        # -- score hyperparameters ----------------------------------------------
        "iss/score/zero":
            lambda: pybnlearn.score(dag, lt, type="bde", iss=0),
        "iss/score/negative":
            lambda: pybnlearn.score(dag, lt, type="bde", iss=-1),
        "iss/score/fractional":
            lambda: pybnlearn.score(dag, lt, type="bde", iss=0.5),
        "iss/bn.fit/zero":
            lambda: pybnlearn.fit(dag, lt, method="bayes", iss=0),
        "iss/bn.fit/negative":
            lambda: pybnlearn.fit(dag, lt, method="bayes", iss=-1),
        "iss/hc/negative": lambda: pybnlearn.hc(lt, score="bde", iss=-1),
        "k/score/negative": lambda: pybnlearn.score(dag, lt, type="aic", k=-1),
        "prior/score/unknown":
            lambda: pybnlearn.score(dag, lt, type="bde", prior="nosuch"),
        "beta/score/negative":
            lambda: pybnlearn.score(dag, lt, type="bde", prior="vsp", beta=-1),
        "beta/score/above-one":
            lambda: pybnlearn.score(dag, lt, type="bde", prior="vsp", beta=2),

        # -- networks that do not match the data ----------------------------------
        "mismatch/bn.fit/foreign-nodes": lambda: pybnlearn.fit(foreign, lt),
        "mismatch/bn.fit/subset-nodes": lambda: pybnlearn.fit(subset, lt),
        "mismatch/bn.fit/extra-node":
            lambda: pybnlearn.fit(pybnlearn.add_node(dag, "Z"), lt),
        "mismatch/score/foreign-nodes": lambda: pybnlearn.score(foreign, lt),
        "mismatch/score/subset-nodes": lambda: pybnlearn.score(subset, lt),
        "mismatch/score/extra-node":
            lambda: pybnlearn.score(pybnlearn.add_node(dag, "Z"), lt),
        "mismatch/predict/unknown-node":
            lambda: pybnlearn.predict(fitted, node="Z", data=lt),
        "mismatch/predict/missing-parents":
            lambda: pybnlearn.predict(fitted, node="E", data=lt[["A", "E"]]),
        "mismatch/arc.strength/foreign-nodes":
            lambda: pybnlearn.arc_strength(foreign, lt),
        "mismatch/nparams/foreign-nodes":
            lambda: pybnlearn.nparams(foreign, lt),
        "mismatch/bn.cv/foreign-nodes":
            lambda: pybnlearn.bn_cv(short, foreign, k=2),
        "mismatch/impute/foreign-data":
            lambda: pybnlearn.impute(
                fitted, pd.DataFrame({"X": pd.Categorical(["a", "b"])})),

        # -- cross-validation ------------------------------------------------------
        "k/bn.cv/zero": lambda: pybnlearn.bn_cv(short, dag, k=0),
        "k/bn.cv/one": lambda: pybnlearn.bn_cv(short, dag, k=1),
        "k/bn.cv/negative": lambda: pybnlearn.bn_cv(short, dag, k=-1),
        "k/bn.cv/fractional": lambda: pybnlearn.bn_cv(short, dag, k=2.5),
        "k/bn.cv/above-nrow": lambda: pybnlearn.bn_cv(short, dag, k=21),
        "k/bn.cv/equals-nrow": lambda: pybnlearn.bn_cv(short, dag, k=20),
        "m/bn.cv/zero":
            lambda: pybnlearn.bn_cv(short, dag, method="hold-out", m=0),
        "m/bn.cv/negative":
            lambda: pybnlearn.bn_cv(short, dag, method="hold-out", m=-1),
        "m/bn.cv/equals-nrow":
            lambda: pybnlearn.bn_cv(short, dag, method="hold-out", m=20),
        "m/bn.cv/above-nrow":
            lambda: pybnlearn.bn_cv(short, dag, method="hold-out", m=21),
        "method/bn.cv/unknown":
            lambda: pybnlearn.bn_cv(short, dag, method="nosuch"),
        "loss/bn.cv/unknown":
            lambda: pybnlearn.bn_cv(short, dag, k=2, loss="nosuch"),
        "loss/bn.cv/continuous-on-discrete":
            lambda: pybnlearn.bn_cv(short, dag, k=2, loss="cor", target="A"),
        "loss/bn.cv/no-target":
            lambda: pybnlearn.bn_cv(short, dag, k=2, loss="pred"),
        "loss/bn.cv/unknown-target":
            lambda: pybnlearn.bn_cv(short, dag, k=2, loss="pred", target="Z"),
        "folds/bn.cv/single-fold":
            lambda: pybnlearn.bn_cv(short, dag, method="custom-folds",
                                    folds=[list(range(1, 21))]),
        "folds/bn.cv/incomplete":
            lambda: pybnlearn.bn_cv(short, dag, method="custom-folds",
                                    folds=[list(range(1, 6)),
                                           list(range(6, 11))]),
        "folds/bn.cv/overlapping":
            lambda: pybnlearn.bn_cv(short, dag, method="custom-folds",
                                    folds=[list(range(1, 11)),
                                           list(range(5, 21))]),
        "folds/bn.cv/out-of-range":
            lambda: pybnlearn.bn_cv(short, dag, method="custom-folds",
                                    folds=[list(range(1, 11)),
                                           list(range(11, 22))]),

        # -- resampling --------------------------------------------------------------
        "R/boot.strength/zero":
            lambda: pybnlearn.boot_strength(short, algorithm="hc",
                                            replicates=0),
        "R/boot.strength/negative":
            lambda: pybnlearn.boot_strength(short, algorithm="hc",
                                            replicates=-5),
        "R/boot.strength/fractional":
            lambda: pybnlearn.boot_strength(short, algorithm="hc",
                                            replicates=2.5),
        "m/boot.strength/zero":
            lambda: pybnlearn.boot_strength(short, algorithm="hc",
                                            replicates=2, m=0),
        "m/boot.strength/negative":
            lambda: pybnlearn.boot_strength(short, algorithm="hc",
                                            replicates=2, m=-1),
        "algorithm/boot.strength/unknown":
            lambda: pybnlearn.boot_strength(short, algorithm="nosuch",
                                            replicates=2),
        "R/bn.boot/zero":
            lambda: pybnlearn.bn_boot(short, statistic=pybnlearn.narcs,
                                      algorithm="hc", replicates=0),
        "R/bn.boot/negative":
            lambda: pybnlearn.bn_boot(short, statistic=pybnlearn.narcs,
                                      algorithm="hc", replicates=-5),
        "R/bn.boot/fractional":
            lambda: pybnlearn.bn_boot(short, statistic=pybnlearn.narcs,
                                      algorithm="hc", replicates=2.5),

        # -- arc strengths -------------------------------------------------------------
        "threshold/averaged.network/negative":
            lambda: pybnlearn.averaged_network(strength, threshold=-0.1),
        "threshold/averaged.network/above-one":
            lambda: pybnlearn.averaged_network(strength, threshold=1.5),
        "threshold/averaged.network/zero":
            lambda: pybnlearn.averaged_network(strength, threshold=0),
        "threshold/averaged.network/one":
            lambda: pybnlearn.averaged_network(strength, threshold=1),
        "custom.strength/mismatched-nodes":
            lambda: pybnlearn.custom_strength([dag, foreign],
                                              nodes=list(lt.columns)),
        "custom.strength/subset-nodes":
            lambda: pybnlearn.custom_strength([dag, subset],
                                              nodes=list(lt.columns)),
        "custom.strength/unknown-node-set":
            lambda: pybnlearn.custom_strength([dag, dag],
                                              nodes=list(lt.columns) + ["Z"]),
        "custom.strength/weights/negative":
            lambda: pybnlearn.custom_strength([dag, dag],
                                              nodes=list(lt.columns),
                                              weights=[-1, 1]),
        "custom.strength/weights/wrong-length":
            lambda: pybnlearn.custom_strength([dag, dag],
                                              nodes=list(lt.columns),
                                              weights=[1, 1, 1]),

        # -- graph constructors -----------------------------------------------------------
        "empty.graph/no-nodes": lambda: pybnlearn.empty_graph([]),
        "empty.graph/single-node": lambda: pybnlearn.empty_graph(["A"]),
        "empty.graph/duplicate-nodes":
            lambda: pybnlearn.empty_graph(["A", "A"]),
        "empty.graph/empty-label": lambda: pybnlearn.empty_graph(["A", ""]),
        "complete.graph/no-nodes": lambda: pybnlearn.complete_graph([]),
        "complete.graph/single-node": lambda: pybnlearn.complete_graph(["A"]),
        "complete.graph/duplicate-nodes":
            lambda: pybnlearn.complete_graph(["A", "A"]),
        "model2network/reciprocal":
            lambda: pybnlearn.model2network("[A|B][B|A]"),
        "model2network/cycle":
            lambda: pybnlearn.model2network("[A|C][B|A][C|B]"),
        "model2network/self-loop": lambda: pybnlearn.model2network("[A|A]"),
        "model2network/duplicate-node":
            lambda: pybnlearn.model2network("[A][A]"),
        "model2network/undeclared-parent":
            lambda: pybnlearn.model2network("[A|Z]"),
        "model2network/single-node": lambda: pybnlearn.model2network("[A]"),
        "model2network/empty": lambda: pybnlearn.model2network(""),
        "random.graph/no-nodes": lambda: pybnlearn.random_graph([]),
        "random.graph/single-node": lambda: pybnlearn.random_graph(["A"]),
        "random.graph/duplicate-nodes":
            lambda: pybnlearn.random_graph(["A", "A"]),
        "random.graph/num-zero":
            lambda: pybnlearn.random_graph(["A", "B", "C"], num=0),
        "random.graph/num-negative":
            lambda: pybnlearn.random_graph(["A", "B", "C"], num=-1),
        "random.graph/method-unknown":
            lambda: pybnlearn.random_graph(["A", "B", "C"], method="nosuch"),
        "random.graph/prob-negative":
            lambda: pybnlearn.random_graph(["A", "B", "C"], method="ordered",
                                           prob=-0.1),
        "random.graph/prob-above-one":
            lambda: pybnlearn.random_graph(["A", "B", "C"], method="ordered",
                                           prob=1.5),
        "random.graph/every-zero":
            lambda: pybnlearn.random_graph(["A", "B", "C"], method="ic-dag",
                                           every=0),
        "random.graph/burn.in-negative":
            lambda: pybnlearn.random_graph(["A", "B", "C"], method="ic-dag",
                                           burn_in=-1),
        "random.graph/max.degree-zero":
            lambda: pybnlearn.random_graph(["A", "B", "C"], method="ic-dag",
                                           max_degree=0),
        "subgraph/unknown-node": lambda: pybnlearn.subgraph(dag, ["A", "Z"]),
        "subgraph/no-nodes": lambda: pybnlearn.subgraph(dag, []),
        "subgraph/single-node": lambda: pybnlearn.subgraph(dag, ["A"]),
        "subgraph/duplicate-nodes":
            lambda: pybnlearn.subgraph(dag, ["A", "A"]),
        "compare/mismatched-nodes": lambda: pybnlearn.compare(dag, foreign),
        "shd/mismatched-nodes": lambda: pybnlearn.shd(dag, foreign),
        "hamming/mismatched-nodes": lambda: pybnlearn.hamming(dag, foreign),
        "compare/subset-nodes": lambda: pybnlearn.compare(dag, subset),
        "count.graphs/nodes-negative":
            lambda: pybnlearn.count_graphs("all-dags", nodes=-1),
        "count.graphs/nodes-zero":
            lambda: pybnlearn.count_graphs("all-dags", nodes=0),
        "count.graphs/nodes-fractional":
            lambda: pybnlearn.count_graphs("all-dags", nodes=2.5),
        "count.graphs/type-unknown":
            lambda: pybnlearn.count_graphs("nosuch", nodes=3),
        "count.graphs/k-above-nodes":
            lambda: pybnlearn.count_graphs("dags-with-k-roots", nodes=3, k=5),
        "count.graphs/k-zero":
            lambda: pybnlearn.count_graphs("dags-with-k-roots", nodes=3, k=0),
        "count.graphs/r-negative":
            lambda: pybnlearn.count_graphs("dags-with-r-arcs", nodes=3, r=-1),

        # -- node and arc accessors --------------------------------------------------------
        "parents/unknown-node": lambda: pybnlearn.parents(dag, "Z"),
        "children/unknown-node": lambda: pybnlearn.children(dag, "Z"),
        "mb/unknown-node": lambda: pybnlearn.mb(dag, "Z"),
        "nbr/unknown-node": lambda: pybnlearn.nbr(dag, "Z"),
        "ancestors/unknown-node": lambda: pybnlearn.ancestors(dag, "Z"),
        "descendants/unknown-node": lambda: pybnlearn.descendants(dag, "Z"),
        "in.degree/unknown-node": lambda: pybnlearn.in_degree(dag, "Z"),
        "incident.arcs/unknown-node":
            lambda: pybnlearn.incident_arcs(dag, "Z"),
        "parents/several-nodes": lambda: pybnlearn.parents(dag, ["A", "B"]),
        "set.arc/unknown-node": lambda: pybnlearn.set_arc(dag, "Z", "A"),
        "set.arc/self-loop": lambda: pybnlearn.set_arc(dag, "A", "A"),
        "set.arc/reversal": lambda: pybnlearn.set_arc(dag, "B", "A"),
        "set.arc/cycle": lambda: pybnlearn.set_arc(tri, "C", "A"),
        "drop.arc/unknown-node": lambda: pybnlearn.drop_arc(dag, "Z", "A"),
        "drop.arc/absent-arc": lambda: pybnlearn.drop_arc(dag, "A", "F"),
        "reverse.arc/absent-arc":
            lambda: pybnlearn.reverse_arc(dag, "A", "F"),
        "reverse.arc/cycle": lambda: pybnlearn.reverse_arc(tri, "A", "C"),
        "set.edge/self-loop": lambda: pybnlearn.set_edge(dag, "A", "A"),
        "add.node/existing-node": lambda: pybnlearn.add_node(dag, "A"),
        "remove.node/unknown-node": lambda: pybnlearn.remove_node(dag, "Z"),
        "rename.nodes/wrong-length":
            lambda: pybnlearn.rename_nodes(dag, ["a", "b"]),
        "rename.nodes/duplicates":
            lambda: pybnlearn.rename_nodes(dag,
                                           ["a", "a", "c", "d", "e", "f"]),
        "amat/set/wrong-size":
            lambda: pybnlearn.set_amat(dag, pybnlearn.amat(dag).iloc[:3, :3]),
        "amat/set/cyclic":
            lambda: pybnlearn.set_amat(
                tri, amat_with(tri, {("A", "C"): 0, ("C", "A"): 1})),
        "amat/set/undirected":
            lambda: pybnlearn.set_amat(tri,
                                       amat_with(tri, {("C", "A"): 1})),

        # -- simulation and inference ---------------------------------------------------------
        "n/rbn/zero": lambda: pybnlearn.rbn(fitted, n=0),
        "n/rbn/negative": lambda: pybnlearn.rbn(fitted, n=-1),
        "n/rbn/fractional": lambda: pybnlearn.rbn(fitted, n=1.5),
        "n/cpdist/zero": lambda: pybnlearn.cpdist(fitted, nodes=["A"], n=0),
        "n/cpdist/negative":
            lambda: pybnlearn.cpdist(fitted, nodes=["A"], n=-1),
        "nodes/cpdist/unknown":
            lambda: pybnlearn.cpdist(fitted, nodes=["Z"]),
        "n/predict/zero":
            lambda: pybnlearn.predict(fitted, node="A", data=lt,
                                      method="bayes-lw", n=0),
        "n/predict/negative":
            lambda: pybnlearn.predict(fitted, node="A", data=lt,
                                      method="bayes-lw", n=-1),
        "n/impute/zero":
            lambda: pybnlearn.impute(fitted, lt, method="bayes-lw", n=0),
        "n/impute/negative":
            lambda: pybnlearn.impute(fitted, lt, method="bayes-lw", n=-1),

        # -- preprocessing -----------------------------------------------------------------------
        "breaks/discretize/zero":
            lambda: pybnlearn.discretize(gt, method="quantile", breaks=0),
        "breaks/discretize/one":
            lambda: pybnlearn.discretize(gt, method="quantile", breaks=1),
        "breaks/discretize/negative":
            lambda: pybnlearn.discretize(gt, method="quantile", breaks=-1),
        "breaks/discretize/fractional":
            lambda: pybnlearn.discretize(gt, method="quantile", breaks=2.5),
        "ibreaks/discretize/below-breaks":
            lambda: pybnlearn.discretize(gt, method="hartemink", breaks=3,
                                         ibreaks=2),
        "discretize/discrete-data":
            lambda: pybnlearn.discretize(lt, method="quantile"),
        "threshold/dedup/negative": lambda: pybnlearn.dedup(gt,
                                                            threshold=-0.1),
        "threshold/dedup/above-one": lambda: pybnlearn.dedup(gt,
                                                             threshold=1.5),
        "dedup/discrete-data": lambda: pybnlearn.dedup(lt),

        # -- classifiers ----------------------------------------------------------------------------
        "training/naive.bayes/unknown":
            lambda: pybnlearn.naive_bayes(lt, training="Z"),
        "training/naive.bayes/continuous":
            lambda: pybnlearn.naive_bayes(gt, training="A"),
        "explanatory/naive.bayes/unknown":
            lambda: pybnlearn.naive_bayes(lt, training="A",
                                          explanatory=["Z"]),
        "explanatory/naive.bayes/includes-training":
            lambda: pybnlearn.naive_bayes(lt, training="A",
                                          explanatory=["A", "B"]),
        "training/tree.bayes/unknown":
            lambda: pybnlearn.tree_bayes(lt, training="Z"),
        "root/tree.bayes/unknown":
            lambda: pybnlearn.tree_bayes(lt, training="A", root="Z"),
        "root/tree.bayes/is-training":
            lambda: pybnlearn.tree_bayes(lt, training="A", root="A"),
    }


def test_every_case_is_accounted_for(cases):
    """The generator and this file have to stay in step.

    A fixture id with no entry anywhere would otherwise be silently untested,
    which is exactly the failure this whole file exists to prevent -- and the
    likeliest way for it to happen is someone adding a row to the R side and
    not the Python side.
    """
    registered = set(cases) | set(NOT_PORTED)
    recorded = {record["id"] for record in _records()}

    assert recorded, "fixtures/rejections.json is missing or empty"

    assert not recorded - registered, (
        "in the fixtures but not in this file: "
        + ", ".join(sorted(recorded - registered)))
    assert not registered - recorded, (
        "in this file but not in the fixtures: "
        + ", ".join(sorted(registered - recorded)))

    # NOT_PORTED has no call to make, so it cannot also be in CASES; STRICTER
    # annotates a call that is there, so it must be.
    assert not set(cases) & set(NOT_PORTED), (
        "both called and declared unported: "
        + ", ".join(sorted(set(cases) & set(NOT_PORTED))))
    assert set(STRICTER) <= set(cases), (
        "declared stricter but never called: "
        + ", ".join(sorted(set(STRICTER) - set(cases))))


def _live():
    return [r for r in _records() if r["id"] not in NOT_PORTED]


def _expects_rejection(record):
    """R's verdict, except where STRICTER says to invert it."""
    if record["id"] in STRICTER:
        return not record["errors"]
    return record["errors"]


@pytest.mark.parametrize(
    "case", [r for r in _live() if _expects_rejection(r)],
    ids=lambda r: r["id"])
def test_input_r_rejects_is_rejected(case, cases):
    """The direction that matters: R stopped, so this has to stop too.

    The exception type is checked loosely on purpose.  What is being pinned
    down is that the call has no answer, not which of ValueError, TypeError
    or KeyError says so -- but it does have to be one of those, because an
    AttributeError or an IndexError would mean the call fell over somewhere
    downstream of the check rather than being refused by it.
    """
    with pytest.raises((ValueError, TypeError, KeyError,
                        pybnlearn.BNLearnError)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cases[case["id"]]()


@pytest.mark.parametrize(
    "case", [r for r in _live() if not _expects_rejection(r)],
    ids=lambda r: r["id"])
def test_input_r_accepts_is_accepted(case, cases):
    """The other direction, which is a parity bug just as much as the first.

    R's boundaries are not where intuition puts them: alpha = 0 and alpha = 1
    are both inside is.probability(), restart = 0 short-circuits before the
    check that would reject it while perturb = 0 does not, and a maxp above
    the node count only warns.  Refusing any of those would reject code that
    works against R, and the traceback would blame an argument that is fine.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases[case["id"]]()


@pytest.mark.parametrize("case", _records(), ids=lambda r: r["id"])
def test_r_recorded_a_reason_for_every_rejection(case):
    """A rejection with no message means the generator caught something other
    than a stop() -- an interrupt, or an error raised building the call
    itself -- and the row is then about the generator rather than about
    bnlearn."""
    if case["errors"]:
        assert case["message"].strip(), "R stopped without saying why"
    else:
        assert case["message"] == ""


def test_the_grid_covers_the_failure_modes_that_return_a_result():
    """The point of the file, asserted rather than left to the comments.

    Each of these groups is one where the wrong answer is a *network* or a
    *number* rather than an exception, which is the only kind of wrong answer
    a parity suite built from valid inputs cannot catch.  If a refactor
    thinned the grid down to label typos and unknown node names it would
    still pass everything above and be worth much less.
    """
    groups = {record["group"] for record in _records() if record["errors"]}

    for required in ("search-counters", "cross-validation", "mismatch",
                     "data", "resampling", "alpha", "max.sx"):
        assert required in groups, f"nothing in the grid rejects for {required}"


def test_a_finite_extreme_is_still_data(datasets):
    """The complement of data/hc/infinite: the check is for infinity, not for
    magnitude.  A large number is unusual, not invalid, and rejecting it would
    be a new restriction rather than a reproduction of R's."""
    extreme = datasets["gaussian.test"].head(100).copy()
    extreme.loc[extreme.index[0], "A"] = 1e300

    assert pybnlearn.hc(extreme).nodes == list(
        datasets["gaussian.test"].columns)
