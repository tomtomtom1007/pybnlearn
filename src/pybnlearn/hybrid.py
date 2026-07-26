"""Hybrid structure learning.

A port of bnlearn's hybrid.search() in R/learning-algorithms.R.  These run in
two phases: a constraint-based (or pairwise) algorithm learns an undirected
skeleton, and then a score-based search is run with everything outside that
skeleton blacklisted.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import pandas as pd

from ._core import complement_arcs, consistent_extension
from . import constraint, graph, structure

__all__ = ["h2pc", "mmhc", "rsmax2"]


_RESTRICT = {
    "gs": constraint.gs,
    "iamb": constraint.iamb,
    "inter.iamb": constraint.inter_iamb,
    "iamb.fdr": constraint.iamb_fdr,
    "mmpc": constraint.mmpc,
    "si.hiton.pc": constraint.si_hiton_pc,
    "pc.stable": constraint.pc_stable,
    "chow.liu": graph.chow_liu,
    "aracne": graph.aracne,
}

# the pairwise learners take a different set of arguments from the
# constraint-based ones.
_PAIRWISE = {"chow.liu", "aracne"}

_MAXIMIZE = {
    "hc": structure.hc,
    "tabu": structure.tabu,
}


def _hybrid(data, restrict, maximize, whitelist, blacklist,
            restrict_args, maximize_args, algorithm):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if restrict not in _RESTRICT:
        raise ValueError(
            f"unknown restrict algorithm {restrict!r}; available are "
            + ", ".join(sorted(_RESTRICT)))
    if maximize not in _MAXIMIZE:
        raise ValueError(
            f"unknown maximize algorithm {maximize!r}; available are "
            + ", ".join(sorted(_MAXIMIZE)))

    nodes = [str(c) for c in data.columns]
    whitelist = [(str(a), str(b)) for a, b in (whitelist or ())]
    blacklist = [(str(a), str(b)) for a, b in (blacklist or ())]

    # --- restrict phase: learn an undirected skeleton -----------------------
    if restrict in _PAIRWISE:
        learned = _RESTRICT[restrict](data, whitelist=whitelist or None,
                                      blacklist=blacklist or None,
                                      **(restrict_args or {}))
    else:
        learned = _RESTRICT[restrict](data, whitelist=whitelist or None,
                                      blacklist=blacklist or None,
                                      undirected=True,
                                      **(restrict_args or {}))

    # Everything the restrict phase ruled out becomes a blacklist for the
    # search, together with whatever the caller had already banned.
    constraints = complement_arcs(
        learned.arcs, nodes,
        whitelist=learned.learning.get("blacklist") or None)

    # --- maximize phase -----------------------------------------------------
    # The constraint-based phase accepts an arc whitelisted in both directions;
    # the score-based one cannot, so an undirected whitelist is replaced by a
    # consistent extension of it.
    if any((b, a) in whitelist for a, b in whitelist):
        whitelist = consistent_extension(whitelist, nodes)

    dag = _MAXIMIZE[maximize](data, whitelist=whitelist or None,
                              blacklist=constraints or None,
                              **(maximize_args or {}))

    dag.learning = dict(dag.learning)
    dag.learning.update({
        "algo": algorithm,
        "restrict": restrict,
        "rstest": learned.learning.get("test"),
        "maximize": maximize,
        "maxscore": dag.learning.get("test"),
        "whitelist": whitelist,
        "blacklist": blacklist,
    })

    return dag


def rsmax2(data, whitelist=None, blacklist=None, restrict="si.hiton.pc",
           maximize="hc", restrict_args=None, maximize_args=None):
    """A general two-phase hybrid, as bnlearn's rsmax2()."""
    return _hybrid(data, restrict, maximize, whitelist, blacklist,
                   restrict_args, maximize_args, "rsmax2")


def mmhc(data, whitelist=None, blacklist=None, restrict_args=None,
         maximize_args=None):
    """Max-Min Hill Climbing: mmpc to restrict, hc to maximize."""
    return _hybrid(data, "mmpc", "hc", whitelist, blacklist,
                   restrict_args, maximize_args, "mmhc")


def h2pc(data, whitelist=None, blacklist=None, restrict_args=None,
         maximize_args=None):
    """Hybrid HPC: not available yet.

    bnlearn's h2pc() restricts with hpc(), which is not ported.  rsmax2() with
    another restrict algorithm is the nearest thing available.
    """
    raise NotImplementedError(
        "h2pc() needs the hpc() algorithm, which is not ported yet; use "
        "rsmax2() with restrict='si.hiton.pc' or restrict='mmpc' instead")
