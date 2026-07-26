"""Constraint-based structure learning.

A port of bnlearn's R/grow-shrink.R, R/incremental-association.R,
R/inter-iamb.R and the shared machinery in R/backend-indep.R.

These algorithms are control flow around conditional independence tests: learn
each node's Markov blanket, reduce it to a neighbourhood, then orient what can
be oriented.  The loops are plain Python; every test, and the graph surgery at
the end, goes through the same C core R uses.

The order in which tests are run is part of the algorithm, not an
implementation detail -- `gs` breaks out of its grow loop at the *first* node
that passes, so iterating candidates in a different order gives a different
Markov blanket.  The loops below therefore follow the R source step for step
rather than being rewritten in a more idiomatic shape.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import itertools
import math

import pandas as pd

from ._core import Tester, cpdag_arcs, recover_structure
from .structure import (BayesianNetwork, _check_complete, _data_type,
                        build_blacklist)

__all__ = ["gs", "iamb", "inter_iamb"]


_DISCRETE_TESTS = {"mi", "mi-adf", "mi-sh", "x2", "x2-adf"}
_CONTINUOUS_TESTS = {"cor", "zf", "mi-g", "mi-g-sh"}


def _check_test(test, data):
    """check.test(): the default test depends on the data type."""
    kind = _data_type(data)

    if test is None:
        return {"discrete": "mi", "continuous": "cor", "mixed-cg": "mi-cg"}[kind]

    known = _DISCRETE_TESTS | _CONTINUOUS_TESTS
    if test not in known:
        raise ValueError(
            f"test {test!r} is not implemented yet; available tests are "
            + ", ".join(sorted(known)))
    if kind != "discrete" and test in _DISCRETE_TESTS:
        raise ValueError(f"test {test!r} may only be used with discrete data")
    if kind != "continuous" and test in _CONTINUOUS_TESTS:
        raise ValueError(f"test {test!r} may only be used with continuous data")

    return test


def _smaller(a, b):
    """smaller(): bnlearn picks whichever d-separating set is cheaper to
    enumerate, breaking ties towards the first."""
    return a if len(a) <= len(b) else b


def _listed(arcs, pair, either=False, both=False):
    a, b = pair
    if either:
        return (a, b) in arcs or (b, a) in arcs
    if both:
        return (a, b) in arcs and (b, a) in arcs
    return (a, b) in arcs


# ---------------------------------------------------------------------------
# Markov blanket learning
# ---------------------------------------------------------------------------

def _gs_markov_blanket(tester, target, nodes, alpha, whitelist, blacklist,
                       max_sx):
    """gs.markov.blanket(): grow, then shrink."""
    candidates = [n for n in nodes if n != target]
    whitelisted = [y for y in candidates
                   if _listed(whitelist, (target, y), either=True)]

    mb = list(dict.fromkeys(whitelisted))
    candidates = [n for n in candidates if n not in mb]

    # grow: add the first node that is dependent on the target, then restart.
    while True:
        size = len(mb)
        for y in list(candidates):
            if size > max_sx:
                continue
            if tester.pvalue(target, y, mb) <= alpha:
                mb.append(y)
                candidates.remove(y)
                break
        if len(mb) == size:
            break

    # shrink: drop nodes that the rest of the blanket makes redundant.
    removable = [n for n in mb if n not in whitelisted]
    while True:
        size = len(mb)
        for y in list(removable):
            rest = [n for n in mb if n != y]
            if tester.pvalue(target, y, rest) > alpha:
                mb.remove(y)
                removable.remove(y)
                break
        if len(mb) == size:
            break

    return mb


def _ia_markov_blanket(tester, target, nodes, alpha, whitelist, blacklist,
                       max_sx):
    """ia.markov.blanket(): forward selection on the strongest association,
    then one round-robin backward pass."""
    candidates = [n for n in nodes if n != target]
    whitelisted = [y for y in candidates
                   if _listed(whitelist, (target, y), either=True)]

    mb = list(dict.fromkeys(whitelisted))
    candidates = [n for n in candidates if n not in mb]
    to_add = ""

    while candidates and len(mb) <= max_sx:
        association = tester.pvalues(candidates, target, mb)
        if all(p > alpha for p in association.values()):
            break
        to_add = min(association, key=association.get)
        if association[to_add] <= alpha:
            mb.append(to_add)
            candidates.remove(to_add)

    # The node added last can never be dropped: the test that admitted it is
    # the same test that would remove it.
    fixed = [n for n in ([to_add] + whitelisted) if n]
    pv = tester.roundrobin(target, mb, fixed=fixed, alpha=alpha)
    keep = set(k for k, v in pv.items() if v < alpha) | set(fixed)

    return [n for n in mb if n in keep]


def _inter_ia_markov_blanket(tester, target, nodes, alpha, whitelist,
                             blacklist, max_sx):
    """inter.ia.markov.blanket(): interleave the backward pass with the
    forward one, so a node admitted early can be dropped again."""
    candidates = [n for n in nodes if n != target]
    whitelisted = [y for y in candidates
                   if _listed(whitelist, (target, y), either=True)]

    mb = list(dict.fromkeys(whitelisted))
    candidates = [n for n in candidates if n not in mb]
    culprit = []
    seen = []

    while True:
        available = [n for n in candidates if n not in mb and n not in culprit]
        if not available or len(mb) > max_sx:
            break

        snapshot = list(mb)
        association = tester.pvalues(available, target, mb)
        if all(p > alpha for p in association.values()):
            break

        to_add = min(association, key=association.get)
        if association[to_add] <= alpha:
            mb.append(to_add)

        # backward pass over everything except the node just added.
        for y in [n for n in mb if n != to_add and n not in whitelisted]:
            rest = [n for n in mb if n != y]
            if tester.pvalue(target, y, rest) > alpha:
                mb.remove(y)

        if mb == snapshot:
            # the node that was added got removed again in the same round;
            # do not consider it any more, or the loop never ends.
            culprit.append(to_add)

        if mb in seen:
            break
        seen.append(list(mb))

    return mb


# ---------------------------------------------------------------------------
# neighbourhoods and orientation
# ---------------------------------------------------------------------------

def _neighbour(tester, target, blankets, alpha, whitelist, blacklist, max_sx):
    """neighbour(): cut the Markov blanket down to the direct neighbours."""
    candidates = list(blankets[target])

    if not candidates:
        return {"mb": [], "nbr": []}

    blacklisted = [y for y in candidates
                   if _listed(blacklist, (target, y), both=True)]
    whitelisted = [y for y in candidates
                   if _listed(whitelist, (target, y), either=True)]

    candidates = [n for n in candidates if n not in blacklisted]
    for y in whitelisted:
        if y not in candidates:
            candidates.append(y)

    if len(candidates) <= 1:
        return {"mb": list(blankets[target]), "nbr": candidates}

    for y in [n for n in list(candidates) if n not in whitelisted]:
        dsep = _smaller([n for n in blankets[target] if n != y],
                        [n for n in blankets[y] if n != target])
        result = tester.allsubs(target, y, sx=dsep, alpha=alpha,
                                min=0, max=min(len(dsep), max_sx))
        if result["p.value"] > alpha:
            candidates.remove(y)

    return {"mb": list(blankets[target]), "nbr": candidates}


def _sets2arcs(structure, nodes):
    return [(node, nbr) for node in nodes
            for nbr in structure[node]["nbr"]]


def _vstruct_detect(tester, nodes, arcs, structure, alpha, blacklist, max_sx):
    """vstruct.detect(): find unshielded triples y -> x <- z."""
    found = []

    for x in nodes:
        incoming = [a for a, b in arcs if b == x]
        if len(incoming) < 2:
            continue

        for y, z in itertools.combinations(incoming, 2):
            if _listed(arcs, (y, z), either=True):
                continue

            sx = _smaller([n for n in structure[y]["mb"] if n not in (x, z)],
                          [n for n in structure[z]["mb"] if n not in (x, y)])
            result = tester.allsubs(y, z, sx=sx, fixed=[x], alpha=alpha,
                                    max=min(max_sx, len(sx)))

            if result["p.value"] <= alpha:
                found.append((result["max.p.value"], y, x, z))

    return found


def _vstruct_apply(arcs, vstructures, nodes, acyclic):
    """vstruct.apply(): orient the v-structures, most significant first,
    skipping any that conflict with what is already oriented."""
    arcs = list(arcs)

    for _, y, x, z in vstructures:
        if not (_listed(arcs, (y, x)) and _listed(arcs, (z, x))):
            continue

        candidate = _set_arc(_set_arc(arcs, y, x), z, x)
        if not acyclic(candidate):
            continue

        arcs = candidate

    return arcs


def _set_arc(arcs, frm, to):
    """set.arc.direction() on an undirected arc set."""
    if (to, frm) in arcs and (frm, to) in arcs:
        return [a for a in arcs if a != (to, frm)]
    if (to, frm) in arcs:
        return [a for a in arcs if a != (to, frm)] + [(frm, to)]
    if (frm, to) in arcs:
        return list(arcs)
    return list(arcs) + [(frm, to)]


# ---------------------------------------------------------------------------
# the shared driver
# ---------------------------------------------------------------------------

def _constraint_learn(data, blanket_fn, algorithm, whitelist, blacklist,
                      test, alpha, max_sx, undirected):
    _check_complete(data)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    nodes = [str(c) for c in data.columns]
    test = _check_test(test, data)

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    max_sx = len(nodes) if max_sx is None else int(max_sx)

    whitelist = [(str(a), str(b)) for a, b in (whitelist or ())]
    blacklist = [(str(a), str(b)) for a, b in (blacklist or ())]
    for a, b in whitelist + blacklist:
        if a not in nodes or b not in nodes:
            raise ValueError(f"unknown node in arc ({a}, {b})")

    # build.blacklist(): the reverse of a whitelisted arc is forbidden,
    # which is what fixes its direction during the orientation phase.
    blacklist = build_blacklist(blacklist, whitelist)

    with Tester(data, test) as tester:
        blankets = {
            node: blanket_fn(tester, node, nodes, alpha, whitelist, blacklist,
                             max_sx)
            for node in nodes
        }
        # the per-node results need not agree; make them symmetric.
        blankets = recover_structure(blankets, nodes, markov_blankets=True)

        structure = {
            node: _neighbour(tester, node, blankets, alpha, whitelist,
                             blacklist, max_sx)
            for node in nodes
        }
        structure = recover_structure(structure, nodes, markov_blankets=False)

        arcs = _sets2arcs(structure, nodes)
        arcs = [a for a in arcs if not _listed(blacklist, a)]

        if not undirected:
            vs = _vstruct_detect(tester, nodes, arcs, structure, alpha,
                                 blacklist, max_sx)
            if vs:
                vs.sort(key=lambda row: row[0])

                def acyclic(candidate):
                    return _is_acyclic(candidate, nodes)

                arcs = _vstruct_apply(arcs, vs, nodes, acyclic)

            arcs = cpdag_arcs(arcs, nodes, whitelist=whitelist,
                              blacklist=blacklist, fix=True,
                              wlbl=bool(whitelist or blacklist))

    return BayesianNetwork(
        nodes, arcs,
        learning={
            "algo": algorithm,
            "test": test,
            "args": {"alpha": alpha},
            "whitelist": whitelist,
            "blacklist": blacklist,
            "undirected": undirected,
            "max.sx": max_sx,
        },
    )


def _is_acyclic(arcs, nodes):
    """A directed-cycle check over the directed arcs only, which is what
    vstruct.apply() asks for."""
    directed = [(a, b) for a, b in arcs if (b, a) not in arcs]
    children = {n: [] for n in nodes}
    for a, b in directed:
        children[a].append(b)

    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(nodes, WHITE)

    def visit(node):
        colour[node] = GREY
        for nxt in children[node]:
            if colour[nxt] == GREY:
                return False
            if colour[nxt] == WHITE and not visit(nxt):
                return False
        colour[node] = BLACK
        return True

    return all(colour[n] != WHITE or visit(n) for n in nodes)


# ---------------------------------------------------------------------------
# the public algorithms
# ---------------------------------------------------------------------------

def gs(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
       max_sx=None, undirected=False):
    """Grow-Shrink, as bnlearn's gs()."""
    return _constraint_learn(data, _gs_markov_blanket, "gs", whitelist,
                             blacklist, test, alpha, max_sx, undirected)


def iamb(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
         max_sx=None, undirected=False):
    """Incremental Association, as bnlearn's iamb()."""
    return _constraint_learn(data, _ia_markov_blanket, "iamb", whitelist,
                             blacklist, test, alpha, max_sx, undirected)


def inter_iamb(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
               max_sx=None, undirected=False):
    """Interleaved Incremental Association, as bnlearn's inter.iamb()."""
    return _constraint_learn(data, _inter_ia_markov_blanket, "inter.iamb",
                             whitelist, blacklist, test, alpha, max_sx,
                             undirected)
