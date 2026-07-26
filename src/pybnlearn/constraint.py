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
import warnings

import pandas as pd

from ._core import Tester, cpdag_arcs, recover_structure
from .structure import (BayesianNetwork, _check_complete, _data_type,
                        build_blacklist)

__all__ = ["gs", "iamb", "iamb_fdr", "inter_iamb", "mmpc",
           "si_hiton_pc"]


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


def _fdr_thresholds(m):
    """The Benjamini-Yekutieli correction factors iamb.fdr applies.

    Element i (one-based) is m/i * H_m, and it is matched against the i-th
    smallest p-value, so the ordering of the tests is what decides which
    threshold each node is held to.
    """
    harmonic = sum(1.0 / i for i in range(1, m + 1))
    return [m / i * harmonic for i in range(1, m + 1)]


def _ia_fdr_markov_blanket(tester, target, nodes, alpha, whitelist, blacklist,
                           max_sx):
    """ia.fdr.markov.blanket(): add and remove nodes by false discovery rate
    rather than by raw p-value."""
    candidates = [n for n in nodes if n != target]
    thresholds = _fdr_thresholds(len(candidates))

    whitelisted = [y for y in candidates
                   if _listed(whitelist, (target, y), either=True)]

    mb = list(dict.fromkeys(whitelisted))
    culprit = []
    state = []
    last_added = last_removed = None

    while True:
        if len(mb) > max_sx:
            break

        if any(set(mb) == set(previous) for previous in state):
            # The blanket has come back to somewhere it has already been.
            # Undo whatever move led here and refuse to consider that node
            # again, or the search never terminates.
            if last_removed is not None:
                mb = mb + [last_removed]
                culprit.append(last_removed)
            elif last_added is not None:
                mb = [n for n in mb if n != last_added]
                culprit.append(last_added)
            if state:
                state.pop()
            warnings.warn(
                "prevented an infinite loop while learning the Markov blanket "
                f"of {target!r}", stacklevel=2)

        state.append(list(mb))
        snapshot = list(mb)

        association = {node: tester.pvalue(target, node,
                                           [n for n in mb if n != node])
                       for node in candidates}
        # smallest p-value first; the threshold a node is held to depends on
        # its rank, so this ordering is part of the algorithm.
        ordered = sorted(association, key=lambda n: association[n])
        threshold = dict(zip(ordered, thresholds))

        # removal, from the weakest association down.
        blocked = set(whitelisted) | set(culprit)
        if last_added is not None:
            blocked.add(last_added)
        for node in reversed([n for n in ordered if n in mb
                              and n not in blocked]):
            if association[node] * threshold[node] > alpha:
                mb = [n for n in mb if n != node]
                last_added, last_removed = None, node
                break

        if mb != snapshot:
            continue

        # addition, from the strongest association up.
        blocked = set(mb) | set(culprit)
        if last_removed is not None:
            blocked.add(last_removed)
        for node in [n for n in ordered if n in candidates
                     and n not in blocked]:
            if association[node] * threshold[node] <= alpha:
                mb = mb + [node]
                last_added, last_removed = node, None
                break

        if mb == snapshot:
            break

    return mb


# ---------------------------------------------------------------------------
# neighbourhoods and orientation
# ---------------------------------------------------------------------------

def _mmpc_forward(tester, target, nodes, alpha, whitelist, blacklist, max_sx):
    """maxmin.pc.forward.phase(): grow the candidate parent-children set by
    max-min association."""
    candidates = [n for n in nodes if n != target]
    whitelisted = [y for y in candidates
                   if _listed(whitelist, (target, y), either=True)]
    blacklisted = [y for y in candidates
                   if _listed(blacklist, (target, y), both=True)]

    cpc = list(dict.fromkeys(whitelisted))
    available = [n for n in candidates
                 if n not in cpc and n not in blacklisted]

    # A node's association can only get weaker as the conditioning set grows,
    # so once it is above alpha it never has to be tested again.
    association = {n: 0.0 for n in available}

    while True:
        if len(cpc) > max_sx:
            break

        to_check = [n for n in association if n not in cpc]
        if not to_check:
            break

        # only subsets containing the node added last are new.
        fixed = cpc[-1:] if cpc else []
        sx = cpc[:-1] if cpc else []

        updated = {}
        for node in to_check:
            result = tester.allsubs(node, target, sx=sx, fixed=fixed,
                                    alpha=alpha)
            updated[node] = max(association[node], result["max.p.value"])
        association = updated

        if not association or all(p > alpha for p in association.values()):
            break
        if not available:
            break

        to_add = min(association, key=association.get)
        if association[to_add] <= alpha:
            cpc.append(to_add)
            available = [n for n in available if n != to_add]

    return cpc


def _hiton_forward(tester, target, nodes, alpha, whitelist, blacklist, max_sx):
    """si.hiton.pc.heuristic(): rank candidates by marginal association once,
    then admit them one at a time subject to a backward check."""
    candidates = [n for n in nodes if n != target]
    whitelisted = [y for y in candidates
                   if _listed(whitelist, (target, y), either=True)]
    blacklisted = [y for y in candidates
                   if _listed(blacklist, (target, y), both=True)]

    cpc = list(dict.fromkeys(whitelisted))
    available = [n for n in candidates
                 if n not in cpc and n not in blacklisted]

    if not available:
        return cpc

    association = tester.pvalues(available, target, None)
    available = [n for n in available if association[n] <= alpha]
    if all(p > alpha for p in association.values()):
        return cpc

    while True:
        if (not association or not available or len(cpc) > max_sx
                or all(p > alpha for p in association.values())):
            break

        to_add = min(association, key=association.get)

        # the candidate stays only if no subset of the current set separates
        # it from the target.
        keep = True
        if cpc:
            result = tester.allsubs(target, to_add, sx=cpc, min=1, alpha=alpha)
            keep = result["p.value"] <= alpha

        if keep:
            cpc.append(to_add)

        available = [n for n in available if n != to_add]
        association = {k: v for k, v in association.items() if k != to_add}

    return cpc


def _fake_markov_blanket(structure, target):
    """fake.markov.blanket(): everything within distance two of the target.

    mmpc and si.hiton.pc learn neighbourhoods directly and never compute a
    Markov blanket, but the orientation phase wants one; this superset is what
    bnlearn substitutes.
    """
    out = []
    for neighbour_ in structure[target]["nbr"]:
        out.extend(structure[neighbour_]["nbr"])
    out.extend(structure[target]["nbr"])
    return [n for n in dict.fromkeys(out) if n != target]


def _neighbour(tester, target, blankets, alpha, whitelist, blacklist, max_sx,
               markov=True, empty_dsep=True):
    """neighbour(): cut the candidate set down to the direct neighbours.

    `markov` picks the cheaper of the two Markov blankets to search over, which
    only makes sense when they really are Markov blankets; mmpc and
    si.hiton.pc learn neighbourhoods, so they pass False.  `empty_dsep` allows
    the empty conditioning set, which si.hiton.pc has already tested.
    """
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
        if markov:
            dsep = _smaller([n for n in blankets[target] if n != y],
                            [n for n in blankets[y] if n != target])
        else:
            dsep = [n for n in blankets[target] if n != y]
        result = tester.allsubs(target, y, sx=dsep, alpha=alpha,
                                min=0 if empty_dsep else 1,
                                max=min(len(dsep), max_sx))
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
                      test, alpha, max_sx, undirected,
                      markov_blankets=True, empty_dsep=True):
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

        if markov_blankets:
            # the per-node results need not agree; make them symmetric.
            blankets = recover_structure(blankets, nodes,
                                         markov_blankets=True)

        structure = {
            node: _neighbour(tester, node, blankets, alpha, whitelist,
                             blacklist, max_sx, markov=markov_blankets,
                             empty_dsep=empty_dsep)
            for node in nodes
        }

        if not markov_blankets:
            # these algorithms never compute a Markov blanket, but the
            # orientation phase needs one; distance two is the superset
            # bnlearn substitutes.
            for node in nodes:
                structure[node]["mb"] = _fake_markov_blanket(structure, node)

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


def iamb_fdr(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
             max_sx=None, undirected=False):
    """IAMB with false discovery rate control, as bnlearn's iamb.fdr()."""
    return _constraint_learn(data, _ia_fdr_markov_blanket, "iamb.fdr",
                             whitelist, blacklist, test, alpha, max_sx,
                             undirected)


def mmpc(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
         max_sx=None, undirected=True):
    """Max-Min Parents and Children, as bnlearn's mmpc().

    Returns an undirected graph by default: this learns each node's set of
    parents and children without distinguishing the two, so orienting the
    result would claim more than the algorithm establishes.  bnlearn defaults
    the same way.
    """
    return _constraint_learn(data, _mmpc_forward, "mmpc", whitelist,
                             blacklist, test, alpha, max_sx, undirected,
                             markov_blankets=False)


def si_hiton_pc(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
                max_sx=None, undirected=True):
    """Semi-Interleaved HITON-PC, as bnlearn's si.hiton.pc().

    Like mmpc(), this returns an undirected graph by default.
    """
    return _constraint_learn(data, _hiton_forward, "si.hiton.pc", whitelist,
                             blacklist, test, alpha, max_sx, undirected,
                             markov_blankets=False, empty_dsep=False)
