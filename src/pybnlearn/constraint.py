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

from ._core import (Tester, cpdag_arcs, neighbourhoods,
                    recover_structure)
from .structure import (BayesianNetwork, _check_complete, _data_type,
                        build_blacklist)

__all__ = ["fast_iamb", "gs", "hpc", "iamb", "iamb_fdr", "inter_iamb",
           "learn_mb", "learn_nbr", "mmpc", "pc_stable", "si_hiton_pc"]


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
                           max_sx, start=()):
    """ia.fdr.markov.blanket(): add and remove nodes by false discovery rate
    rather than by raw p-value.

    `start` seeds the blanket with nodes that are already known to belong;
    hpc() uses it to avoid repeating the low-order tests it has just run.
    """
    candidates = [n for n in nodes if n != target]
    thresholds = _fdr_thresholds(len(candidates))

    whitelisted = [y for y in candidates
                   if _listed(whitelist, (target, y), either=True)]

    mb = list(dict.fromkeys(list(start) + whitelisted))
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
# PC-stable
#
# This one does not fit the Markov-blanket shape at all: it starts from the
# complete graph and removes edges, growing the conditioning sets one node at a
# time, and it remembers the separating set that removed each edge.  The
# orientation phase then reads those sets instead of running fresh tests, which
# is what makes it "stable" -- the result does not depend on the order the
# nodes happen to be in.
# ---------------------------------------------------------------------------

def _adjacent(skeleton, node, exclude):
    """The nodes adjacent to `node` in the current skeleton.

    Order matters: it decides the order allsubs.test() enumerates candidate
    separating sets in, and so which one is found first when several separate
    the pair.  That set is stored and later decides v-structures.
    """
    out = [a for a, b in skeleton if b == node]
    out += [b for a, b in skeleton if a == node]
    return [n for n in dict.fromkeys(out) if n != exclude]


def _pc_heuristic(tester, pair, alpha, whitelist, blacklist, skeleton,
                  dsep_size):
    x, y = pair["arc"]

    if _listed(whitelist, (x, y), either=True):
        return {"arc": (x, y), "p.value": 0.0, "dsep.set": None,
                "max.adjacent": 0}
    if _listed(blacklist, (x, y), both=True):
        return {"arc": (x, y), "p.value": 1.0, "dsep.set": None,
                "max.adjacent": 0}

    # already separated by an earlier, smaller conditioning set.
    if pair["dsep.set"] is not None:
        return pair

    nbr1 = _adjacent(skeleton, x, y)
    nbr2 = _adjacent(skeleton, y, x)

    if len(nbr1) < dsep_size and len(nbr2) < dsep_size:
        return {"arc": (x, y), "p.value": pair["p.value"],
                "dsep.set": pair["dsep.set"], "max.adjacent": 0}

    if len(nbr1) >= dsep_size:
        result = tester.allsubs(x, y, sx=nbr1, alpha=alpha,
                                min=dsep_size, max=dsep_size)
        if result["p.value"] > alpha:
            return {"arc": (x, y), "p.value": result["p.value"],
                    "dsep.set": result["dsep.set"], "max.adjacent": 0}

    # Testing the second endpoint is redundant when the conditioning sets are
    # of size one (the same single node would be tried again) or when the two
    # neighbourhoods coincide.
    if dsep_size == 1:
        nbr2 = [n for n in nbr2 if n not in nbr1]

    if len(nbr2) >= dsep_size and dsep_size > 0 and set(nbr1) != set(nbr2):
        result = tester.allsubs(y, x, sx=nbr2, alpha=alpha,
                                min=dsep_size, max=dsep_size)
        if result["p.value"] > alpha:
            return {"arc": (x, y), "p.value": result["p.value"],
                    "dsep.set": result["dsep.set"], "max.adjacent": 0}

    return {"arc": (x, y), "p.value": 0.0, "dsep.set": None,
            "max.adjacent": max(len(nbr1), len(nbr2))}


def _pc_stable_backend(tester, nodes, alpha, whitelist, blacklist, max_sx):
    n = len(nodes)
    pairs = [{"arc": pair, "p.value": None, "dsep.set": None,
              "max.adjacent": n - 1}
             for pair in itertools.combinations(nodes, 2)]
    skeleton = [p["arc"] for p in pairs]
    nbr_size = [n - 1] * len(pairs)

    for dsep_size in range(0, min(max_sx, n - 2) + 1):
        for i, pair in enumerate(pairs):
            if dsep_size <= nbr_size[i]:
                pairs[i] = _pc_heuristic(tester, pair, alpha, whitelist,
                                         blacklist, skeleton, dsep_size)

        skeleton = [p["arc"] for p in pairs if p["p.value"] < alpha]
        nbr_size = [p["max.adjacent"] for p in pairs]

        if all(size <= dsep_size for size in nbr_size):
            break

    both = [a for pair in skeleton for a in (pair, (pair[1], pair[0]))]
    return neighbourhoods(nodes, both), pairs


def _vstruct_detect_from_dsep(nodes, arcs, pairs, alpha):
    """vstruct.detect() using the separating sets pc.stable stored.

    An unshielded triple y - x - z is a v-structure exactly when x is not in
    the set that separated y and z, so no further tests are needed.
    """
    by_pair = {frozenset(p["arc"]): p for p in pairs}
    found = []

    for x in nodes:
        incoming = [a for a, b in arcs if b == x]
        if len(incoming) < 2:
            continue

        for y, z in itertools.combinations(incoming, 2):
            if _listed(arcs, (y, z), either=True):
                continue

            entry = by_pair.get(frozenset((y, z)))
            if entry is None or entry["dsep.set"] is None:
                continue
            if x not in entry["dsep.set"]:
                found.append((entry["p.value"], y, x, z))

    return found


# ---------------------------------------------------------------------------
# the shared driver
# ---------------------------------------------------------------------------

def _constraint_learn(data, blanket_fn, algorithm, whitelist, blacklist,
                      test, alpha, max_sx, undirected,
                      markov_blankets=True, empty_dsep=True,
                      skeleton_fn=None, structure_fn=None):
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
        dsep_pairs = None

        if skeleton_fn is not None:
            # pc.stable builds the whole skeleton at once and remembers the
            # separating sets, so it skips the per-node phases entirely.
            structure, dsep_pairs = skeleton_fn(
                tester, nodes, alpha, whitelist, blacklist, max_sx)
            return _orient(tester, data, structure, nodes, algorithm, test,
                           alpha, max_sx, whitelist, blacklist, undirected,
                           dsep_pairs)

        if structure_fn is not None:
            # hpc does its own filtering, so there is no separate
            # neighbourhood phase to run afterwards.
            structure = {
                node: structure_fn(tester, node, nodes, alpha, whitelist,
                                   blacklist, max_sx)
                for node in nodes
            }
            for node in nodes:
                structure[node]["mb"] = _fake_markov_blanket(structure, node)

            structure = recover_structure(structure, nodes,
                                          markov_blankets=False)
            return _orient(tester, data, structure, nodes, algorithm, test,
                           alpha, max_sx, whitelist, blacklist, undirected,
                           None)

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

        return _orient(tester, data, structure, nodes, algorithm, test, alpha,
                       max_sx, whitelist, blacklist, undirected, None)


def _orient(tester, data, structure, nodes, algorithm, test, alpha, max_sx,
            whitelist, blacklist, undirected, dsep_pairs):
    """learn.arc.directions(): find the v-structures, apply them, then let
    cpdag() propagate what follows.

    An undirected result skips all of this: R returns the adjacencies as
    they are, without even filtering by the blacklist.  That is not an
    oversight -- a blacklist forbids an arc in one direction, and an
    undirected graph has not claimed a direction to forbid.
    """
    arcs = _sets2arcs(structure, nodes)

    if not undirected:
        arcs = [a for a in arcs if not _listed(blacklist, a)]
        if dsep_pairs is not None:
            vs = _vstruct_detect_from_dsep(nodes, arcs, dsep_pairs, alpha)
        else:
            vs = _vstruct_detect(tester, nodes, arcs, structure, alpha,
                                 blacklist, max_sx)

        if vs:
            vs.sort(key=lambda row: row[0])
            arcs = _vstruct_apply(arcs, vs, nodes,
                                  lambda c: _is_acyclic(c, nodes))

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


def pc_stable(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
              max_sx=None, undirected=False):
    """The order-independent PC algorithm, as bnlearn's pc.stable()."""
    return _constraint_learn(data, None, "pc.stable", whitelist, blacklist,
                             test, alpha, max_sx, undirected,
                             skeleton_fn=_pc_stable_backend)


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


def _observations_per_cell(counts, n, target, node, conditioning):
    """obs.per.cell(): how thinly the data are spread over the contingency
    table a test would build.

    Continuous data have no cells, so R returns infinity and the caller's
    countermeasure never fires.
    """
    if counts is None:
        return float("inf")

    cells = counts[target] * counts[node]
    for name in conditioning:
        cells *= counts[name]
    return n / cells


def _fast_ia_markov_blanket(counts, n):
    """fast.ia.markov.blanket(): IAMB that adds several nodes per pass.

    Ordinary IAMB adds the single best candidate and then re-tests
    everything.  This one sorts every candidate that looks associated and
    adds them all speculatively, then removes what the round-robin pass
    rejects -- far fewer passes over the data, at the cost of admitting
    nodes on evidence that a later test may overturn.

    The speculation stops when the contingency table would get too sparse
    for an asymptotic test to mean anything, which is the `obs.per.cell`
    check and the reason this needs the data as well as the tester.
    """
    def blanket(tester, target, nodes, alpha, whitelist, blacklist, max_sx):
        candidates = [x for x in nodes if x != target]
        whitelisted = [y for y in candidates
                       if _listed(whitelist, (target, y), either=True)]

        mb = list(dict.fromkeys(whitelisted))
        candidates = [x for x in candidates if x not in mb]
        insufficient = False

        while True:
            remaining = [x for x in candidates if x not in mb]
            if not remaining or len(mb) > max_sx:
                break

            snapshot = list(mb)
            insufficient = False

            association = tester.pvalues(remaining, target, mb)
            if all(p > alpha for p in association.values()):
                break

            # every candidate that looks associated, strongest first.
            admitted = sorted((x for x, p in association.items()
                               if p <= alpha),
                              key=lambda x: association[x])

            for node in admitted:
                if len(mb) > max_sx:
                    continue

                if _observations_per_cell(counts, n, target, node, mb) < 5:
                    # adding this would make the next round of tests
                    # meaningless; stop rather than test on air.
                    insufficient = True
                    break

                mb.append(node)

            before = len(mb)

            fixed = [x for x in whitelisted if x]
            pv = tester.roundrobin(target, mb, fixed=fixed, alpha=alpha)
            keep = {k for k, v in pv.items() if v < alpha} | set(fixed)
            mb = [x for x in mb if x in keep]

            if mb == snapshot:
                break
            if insufficient and before == len(mb):
                break

            candidates = [x for x in candidates if x not in mb]

        return mb

    return blanket


def _level_counts(data):
    """How many levels each variable has, or None for continuous data."""
    if _data_type(data) != "discrete":
        return None
    return {str(c): len(data[c].astype("category").cat.categories)
            for c in data.columns}


def fast_iamb(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
              max_sx=None, undirected=False):
    """Fast Incremental Association, as bnlearn's fast.iamb().

    The same answer IAMB is looking for, reached in fewer passes over the
    data by admitting several nodes at a time and letting the backward pass
    sort them out.

    bnlearn has deprecated this and will remove it in 2027; it is here so
    that code written against R's version keeps working, not because it is
    a good default.  `iamb` or `inter_iamb` are the ones to reach for.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    blanket = _fast_ia_markov_blanket(_level_counts(data), len(data))
    return _constraint_learn(data, blanket, "fast.iamb", whitelist, blacklist,
                             test, alpha, max_sx, undirected)


_BLANKETS = {
    "gs": _gs_markov_blanket,
    "iamb": _ia_markov_blanket,
    "inter.iamb": _inter_ia_markov_blanket,
    "iamb.fdr": _ia_fdr_markov_blanket,
}


_NEIGHBOURHOODS = {
    "mmpc": _mmpc_forward,
    "si.hiton.pc": _hiton_forward,
}


def learn_mb(data, node, method="gs", whitelist=None, blacklist=None,
             test=None, alpha=0.05, max_sx=None):
    """The Markov blanket of one node, without learning the whole network.

    Useful when only one variable matters: the blanket is everything needed
    to predict it, and finding it costs a fraction of a full structure
    learning run.

    Only the algorithms that look for a blanket in the first place are
    accepted, which is R's rule too -- the neighbourhood algorithms never
    compute one.
    """
    if method == "fast.iamb":
        blanket_fn = _fast_ia_markov_blanket(_level_counts(data), len(data))
    elif method in _BLANKETS:
        blanket_fn = _BLANKETS[method]
    else:
        raise ValueError(
            f"{method!r} does not learn Markov blankets; the ones that do "
            "are " + ", ".join(sorted(_BLANKETS) + ["fast.iamb"]))

    return _learn_local(data, node, blanket_fn, whitelist, blacklist, test,
                        alpha, max_sx)


def learn_nbr(data, node, method="mmpc", whitelist=None, blacklist=None,
              test=None, alpha=0.05, max_sx=None):
    """The neighbours of one node: the nodes adjacent to it in the graph,
    which is its Markov blanket without the spouses."""
    if method not in set(_NEIGHBOURHOODS) | {"hpc"}:
        raise ValueError(
            f"{method!r} does not learn neighbourhoods; the ones that do are "
            + ", ".join(sorted(set(_NEIGHBOURHOODS) | {"hpc"})))

    def blanket(tester, target, nodes, alpha, whitelist, blacklist, max_sx):
        if method == "hpc":
            # hpc filters as it goes, so there is nothing left to do after it.
            return _hpc_heuristic(tester, target, nodes, alpha, whitelist,
                                  blacklist, max_sx)["nbr"]

        # for the others the forward phase proposes and the backward phase
        # disposes: a candidate stays only if no subset of the rest
        # separates it from the target.  Skipping it leaves in nodes that
        # are two steps away.
        found = _NEIGHBOURHOODS[method](tester, target, nodes, alpha,
                                        whitelist, blacklist, max_sx)
        return _neighbour(tester, target, {target: found}, alpha, whitelist,
                          blacklist, max_sx, markov=False,
                          empty_dsep=True)["nbr"]

    return _learn_local(data, node, blanket, whitelist, blacklist, test,
                        alpha, max_sx)


def _learn_local(data, node, blanket_fn, whitelist, blacklist, test, alpha,
                 max_sx):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    _check_complete(data)

    nodes = [str(c) for c in data.columns]
    node = str(node)
    if node not in nodes:
        raise ValueError(f"unknown node {node!r}")

    test = _check_test(test, data)
    max_sx = len(nodes) if max_sx is None else int(max_sx)

    whitelist = [(str(a), str(b)) for a, b in (whitelist or ())]
    blacklist = build_blacklist(
        [(str(a), str(b)) for a, b in (blacklist or ())], whitelist)

    with Tester(data, test) as tester:
        return blanket_fn(tester, node, nodes, alpha, whitelist, blacklist,
                          max_sx)


# ---------------------------------------------------------------------------
# hybrid parents and children, from R/hybrid-pc.R
# ---------------------------------------------------------------------------

def _hpc_de_pcs(tester, target, nodes, alpha, whitelist, blacklist):
    """hybrid.pc.de.pcs(): a superset of the parents and children, found
    with tests of order zero and one only.

    Cheap tests first is the whole idea of HPC: everything expensive later
    is done inside this superset rather than over all the nodes.  The
    separating sets found here are kept, because the spouse search needs
    them -- a node that separates the target from something is a candidate
    for being a spouse.
    """
    whitelisted = [y for y in nodes
                   if y != target and _listed(whitelist, (target, y),
                                              either=True)]
    blacklisted = [y for y in nodes
                   if y != target and _listed(blacklist, (target, y),
                                              both=True)]

    to_check = [n for n in nodes
                if n != target and n not in whitelisted
                and n not in blacklisted]

    association = tester.pvalues(to_check, target, [])
    keep = [n for n in to_check if association[n] <= alpha]

    # weakest association first, so the least convincing candidates face
    # the exclusion tests before the strong ones are in the conditioning set.
    keep.sort(key=lambda n: association[n], reverse=True)

    # whitelisted nodes get a p-value of zero and go first, which puts them
    # at the front of every conditioning set built below.
    pvalues = {n: 0.0 for n in whitelisted}
    pvalues.update({n: association[n] for n in keep})

    dsep_set = {}
    if len(keep) <= 1:
        return pvalues, dsep_set

    for node in keep:
        if node not in pvalues:
            continue

        # strongest association first this time: the nodes most likely to
        # separate the pair are tried before the rest.
        against = [n for n in sorted(pvalues, key=lambda k: pvalues[k])
                   if n != node]
        if not against:
            continue

        result = tester.allsubs(target, node, sx=against, min=1, max=1,
                                alpha=alpha)

        if result["p.value"] > alpha:
            del pvalues[node]
            dsep_set[node] = list(result["dsep.set"])
        else:
            pvalues[node] = max(pvalues[node], result["max.p.value"])

    return pvalues, dsep_set


def _hpc_de_sps(tester, target, nodes, pc_superset, dsep_set, alpha, max_sx):
    """hybrid.pc.de.sps(): a superset of the spouses.

    A spouse is not associated with the target on its own -- that is what
    makes it a spouse rather than a neighbour -- so it can only be found by
    conditioning on the common child.  That is what the loop over `cpc`
    does, and why this needs the separating sets from the previous phase.
    """
    superset = []

    for cpc in pc_superset:
        pvalues = {}

        for y in nodes:
            if y == target or y in pc_superset:
                continue

            dsep = dsep_set.get(y, [])
            if cpc in dsep:
                continue
            if len(dsep) + 1 > max_sx:
                continue

            p = tester.pvalue(target, y, list(dsep) + [cpc])
            if p <= alpha:
                pvalues[y] = p

        # weakest first again, for the same reason.
        ordered = sorted(pvalues, key=lambda n: pvalues[n], reverse=True)
        kept = dict.fromkeys(ordered)

        for y in ordered:
            sx = [n for n in kept if n != y]
            if not sx:
                continue

            result = tester.allsubs(target, y, sx=sx, fixed=[cpc], min=1,
                                    max=1, alpha=alpha)
            if result["p.value"] > alpha:
                del kept[y]

        for y in kept:
            if y not in superset:
                superset.append(y)

    return superset


def _hpc_filter(tester, target, pc_superset, sp_superset, nodes, alpha,
                whitelist, blacklist, max_sx):
    """hybrid.pc.filter(): keep only the candidates that no subset of the
    Markov blanket separates from the target."""
    whitelisted = [y for y in nodes
                   if y != target and _listed(whitelist, (target, y),
                                              either=True)]
    blacklisted = [y for y in nodes
                   if y != target and _listed(blacklist, (target, y),
                                              both=True)]

    candidates = [n for n in pc_superset if n not in blacklisted]
    for n in whitelisted:
        if n not in candidates:
            candidates.append(n)

    blanket = list(dict.fromkeys(list(pc_superset) + list(sp_superset)
                                 + whitelisted))
    if not blanket:
        return []

    pc = []
    for node in candidates:
        if node in whitelisted:
            continue
        result = tester.allsubs(target, node,
                                sx=[n for n in blanket if n != node],
                                max=max_sx, alpha=alpha)
        if result["p.value"] < alpha:
            pc.append(node)

    for node in whitelisted:
        if node not in pc:
            pc.append(node)

    return pc


def _hpc_nbr_search(tester, target, nodes, alpha, whitelist, blacklist,
                    max_sx, start, looking_for=None):
    """hybrid.pc.nbr.search(): a full blanket-then-filter pass, but over the
    restricted node set the supersets define rather than over everything."""
    mb = _ia_fdr_markov_blanket(tester, target, nodes, alpha, whitelist,
                                blacklist, max_sx, start=start)

    # not in the blanket, so it cannot be a neighbour either.
    if looking_for is not None and looking_for not in mb:
        return []

    return _hpc_filter(tester, target, mb, [], nodes, alpha, whitelist,
                       blacklist, max_sx)


def _hpc_heuristic(tester, target, nodes, alpha, whitelist, blacklist,
                   max_sx):
    """hybrid.pc.heuristic(): the whole of HPC for one node.

    Three passes.  Cheap tests find a superset of the neighbours and, from
    the separating sets they produce, a superset of the spouses; the
    expensive tests then run only inside that.  The last loop is HPC's "OR"
    rule: a node the target rejected is kept anyway if it, looking back,
    accepts the target -- which recovers neighbours that one node's tests
    happened to miss.
    """
    pvalues, dsep_set = _hpc_de_pcs(tester, target, nodes, alpha, whitelist,
                                    blacklist)
    pc_superset = list(pvalues)

    if len(pc_superset) < 2:
        return {"nbr": pc_superset, "mb": []}

    sp_superset = _hpc_de_sps(tester, target, nodes, pc_superset, dsep_set,
                              alpha, max_sx)

    # two candidates and no spouse leaves nothing for the expensive tests to
    # rule out: the superset is the set.
    if len(pc_superset) == 2 and not sp_superset:
        return {"nbr": pc_superset, "mb": list(pc_superset)}

    # the two strongest candidates would only be re-tested with the same
    # low-order tests, so they seed the search instead.
    start = sorted(pvalues, key=lambda n: pvalues[n])[:2]

    restricted = [target] + list(pc_superset) + list(sp_superset)

    pc = _hpc_nbr_search(tester, target, restricted, alpha, whitelist,
                         blacklist, max_sx, start)

    for node in pc_superset:
        if node in pc:
            continue

        found = _hpc_nbr_search(tester, node, restricted, alpha, whitelist,
                                blacklist, max_sx, start, looking_for=target)

        if target in found:
            pc.append(node)

    return {"nbr": pc, "mb": list(pc_superset) + list(sp_superset)}


def hpc(data, whitelist=None, blacklist=None, test=None, alpha=0.05,
        max_sx=None, undirected=True):
    """Hybrid Parents and Children, as bnlearn's hpc().

    A neighbourhood algorithm that spends its budget where it matters: two
    cheap passes narrow the candidates down to a superset of the neighbours
    and their spouses, and only then does it run the tests with large
    conditioning sets, over that superset rather than over every variable.

    Undirected by default, like the other neighbourhood algorithms and like
    bnlearn: it learns which nodes are adjacent without saying which way the
    arcs point.
    """
    return _constraint_learn(data, None, "hpc", whitelist, blacklist, test,
                             alpha, max_sx, undirected,
                             markov_blankets=False,
                             structure_fn=_hpc_heuristic)
