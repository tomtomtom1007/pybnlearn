"""Interventions, twin networks and counterfactuals.

A Bayesian network says how the variables are distributed.  A structural
causal model says how each one is *produced*: from its parents and from its
own exogenous noise, which is made an explicit node here rather than left
implicit.  That extra node is what the rest of this file is for.

`intervention` is the do-operator: fixing a variable removes the arcs into
it, because an intervened variable is no longer produced by its parents.

`twin` builds Pearl's twin network -- a second copy of every variable, run
on the *same* exogenous noise.  That shared noise is the whole point: it is
what lets you ask what would have happened to this individual, rather than
to someone else drawn from the same distribution.

`counterfactual` intervenes on the copy and leaves the original alone, so
the two can be compared.  With `merging` on, a copy that the intervention
cannot have reached is dropped and its children rewired to the original,
which is not an optimisation -- it is the statement that those variables
would have been unaffected.

This mirrors bnlearn's R/scm.R, R/causal.R and R/frontend-causal.R.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import numpy as np

from .fit import DiscreteNode, FittedNetwork, GaussianNode
from .structure import BayesianNetwork

__all__ = ["StructuralCausalModel", "as_bn", "as_scm", "counterfactual",
           "intervention", "mutilated", "twin"]


def _counterfactual_name(node):
    """The label of a node's counterfactual copy: R appends a full stop."""
    return f"{node}."


def _exogenous_name(node):
    """The label of a node's exogenous noise: R prepends a `u`."""
    return f"u{node}"


def _factual_of_counterfactual(node):
    return node[:-1] if node.endswith(".") else node


def _factual_of_exogenous(node):
    return node[1:] if node.startswith("u") else node


class StructuralCausalModel:
    """A network with its exogenous noise made explicit.

    Every variable gains a parent that stands for everything about it the
    model does not explain.  Nothing about the joint distribution changes;
    what changes is that the noise can now be referred to, which is what
    counterfactuals need.
    """

    def __init__(self, roles, nodes, arcs, twin=False, counterfactual=False):
        self.roles = {k: list(v) for k, v in roles.items()}
        self.nodes = {k: {key: list(value) for key, value in v.items()}
                      for k, v in nodes.items()}
        self.arcs = [(str(a), str(b)) for a, b in arcs]
        self.is_twin = bool(twin)
        self.is_counterfactual = bool(counterfactual)

    @property
    def factual(self):
        return list(self.roles["factual"])

    @property
    def exogenous(self):
        return list(self.roles["exogenous"])

    @property
    def counterfactual(self):
        return list(self.roles["counterfactual"])

    def __repr__(self):
        kind = ("counterfactual " if self.is_counterfactual
                else "twin " if self.is_twin else "")
        return (f"StructuralCausalModel({kind}{len(self.factual)} factual, "
                f"{len(self.counterfactual)} counterfactual, "
                f"{len(self.exogenous)} exogenous nodes, "
                f"{len(self.arcs)} arcs)")


def as_scm(network):
    """Read a network as a structural causal model.

    Nothing is added to the model in a statistical sense -- the exogenous
    node is the error term the network already had -- but it becomes a node
    you can name, which is what interventions and counterfactuals work on.
    """
    from .graph import acyclic, directed
    from .nodes import _graph

    net = _graph(network)

    if not directed(net):
        raise ValueError("the graph is only partially directed")
    if not acyclic(net, directed=True):
        raise ValueError("the graph contains cycles")

    if net.learning.get("roles"):
        return _twin_bn_to_scm(net)

    factual = list(net.nodes)
    exogenous = [_exogenous_name(n) for n in factual]

    if len(set(factual + exogenous)) != len(factual) + len(exogenous):
        raise ValueError("duplicated node labels in the causal model")

    arcs = list(net.arcs) + [(_exogenous_name(n), n) for n in factual]

    nodes = {}
    for node in factual:
        nodes[node] = {"counterfactual": [],
                       "exogenous": [_exogenous_name(node)],
                       "factual": [],
                       "parents": net.parents(node),
                       "children": net.children(node)}
    for node in exogenous:
        nodes[node] = {"counterfactual": [], "exogenous": [],
                       "factual": [_factual_of_exogenous(node)],
                       "parents": [], "children": []}

    return StructuralCausalModel(
        {"factual": factual, "exogenous": exogenous, "counterfactual": []},
        nodes, arcs)


def _twin_bn_to_scm(net):
    """from.bn.twin.to.scm.twin(): a twin network that has been flattened
    back into a graph, read as a causal model again.

    The roles have to be carried on the graph for this to be possible at
    all: which copy a node is cannot be recovered from the arcs.
    """
    roles = {k: list(v) for k, v in net.learning["roles"].items()}
    nodes = {}

    for node in net.nodes:
        if node in roles["factual"]:
            entry = {"counterfactual": [_counterfactual_name(node)],
                     "exogenous": [_exogenous_name(node)],
                     "factual": [],
                     "parents": [p for p in net.parents(node)
                                 if p in roles["factual"]],
                     "children": [c for c in net.children(node)
                                  if c in roles["factual"]]}
        elif node in roles["counterfactual"]:
            factual = _factual_of_counterfactual(node)
            entry = {"counterfactual": [],
                     "exogenous": [_exogenous_name(factual)],
                     "factual": [factual],
                     "parents": [p for p in net.parents(node)
                                 if p in roles["counterfactual"]],
                     "children": [c for c in net.children(node)
                                  if c in roles["counterfactual"]]}
        else:
            factual = _factual_of_exogenous(node)
            entry = {"counterfactual": [_counterfactual_name(factual)],
                     "exogenous": [], "factual": [factual],
                     "parents": [], "children": []}

        nodes[node] = entry

    return StructuralCausalModel(
        roles, nodes, net.arcs, twin=True,
        counterfactual=bool(net.learning.get("counterfactual")))


def as_bn(scm):
    """Turn a causal model back into a network.

    The exogenous nodes are dropped, because a Bayesian network keeps its
    error terms implicit -- except for a twin network, where dropping them
    would lose the shared noise that makes the two copies twins at all.
    """
    if not isinstance(scm, StructuralCausalModel):
        raise TypeError("as_bn() needs a StructuralCausalModel")

    keep = (list(scm.nodes) if scm.is_twin else list(scm.roles["factual"]))
    kept = set(keep)

    arcs = [(a, b) for a, b in scm.arcs if a in kept and b in kept]

    learning = {}
    if scm.is_twin:
        learning = {"algo": "twin", "roles": scm.roles,
                    "counterfactual": scm.is_counterfactual}

    return BayesianNetwork(keep, arcs, learning)


# ---------------------------------------------------------------------------
# interventions
# ---------------------------------------------------------------------------

def intervention(x, evidence):
    """Fix variables by intervention rather than by observation.

    `do(X = x)`: the arcs into X are removed, because an intervened variable
    is no longer produced by its parents.  That is what makes this different
    from conditioning, which leaves the arcs alone and lets information flow
    back up them.

    Works on a graph, a causal model, or a fitted network -- for the last,
    the node's distribution is replaced by a point mass as well.
    """
    if isinstance(x, StructuralCausalModel):
        return _intervene_scm(x, evidence)

    if isinstance(x, FittedNetwork):
        return _intervene_fitted(x, evidence)

    if isinstance(x, BayesianNetwork):
        return as_bn(_intervene_scm(as_scm(x), evidence))

    raise TypeError(
        "a BayesianNetwork, FittedNetwork or StructuralCausalModel is "
        "required")


# bnlearn spells the same operation both ways.
mutilated = intervention


def _intervene_scm(scm, evidence):
    fixed = list(evidence or {})
    unknown = [n for n in fixed if n not in scm.nodes]
    if unknown:
        raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))

    if not fixed:
        return scm

    nodes = {k: {key: list(v) for key, v in entry.items()}
             for k, entry in scm.nodes.items()}
    arcs = [(a, b) for a, b in scm.arcs if b not in fixed]

    for node in fixed:
        # the exogenous parent goes too: an intervened variable has no noise
        # of its own left, it is simply set.
        for exogenous in nodes[node]["exogenous"]:
            nodes[exogenous]["factual"] = []
        nodes[node]["exogenous"] = []

        for parent in nodes[node]["parents"]:
            nodes[parent]["children"] = [c for c in nodes[parent]["children"]
                                         if c != node]
        nodes[node]["parents"] = []

    return StructuralCausalModel(scm.roles, nodes, arcs, twin=scm.is_twin,
                                 counterfactual=scm.is_counterfactual)


def _intervene_fitted(fitted, evidence):
    """intervention.backend.fitted(): the parameters change too.

    The node's conditional distribution is replaced by one that puts all its
    mass on the value it was set to -- a degenerate factor, which is exactly
    what an intervention makes it.
    """
    evidence = dict(evidence or {})
    unknown = [n for n in evidence if n not in fitted]
    if unknown:
        raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))

    if not evidence:
        return fitted

    # Copies, not the originals: an intervention has to leave the network it
    # was given alone, and the parents' children lists are edited below.
    updated = {node: _copy_node(fitted[node]) for node in fitted.nodes}

    for node, value in evidence.items():
        entry = updated[node]

        if isinstance(entry, DiscreteNode):
            levels = list(entry.levels[0])
            if str(value) not in levels:
                raise ValueError(
                    f"{value!r} is not a level of {node!r}; levels are "
                    f"{levels}")
            mass = np.array([1.0 if level == str(value) else 0.0
                             for level in levels])
            # the parents go, the children stay: an intervened variable is
            # no longer produced by anything, but it still produces.
            replacement = DiscreteNode(
                node, [], list(entry.children),
                {"values": mass, "variables": [node], "levels": [levels]})
        else:
            replacement = GaussianNode(
                node, [], list(entry.children),
                {"coefficients": {"(Intercept)": float(value)}, "sd": 0.0})

        for parent in entry.parents:
            other = updated[parent]
            other.children = [c for c in other.children if c != node]

        updated[node] = replacement

    return FittedNetwork(updated, fitted.method, fitted.learning)


def _copy_node(entry):
    """A shallow copy of a fitted node, with its own parent and child lists."""
    if isinstance(entry, DiscreteNode):
        return DiscreteNode(
            entry.node, list(entry.parents), list(entry.children),
            {"values": entry.probabilities, "variables": list(entry.variables),
             "levels": [list(l) for l in entry.levels]})

    return GaussianNode(
        entry.node, list(entry.parents), list(entry.children),
        {"coefficients": dict(entry.coefficients), "sd": entry.sd})


# ---------------------------------------------------------------------------
# twin networks
# ---------------------------------------------------------------------------

def twin(x):
    """Pearl's twin network: a second copy of every variable, on the same
    noise.

    The copies share their exogenous parents with the originals, which is
    what makes them counterfactual rather than merely a second sample.  Ask
    what would have happened *to this individual* and the shared noise is
    the individual.
    """
    if isinstance(x, StructuralCausalModel):
        return _twin_scm(x)

    if isinstance(x, FittedNetwork):
        return _twin_fitted(x)

    if isinstance(x, BayesianNetwork):
        if x.learning.get("roles"):
            return x
        return as_bn(_twin_scm(as_scm(x)))

    raise TypeError(
        "a BayesianNetwork, FittedNetwork or StructuralCausalModel is "
        "required")


def _twin_scm(scm):
    if scm.is_twin:
        return scm

    factual = list(scm.roles["factual"])
    counterfactual = [_counterfactual_name(n) for n in factual]

    everything = factual + list(scm.roles["exogenous"]) + counterfactual
    if len(set(everything)) != len(everything):
        raise ValueError("duplicated node labels in the twin network")

    nodes = {k: {key: list(v) for key, v in entry.items()}
             for k, entry in scm.nodes.items()}

    for node in factual:
        nodes[_counterfactual_name(node)] = {
            "counterfactual": [],
            "exogenous": list(nodes[node]["exogenous"]),
            "factual": [node],
            "parents": [_counterfactual_name(p)
                        for p in nodes[node]["parents"]],
            "children": [_counterfactual_name(c)
                         for c in nodes[node]["children"]]}

    for node in factual:
        nodes[node]["counterfactual"] = [_counterfactual_name(node)]
    for node in scm.roles["exogenous"]:
        nodes[node]["counterfactual"] = [
            _counterfactual_name(f) for f in nodes[node]["factual"]]

    between = [(a, b) for a, b in scm.arcs
               if a in set(factual) and b in set(factual)]
    from_noise = [(a, b) for a, b in scm.arcs
                  if a in set(scm.roles["exogenous"]) and b in set(factual)]

    arcs = (between
            + [(_counterfactual_name(a), _counterfactual_name(b))
               for a, b in between]
            + from_noise
            + [(a, _counterfactual_name(b)) for a, b in from_noise])

    roles = dict(scm.roles)
    roles["counterfactual"] = counterfactual

    return StructuralCausalModel(roles, nodes, arcs, twin=True)


def _twin_fitted(fitted):
    """twin.backend.fitted(): the parameterised twin, for Gaussian networks.

    The noise stops being a residual variance and becomes a node with that
    variance, entering its two children with a coefficient of one -- so both
    copies are now deterministic given their parents, and everything random
    about them is shared.
    """
    if fitted.learning.get("roles"):
        return fitted

    for node in fitted.nodes:
        if not isinstance(fitted[node], GaussianNode):
            raise ValueError(
                "a parameterised twin network is only defined for Gaussian "
                f"networks; {node!r} is not Gaussian")

    original = list(fitted.nodes)
    everything = (original + [_counterfactual_name(n) for n in original]
                  + [_exogenous_name(n) for n in original])
    if len(set(everything)) != len(everything):
        raise ValueError("duplicated node labels in the twin network")

    built = {}
    for node in original:
        entry = fitted[node]
        noise = _exogenous_name(node)

        coefficients = dict(entry.coefficients)
        coefficients[noise] = 1.0
        built[node] = GaussianNode(
            node, list(entry.parents) + [noise], list(entry.children),
            {"coefficients": coefficients, "sd": 0.0})

        built[noise] = GaussianNode(
            noise, [], [node, _counterfactual_name(node)],
            {"coefficients": {"(Intercept)": 0.0}, "sd": entry.sd})

        copied = {"(Intercept)": entry.coefficients["(Intercept)"]}
        for parent in entry.parents:
            copied[_counterfactual_name(parent)] = entry.coefficients[parent]
        copied[noise] = 1.0

        built[_counterfactual_name(node)] = GaussianNode(
            _counterfactual_name(node),
            [_counterfactual_name(p) for p in entry.parents] + [noise],
            [_counterfactual_name(c) for c in entry.children],
            {"coefficients": copied, "sd": 0.0})

    # R lists the originals, then all the copies, then all the noise -- not
    # interleaved.  The order is the network's node order, so it decides how
    # everything downstream is laid out.
    ordered = {}
    for node in original:
        ordered[node] = built[node]
    for node in original:
        ordered[_counterfactual_name(node)] = built[
            _counterfactual_name(node)]
    for node in original:
        ordered[_exogenous_name(node)] = built[_exogenous_name(node)]

    roles = {"factual": original,
             "exogenous": [_exogenous_name(n) for n in original],
             "counterfactual": [_counterfactual_name(n) for n in original]}

    return FittedNetwork(ordered, "mle-g", {"algo": "twin", "roles": roles})


# ---------------------------------------------------------------------------
# counterfactuals
# ---------------------------------------------------------------------------

def counterfactual(x, evidence, merging=True):
    """Set up the network for a counterfactual question.

    Build the twin, intervene on the *copy*, and leave the original alone,
    so that the two can be compared on the same noise.

    `merging` drops the copies the intervention cannot have reached and
    rewires their children back to the originals.  That is a claim, not a
    tidy-up: it says those variables would have been exactly as observed.
    """
    if isinstance(x, StructuralCausalModel):
        if x.is_counterfactual:
            raise ValueError("this is already a counterfactual network")
        return _counterfactual_scm(_twin_scm(x), evidence, merging)

    if isinstance(x, FittedNetwork):
        if x.learning.get("counterfactual"):
            raise ValueError("this is already a counterfactual network")
        return _counterfactual_fitted(_twin_fitted(x), evidence, merging)

    if isinstance(x, BayesianNetwork):
        if x.learning.get("counterfactual"):
            raise ValueError("this is already a counterfactual network")
        return as_bn(_counterfactual_scm(_twin_scm(as_scm(x)), evidence,
                                         merging))

    raise TypeError(
        "a BayesianNetwork, FittedNetwork or StructuralCausalModel is "
        "required")


def _check_counterfactual_evidence(scm_or_fitted, evidence, nodes):
    """The intervention has to land on the copies, not on the originals:
    intervening on the original would change what actually happened."""
    evidence = dict(evidence or {})
    if not evidence:
        raise ValueError("a counterfactual needs an intervention")

    wrong = [n for n in evidence if not str(n).endswith(".")]
    if wrong:
        raise ValueError(
            "a counterfactual intervenes on the counterfactual copies, whose "
            "labels end in a full stop; " + ", ".join(sorted(wrong))
            + " name the factual nodes. Did you mean "
            + ", ".join(sorted(_counterfactual_name(n) for n in wrong)) + "?")

    unknown = [n for n in evidence if n not in nodes]
    if unknown:
        raise ValueError("unknown node(s): " + ", ".join(sorted(unknown)))

    return evidence


def _counterfactual_scm(twin_scm, evidence, merging):
    evidence = _check_counterfactual_evidence(twin_scm, evidence,
                                              twin_scm.nodes)

    result = _intervene_scm(twin_scm, evidence)
    if merging:
        result = _merge_counterfactual_nodes(result, evidence)

    result.is_counterfactual = True
    return result


def _merge_counterfactual_nodes(twin_scm, evidence):
    """counterfactual.node.merging(): drop the copies nothing has changed.

    A copy differs from its original only if the intervention reached it,
    directly or through a parent.  Everything else is the same variable
    twice over, so it is dropped and its children point at the original --
    which is the formal statement that those variables are unaffected.
    """
    fixed = set(evidence)

    droppable = []
    for node in twin_scm.roles["counterfactual"]:
        if node in fixed:
            continue
        if any(p in fixed for p in twin_scm.nodes[node]["parents"]):
            continue
        droppable.append(node)

    dropped = set(droppable)
    nodes = {k: {key: list(v) for key, v in entry.items()}
             for k, entry in twin_scm.nodes.items()}
    arcs = list(twin_scm.arcs)

    for node in droppable:
        factual = nodes[node]["factual"][0]

        nodes[factual]["counterfactual"] = []
        for exogenous in nodes[node]["exogenous"]:
            nodes[exogenous]["counterfactual"] = []

        keep = [c for c in nodes[node]["children"] if c not in dropped]
        for child in keep:
            nodes[child]["parents"] = [
                factual if p == node else p for p in nodes[child]["parents"]]

        arcs = [(factual if a == node and b in keep else a, b)
                for a, b in arcs]

        del nodes[node]

    roles = {k: list(v) for k, v in twin_scm.roles.items()}
    roles["counterfactual"] = [n for n in roles["counterfactual"]
                               if n not in dropped]

    arcs = [(a, b) for a, b in arcs if a not in dropped and b not in dropped]

    return StructuralCausalModel(roles, nodes, arcs, twin=True)


def _counterfactual_fitted(twin_fitted, evidence, merging):
    evidence = _check_counterfactual_evidence(twin_fitted, evidence,
                                              set(twin_fitted.nodes))

    intervened = _intervene_fitted(twin_fitted, evidence)

    if not merging:
        learning = dict(intervened.learning)
        learning["counterfactual"] = True
        return FittedNetwork({n: intervened[n] for n in intervened.nodes},
                             intervened.method, learning)

    from .nodes import _graph

    reduced = as_bn(_merge_counterfactual_nodes(
        _twin_bn_to_scm(_graph(intervened)), evidence))

    fitted = {}
    for node in reduced.nodes:
        entry = intervened[node]
        parents = reduced.parents(node)

        coefficients = dict(entry.coefficients)
        for name in list(coefficients):
            if name != "(Intercept)" and name not in parents:
                # a merged copy: the coefficient now applies to the original.
                coefficients[_factual_of_counterfactual(name)] = (
                    coefficients.pop(name))

        fitted[node] = GaussianNode(
            node, parents, reduced.children(node),
            {"coefficients": coefficients, "sd": entry.sd})

    learning = dict(reduced.learning)
    learning["counterfactual"] = True

    return FittedNetwork(fitted, "mle-g", learning)
