"""Reading and writing the three interchange formats for discrete networks.

BIF, DSC and NET are what the standard benchmark networks -- asia,
hailfinder, munin and the rest -- are distributed as, so this is how a
network that was not learned here gets in.  It mirrors bnlearn's
R/foreign-read.R and R/foreign-write.R.

All three formats say the same thing in different syntax, and the only part
that is genuinely difficult is where each conditional distribution belongs
in the table:

* BIF names the parent configuration by its levels, so the file says where
  each row goes and the order it lists them in does not matter.
* DSC names it by numeric coordinates -- except that some writers emit all
  zeros instead, in which case the coordinates carry no information and the
  order has to be assumed.
* NET names nothing at all: the distributions are a bare list and their
  position is the only clue.

For the last two, bnlearn assumes the *last* parent varies fastest, and that
assumption is reproduced here rather than replaced with a more natural one,
because getting it wrong reads a valid file into a network that is subtly
wrong instead of failing.

Only discrete networks are supported, which is also bnlearn's limit -- the
formats have no way to express a regression.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

from .fit import DiscreteNode, FittedNetwork
from .structure import BayesianNetwork

__all__ = ["read_bif", "read_dsc", "read_net", "write_bif", "write_dot",
           "write_dsc", "write_net"]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_bif(path):
    """Read a BIF file into a fitted network."""
    return _read(path, "bif")


def read_dsc(path):
    """Read a DSC file into a fitted network."""
    return _read(path, "dsc")


def read_net(path):
    """Read a NET (Hugin) file into a fitted network."""
    return _read(path, "net")


_COMMENT = {"bif": "//", "dsc": "//", "net": "%"}
_BANNER = {"bif": r"^network", "dsc": r"^belief network", "net": r"^net"}


def _read(path, fmt):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    lines = []
    for line in text.splitlines():
        line = line.split(_COMMENT[fmt])[0].strip()
        if line:
            lines.append(line)

    if not lines or not re.search(_BANNER[fmt], lines[0]):
        raise ValueError(
            f"{path} does not conform to the {fmt.upper()} format")

    body = " ".join(lines)

    nodes, levels = _parse_declarations(body, fmt, path)
    parents, tables = _parse_potentials(body, fmt, nodes, levels)

    order = list(nodes)
    children = {node: [p for p in order if node in parents.get(p, ())]
                for node in order}

    fitted = {}
    for node in order:
        own = list(parents.get(node, ()))
        variables = [node] + own
        fitted[node] = DiscreteNode(
            node, own, children[node],
            {"values": tables[node], "variables": variables,
             "levels": [levels[v] for v in variables]})

    return FittedNetwork(fitted, "custom", {})


def _blocks(body, keyword):
    """Every `keyword ... { ... }` block, as (header, contents) pairs.

    Braces are matched rather than searched for, because a level label may
    contain one and because the CPT blocks of NET files nest them.
    """
    out = []
    for match in re.finditer(keyword, body):
        opening = body.find("{", match.end())
        if opening < 0:
            continue

        depth, i = 0, opening
        while i < len(body):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        out.append((body[match.end():opening], body[opening + 1:i]))

    return out


def _parse_declarations(body, fmt, path):
    """The node labels and their levels, in the order the file declares
    them."""
    keyword = r"\bvariable\b" if fmt == "bif" else r"\bnode\b"

    if fmt == "net" and re.search(r"\bcontinuous\s+node\b", body):
        raise ValueError(
            f"{path} describes a network with continuous nodes, which the "
            "NET reader does not support")
    if fmt == "net" and re.search(r"\bdecision\s+\w+", body):
        raise ValueError(f"{path} describes a decision network")

    nodes, levels = [], {}
    for header, contents in _blocks(body, keyword):
        name = header.strip()
        if not name:
            continue

        if fmt == "net":
            found = re.search(r"states\s*=?\s*\((.*?)\)", contents, re.S)
            if not found:
                raise ValueError(f"cannot read the levels of node {name!r}")
            values = re.findall(r'"([^"]*)"', found.group(1))
        else:
            found = re.search(
                r"type\s*:?\s*discrete\s*\[\s*\d+\s*\]\s*=?\s*\{(.*?)\}",
                contents, re.S)
            if not found:
                raise ValueError(
                    f"node {name!r} is not discrete; only discrete networks "
                    f"can be read from {fmt.upper()} files")
            values = [v.strip().strip('"').strip()
                      for v in found.group(1).split(",")]

        values = [v for v in values if v != ""]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicated levels for node {name!r}")
        # R drops a node with fewer than two levels rather than reading it.
        if len(values) < 2:
            continue

        nodes.append(name)
        levels[name] = values

    if not nodes:
        raise ValueError(f"{path} declares no usable nodes")

    return nodes, levels


def _parse_potentials(body, fmt, nodes, levels):
    """Each node's parents and its conditional probability table."""
    keyword = r"\bpotential\b" if fmt == "net" else r"\bprobability\b"
    known = set(nodes)

    parents, tables = {}, {}
    for header, contents in _blocks(body, keyword):
        inside = header[header.find("(") + 1: header.rfind(")")]
        target, _, given = inside.partition("|")
        target = target.strip()

        if target not in known:
            continue

        own = [p for p in re.split(r"[,\s]+", given.strip()) if p]
        unknown = [p for p in own if p not in known]
        if unknown:
            raise ValueError(
                f"node {target!r} has the unknown parent(s) "
                + ", ".join(unknown))

        parents[target] = own
        tables[target] = _parse_table(fmt, target, own, levels, contents)

    missing = [node for node in nodes if node not in tables]
    if missing:
        raise ValueError(
            "no conditional probability table for " + ", ".join(missing))

    return parents, tables


def _parse_table(fmt, node, parents, levels, contents):
    shape = tuple(len(levels[v]) for v in [node] + list(parents))
    table = np.zeros(shape)

    if not parents:
        values = _distribution(fmt, node, contents)
        if len(values) != shape[0]:
            raise ValueError(
                f"the distribution of node {node!r} has {len(values)} "
                f"entries but the node has {shape[0]} levels")
        table[:] = _normalise(node, values)
        return table

    if fmt == "net":
        rows = _net_rows(node, contents, shape)
        coordinates = _assumed_order(shape[1:])
    else:
        rows, coordinates = _labelled_rows(fmt, node, parents, levels,
                                           contents, shape)

    expected = int(np.prod(shape[1:]))
    if len(rows) != expected:
        raise ValueError(
            f"node {node!r} needs {expected} conditional distributions but "
            f"the file has {len(rows)}")

    for values, where in zip(rows, coordinates):
        if len(values) != shape[0]:
            raise ValueError(
                f"one of the conditional distributions of node {node!r} has "
                "the wrong number of entries")
        table[(slice(None),) + tuple(where)] = _normalise(node, values)

    return table


def _distribution(fmt, node, contents):
    if fmt == "bif":
        found = re.search(r"table\s+(.*?);", contents, re.S)
    elif fmt == "dsc":
        found = re.search(r"^\s*(.*?);", contents, re.S)
    else:
        found = re.search(r"data\s*=\s*\((.*?)\)\s*;", contents, re.S)

    if not found:
        raise ValueError(f"cannot read the distribution of node {node!r}")

    separator = r"[\s,()]+" if fmt == "net" else ","
    return [float(v) for v in re.split(separator, found.group(1).strip())
            if v not in ("", "(", ")")]


def _labelled_rows(fmt, node, parents, levels, contents, shape):
    """BIF and DSC list each distribution with the configuration it belongs
    to, so the file's own order does not matter."""
    rows, coordinates = [], []

    for match in re.finditer(r"\(([^()]*)\)\s*:?\s*([^;]*);", contents):
        configuration = [v.strip().strip('"')
                         for v in match.group(1).split(",")]
        values = [float(v) for v in match.group(2).split(",") if v.strip()]

        if fmt == "bif":
            where = []
            for parent, label in zip(parents, configuration):
                if label not in levels[parent]:
                    raise ValueError(
                        f"{label!r} is not a level of {parent!r}, which is a "
                        f"parent of {node!r}")
                where.append(levels[parent].index(label))
        else:
            # DSC uses zero-based numeric coordinates.
            where = [int(v) for v in configuration]

        rows.append(values)
        coordinates.append(where)

    # Some DSC writers emit all-zero coordinates instead of real ones, which
    # says nothing; fall back to the assumed order, as bnlearn does.
    if fmt == "dsc" and rows and all(all(w == 0 for w in where)
                                     for where in coordinates):
        coordinates = _assumed_order(shape[1:])

    return rows, coordinates


def _net_rows(node, contents, shape):
    """NET lists the distributions with no coordinates at all, so they are
    read as a flat sequence and cut into distributions."""
    found = re.search(r"data\s*=(.*?);", contents, re.S)
    if not found:
        raise ValueError(f"cannot read the distribution of node {node!r}")

    flat = [float(v) for v in re.split(r"[\s,()]+", found.group(1).strip())
            if v not in ("", "(", ")")]

    total = int(np.prod(shape))
    if len(flat) != total:
        raise ValueError(
            f"node {node!r} needs {total} probabilities but the file has "
            f"{len(flat)}")

    return [flat[i:i + shape[0]] for i in range(0, total, shape[0])]


def _assumed_order(parent_shape):
    """The configurations in the order a file that does not say assumes.

    The last parent varies fastest, which is C order over the parents.  It
    is a guess -- the formats do not record it -- but it is bnlearn's guess,
    and agreeing with bnlearn is the point.
    """
    return list(itertools.product(*(range(n) for n in parent_shape)))


def _normalise(node, values):
    values = np.asarray(values, dtype=float)

    if (values < 0).any() or not np.isfinite(values).all():
        raise ValueError(
            f"one of the distributions of node {node!r} is not a vector of "
            "probabilities")

    total = values.sum()
    # A file that is a little off is rounding; one that is a lot off is
    # wrong.  Some writers emit an all-zero row, which R reads as uniform.
    if total == 0:
        return np.full(len(values), 1.0 / len(values))
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"one of the distributions of node {node!r} does not sum to one")

    return values / total


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def write_bif(path, fitted):
    """Write a discrete network out as a BIF file."""
    _write(path, fitted, "bif")


def write_dsc(path, fitted):
    """Write a discrete network out as a DSC file."""
    _write(path, fitted, "dsc")


def write_net(path, fitted):
    """Write a discrete network out as a NET (Hugin) file."""
    _write(path, fitted, "net")


_PREAMBLE = {"bif": "network unknown {\n}\n",
             "dsc": 'belief network "unknown"\n',
             "net": "net \n{ \n}\n"}


def _write(path, fitted, fmt):
    if not isinstance(fitted, FittedNetwork):
        raise TypeError(f"write_{fmt}() needs a fitted network")

    for node in fitted.nodes:
        if not isinstance(fitted[node], DiscreteNode):
            raise ValueError(
                f"only discrete Bayesian networks can be written as "
                f"{fmt.upper()}; {node!r} is not discrete")

    levels = {node: _sanitise(fitted[node].levels[0]) for node in fitted.nodes}

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_PREAMBLE[fmt])

        for node in fitted.nodes:
            handle.write(_declaration(fmt, node, levels[node]))

        for node in fitted.nodes:
            handle.write(_potential(fmt, node, fitted[node], levels))


def _sanitise(values):
    """Strip the characters the grammars use as punctuation."""
    clean = [re.sub(r"[,{}()#%]", "_", str(v)) for v in values]
    if len(set(clean)) != len(clean):
        raise ValueError("duplicated levels after sanitisation")
    return clean


def _declaration(fmt, node, levels):
    if fmt == "bif":
        return (f"variable {node} {{\n  type discrete [ {len(levels)} ] "
                "{ " + ", ".join(levels) + " };\n}\n")
    if fmt == "dsc":
        return (f"node {node} {{\n  type : discrete [ {len(levels)} ] = "
                "{ " + ", ".join(f'"{v}"' for v in levels) + " };\n}\n")
    return (f"node {node} \n{{\n  states = ( "
            + " ".join(f'"{v}"' for v in levels) + " );\n}\n")


def _number(value):
    return f"{value:.10g}"


def _potential(fmt, node, entry, levels):
    parents = list(entry.parents)
    table = entry.probabilities
    own = levels[node]

    if not parents:
        values = [_number(v) for v in table.reshape(-1)]
        if fmt == "bif":
            return (f"probability ( {node} ) {{\n  table "
                    + ", ".join(values) + ";\n}\n")
        if fmt == "dsc":
            return (f"probability ( {node} ) {{\n  "
                    + ", ".join(values) + ";\n}\n")
        return (f"potential ( {node} ) \n{{\n  data = ( "
                + " ".join(values) + " );\n}\n")

    parent_shape = table.shape[1:]

    if fmt == "net":
        # NET has no coordinates, so the order is the whole message: the
        # last parent has to vary fastest, matching what the reader assumes.
        rows = ["(" + " ".join(_number(v)
                               for v in table[(slice(None),) + where]) + ")"
                for where in _assumed_order(parent_shape)]
        # nest the rows in parentheses, innermost parent first, as Hugin
        # does.  Each row has to be parenthesised before the nesting starts,
        # or consecutive numbers run together with nothing between them.
        for size in reversed(parent_shape):
            grouped = [rows[i:i + size] for i in range(0, len(rows), size)]
            rows = ["(" + "".join(g) + ")" for g in grouped]
        body = "".join(rows)
        return (f"potential ( {node} | " + " ".join(parents)
                + f" ) \n{{\n  data = {body} ;\n}}\n")

    out = [f"probability ( {node} | " + ", ".join(parents) + " ) {\n"]

    # BIF and DSC name the configuration, and enumerate with the *first*
    # parent varying fastest -- the opposite of NET, and of the fallback the
    # reader uses when nothing is named.
    for where in itertools.product(*(range(n) for n in reversed(parent_shape))):
        where = tuple(reversed(where))
        values = ", ".join(_number(v)
                           for v in table[(slice(None),) + where])

        if fmt == "bif":
            names = ", ".join(levels[p][i] for p, i in zip(parents, where))
            out.append(f"  ({names}) {values};\n")
        else:
            names = ", ".join(str(i) for i in where)
            out.append(f"  ({names}) : {values};\n")

    out.append("}\n")
    return "".join(out)


def write_dot(path, graph):
    """Write a graph out in Graphviz's DOT language.

    An undirected arc is written with `dir=none` rather than as two opposing
    arrows, so a partially directed graph survives the round trip to a
    picture.
    """
    from .nodes import _graph

    net = _graph(graph)
    present = set(net.arcs)
    rank = {node: i for i, node in enumerate(net.nodes)}

    loose = [(a, b) for a, b in net.arcs
             if (b, a) in present and rank[a] < rank[b]]
    firm = [(a, b) for a, b in net.arcs if (b, a) not in present]

    undirected_only = bool(net.arcs) and not firm
    keyword = "graph" if undirected_only else "digraph"
    joiner = " -- " if undirected_only else " -> "

    out = [keyword + " {\n"]
    out.extend(f'  "{node}" ;\n' for node in net.nodes)
    for a, b in loose:
        out.append(f'  edge [dir=none] "{a}"{joiner}"{b}" ;\n')
    for a, b in firm:
        out.append(f'  edge [dir=forward] "{a}" -> "{b}" ;\n')
    out.append("}\n")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(out))
