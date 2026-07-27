"""R's argument predicates, and the checks built out of them.

bnlearn does almost all of its argument checking through six one-line
predicates in R/sanitization-types.R -- is.positive(), is.positive.integer(),
is.probability() and friends -- and then quotes them, nearly verbatim, from
about two hundred call sites.  Reproducing the call sites without
reproducing the predicates is how a port ends up rejecting alpha = 0 (R does
not: is.probability() is a *closed* interval) while accepting maxp = 1.5 (R
does not: is.positive.integer() is not just "> 0").

So they live here, once, with the same boundaries R draws:

* a number is a single finite scalar -- length 1, not NaN, not infinite.  A
  vector of two alphas is not an alpha, and R says so rather than silently
  using the first.
* an integer is a number whose fractional part is zero, not a number that
  int() would accept.  1.5 restarts is a typo, and truncating it to 1 turns
  the typo into a slightly different search.
* a probability is in [0, 1] inclusive at both ends.

The reason to be this careful about the *upper* halves of these boundaries
is that the failure they prevent is invisible.  A rejected argument produces
a traceback; an argument quietly rounded, truncated, or clamped produces a
network, and a network is what the caller was going to publish.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import math

__all__ = ["is_number", "is_positive", "is_non_negative", "is_integer",
           "is_positive_integer", "is_non_negative_integer", "is_probability",
           "check_positive_integer", "check_positive", "check_probability",
           "check_node_labels"]


def _scalar(x):
    """The numeric value of x if it is one finite number, else None.

    bool is excluded on purpose: True is not 1 replicate, it is a mistake,
    and R would not accept a logical there either.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        # numpy scalars answer to __float__ and have a shape of (); anything
        # with a length is a vector, which R rejects as "not a single value".
        try:
            if hasattr(x, "shape") and x.shape == () and not hasattr(x, "__len__"):
                x = float(x)
            else:
                return None
        except (TypeError, ValueError):
            return None
    value = float(x)
    return value if math.isfinite(value) else None


def is_number(x):
    """is.real.number(): one finite scalar."""
    return _scalar(x) is not None


def is_positive(x):
    """is.positive()."""
    value = _scalar(x)
    return value is not None and value > 0


def is_non_negative(x):
    """is.non.negative()."""
    value = _scalar(x)
    return value is not None and value >= 0


def is_integer(x):
    """A finite scalar with nothing after the decimal point."""
    value = _scalar(x)
    return value is not None and value == int(value)


def is_positive_integer(x):
    """is.positive.integer()."""
    return is_positive(x) and is_integer(x)


def is_non_negative_integer(x):
    """is.non.negative.integer()."""
    return is_non_negative(x) and is_integer(x)


def is_probability(x):
    """is.probability(): a closed interval, 0 and 1 included.

    R admits both ends, and both mean something: a test at alpha = 0 accepts
    no arc and a threshold of 1 keeps only the unanimous ones.  Narrowing
    this to the open interval would reject calls R runs.
    """
    value = _scalar(x)
    return value is not None and 0 <= value <= 1


# ---------------------------------------------------------------------------
# the checks themselves
#
# Each returns the value coerced the way R coerces it -- as.integer() for the
# counts -- so callers can assign the result and stop thinking about the type.
# ---------------------------------------------------------------------------

def check_positive_integer(value, what, default=None, allow_infinite=False):
    """is.positive.integer(), with R's `missing or NULL means default` rule.

    `allow_infinite` is for the arguments R compares against Inf before it
    checks anything else -- maxp, max.sx, max.iter -- where "no limit" is a
    legal answer and not a very large one.
    """
    if value is None:
        return default
    if allow_infinite and _is_infinity(value):
        return math.inf
    if not is_positive_integer(value):
        raise ValueError(f"{what} must be a positive integer number")
    return int(value)


def check_positive(value, what, default=None):
    """is.positive(): positive, but not necessarily whole."""
    if value is None:
        return default
    if not is_positive(value):
        raise ValueError(f"{what} must be a positive number")
    return float(value)


def check_probability(value, what, default=None):
    """is.probability(), with R's wording for the failure."""
    if value is None:
        return default
    if not is_probability(value):
        raise ValueError(f"{what} must be a numerical value in [0,1]")
    return float(value)


def _is_infinity(value):
    try:
        return math.isinf(float(value))
    except (TypeError, ValueError):
        return False


def check_node_labels(nodes, what="nodes", min_nodes=1):
    """check.nodes() for a bare list of labels.

    The two failures here are the ones a caller reaches by accident rather
    than by typo: an empty list, which arrives from a filter that matched
    nothing, and a repeated label, which arrives from concatenating two node
    sets.  Neither has an answer -- a graph over no nodes is not a graph, and
    a graph with one node listed twice has an adjacency matrix that cannot be
    indexed by name -- so R refuses both, and returning something for them
    would mean returning something wrong.
    """
    labels = [str(n) for n in nodes]
    if len(labels) < min_nodes:
        raise ValueError(f"at least {min_nodes} {what} needed")
    seen = set()
    duplicated = sorted({n for n in labels if n in seen or seen.add(n)})
    if duplicated:
        raise ValueError("node labels must be unique; repeated: "
                         + ", ".join(duplicated))
    return labels
