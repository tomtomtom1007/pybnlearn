"""Turning data into something a discrete network can be learned from.

This mirrors bnlearn's R/frontend-preprocessing.R.

`discretize` has two kinds of method and the difference matters.  "quantile"
and "interval" cut each variable up on its own, which is fast and loses
whatever the variables said about each other.  "hartemink" starts from a
fine marginal discretization and then repeatedly collapses the pair of
adjacent intervals that costs the least mutual information with the *other*
variables -- so the breakpoints of one variable depend on all the rest,
which is the point of discretizing before learning a network at all.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

from __future__ import annotations

import pandas as pd

from ._core import (configuration_factor, dedup_columns,
                    discretize_joint, discretize_marginal)
from ._validate import is_number, is_probability

__all__ = ["configs", "dedup", "discretize"]

_METHODS = ("quantile", "interval", "hartemink")


def discretize(data, method="quantile", breaks=3, ordered=False, idisc=None,
               ibreaks=None):
    """Cut continuous variables into intervals.

    Parameters
    ----------
    data : pandas.DataFrame
    method : {"quantile", "interval", "hartemink"}
        The first two work on one variable at a time; "hartemink" chooses the
        breakpoints jointly, keeping as much mutual information between the
        variables as it can.
    breaks : int or sequence of int
        How many intervals per variable.
    ordered : bool or sequence of bool
        Whether the resulting factors are ordered.
    idisc : {"quantile", "interval"}, optional
        For "hartemink", the marginal discretization it starts from.
    ibreaks : int, optional
        For "hartemink", how many intervals to start from; the default
        depends on how much data there is, as R's does.

    Returns
    -------
    pandas.DataFrame with every column categorical.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if method not in _METHODS:
        raise ValueError("method must be one of " + ", ".join(_METHODS))

    columns = [str(c) for c in data.columns]

    # A single count applies to every variable.  R truncates a fractional
    # one on its way into C rather than refusing it, so 2.5 intervals means
    # 2; matching that is what keeps this from raising where R returns.
    if is_number(breaks):
        breaks = [breaks] * len(columns)
    breaks = [int(b) for b in breaks]
    if len(breaks) != len(columns):
        raise ValueError("breaks must have one element per variable")
    if any(b < 2 for b in breaks):
        raise ValueError("each variable needs at least two intervals")

    if isinstance(ordered, bool):
        ordered = [ordered] * len(columns)
    ordered = [bool(o) for o in ordered]
    if len(ordered) != len(columns):
        raise ValueError("ordered must have one element per variable")

    if method != "hartemink":
        result = discretize_marginal(data, method, breaks, ordered)
    else:
        if len(columns) < 2:
            raise ValueError(
                "at least two variables are needed to compute mutual "
                "information")

        idisc = idisc or "quantile"
        if idisc not in ("quantile", "interval"):
            raise ValueError("idisc must be 'quantile' or 'interval'")

        if ibreaks is None:
            ibreaks = _default_ibreaks(len(data))
        ibreaks = int(ibreaks)
        if any(ibreaks < b for b in breaks):
            raise ValueError(
                "the initial number of breaks is smaller than the final one")

        result = discretize_joint(data, method, breaks, ordered, idisc,
                                  [ibreaks] * len(columns))

    return pd.DataFrame(result, columns=list(result), index=data.index)


def _default_ibreaks(n):
    """check.ibreaks(): how fine the initial discretization is, which R
    picks from the sample size rather than from the data."""
    if n > 500:
        return 50
    if n > 100:
        return 20
    if n > 50:
        return 10
    if n > 10:
        return 5
    return n


def dedup(data, threshold=0.90):
    """Drop variables that say the same thing as an earlier one.

    Two variables correlated above the threshold carry the same information
    for a network's purposes, and keeping both makes the regressions
    singular.  The *earlier* of the pair is kept, so the column order
    decides which survives.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not is_probability(threshold):
        raise ValueError("the correlation threshold must be in [0, 1]")

    # The correlation this is built on is only defined for numbers, and the C
    # backend does not check: handed a factor it reads the level codes as
    # doubles off a buffer that is not there and takes the interpreter down
    # with it.  R has the same hole -- dedup() with no method= segfaults just
    # as readily -- but it does at least refuse the call when the label is
    # given explicitly, which is the check being reproduced here.
    categorical = [str(c) for c in data.columns
                   if not pd.api.types.is_numeric_dtype(data[c])]
    if categorical:
        raise ValueError(
            "method 'cor' may only be used with continuous data; "
            + ", ".join(categorical[:5])
            + (", ..." if len(categorical) > 5 else "")
            + (" is" if len(categorical) == 1 else " are") + " not numeric")

    kept = dedup_columns(data, float(threshold))
    return pd.DataFrame(kept, columns=list(kept), index=data.index)


def configs(data, all=True):
    """The joint configuration of several discrete variables, as one factor.

    With `all` left on, every combination the levels allow becomes a level of
    the result, whether it occurs in the data or not -- which is what keeps
    the numbering stable between two data sets over the same variables.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    return configuration_factor(data, all=bool(all))
