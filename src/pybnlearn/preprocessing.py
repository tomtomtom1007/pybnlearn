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

from ._core import configuration_factor, discretize_joint, discretize_marginal

__all__ = ["configs", "discretize"]

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

    if isinstance(breaks, int):
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


def configs(data, all=True):
    """The joint configuration of several discrete variables, as one factor.

    With `all` left on, every combination the levels allow becomes a level of
    the result, whether it occurs in the data or not -- which is what keeps
    the numbering stable between two data sets over the same variables.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    return configuration_factor(data, all=bool(all))
