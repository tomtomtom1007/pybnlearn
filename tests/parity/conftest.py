"""Shared loading of the fixture data sets.

Two things have to be restored that a CSV round trip loses, and both have
silently corrupted results here before:

* pandas reads the strings "None", "NA" and friends as missing by default, and
  bnlearn's insurance data set has a category literally called "None".  Missing
  values reach the discrete C code as INT_MIN and get used as a table index.

* write.csv() does not record the order of a factor's levels, and pandas
  re-derives categories alphabetically.  Most of the suite does not care --
  mutual information and the network scores are invariant to it -- but a
  conditional probability table is indexed by level, so lizards$Species, whose
  levels R orders Sagrei then Distichus, would come back transposed.

Copyright (C) 2026 the pybnlearn authors.
Licensed under the GNU General Public License version 3 or later.
"""

import json
import pathlib

import pandas as pd
import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# data sets whose columns are all numeric
CONTINUOUS = {"gaussian.test", "marks"}

# data sets that mix factors with numeric columns, where which is which has to
# be inferred per column rather than declared for the whole frame
MIXED = {"clgaussian.test", "cgsmall"}


def _levels():
    path = FIXTURES / "levels.json"
    return json.loads(path.read_text()) if path.exists() else {}


@pytest.fixture(scope="session")
def datasets():
    levels = _levels()
    loaded = {}

    for path in sorted(FIXTURES.glob("*.csv")):
        name = path.stem

        if name in CONTINUOUS:
            loaded[name] = pd.read_csv(path, dtype="float64")
            continue

        if name in MIXED:
            frame = pd.read_csv(path, keep_default_na=False, na_values=[])
            for column in frame.columns:
                if not pd.api.types.is_numeric_dtype(frame[column]):
                    frame[column] = frame[column].astype("category")
        else:
            frame = pd.read_csv(path, dtype="category",
                                keep_default_na=False, na_values=[])

        for column, order in levels.get(name, {}).items():
            if column in frame.columns:
                frame[column] = frame[column].cat.reorder_categories(order)

        loaded[name] = frame

    return loaded
