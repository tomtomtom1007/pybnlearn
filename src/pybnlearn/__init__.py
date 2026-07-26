"""pybnlearn: a Python port of the R package bnlearn.

The numerical core is bnlearn's own C code, compiled against a compatibility
layer that stands in for the R C API, so results agree with R rather than
merely resembling them.  See NOTICE for attribution and licensing.
"""

from ._core import BNLearnError, ci_test
from .constraint import gs, iamb, inter_iamb
from .graph import (aracne, chow_liu, compare, cpdag, empty_graph, hamming,
                    model2network, moral, nparams, pdag2dag, shd, skeleton,
                    subgraph)
from .structure import BayesianNetwork, hc, score, tabu

__all__ = [
    "BNLearnError", "BayesianNetwork",
    # structure learning
    "aracne", "chow_liu", "gs", "hc", "iamb", "inter_iamb", "tabu",
    # testing and scoring
    "ci_test", "score",
    # graphs
    "compare", "cpdag", "empty_graph", "hamming", "model2network", "moral",
    "nparams", "pdag2dag", "shd", "skeleton", "subgraph",
]

__version__ = "0.1.0.dev0"
