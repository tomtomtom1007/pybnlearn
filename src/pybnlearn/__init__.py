"""pybnlearn: a Python port of the R package bnlearn.

The numerical core is bnlearn's own C code, compiled against a compatibility
layer that stands in for the R C API, so results agree with R rather than
merely resembling them.  See NOTICE for attribution and licensing.
"""

from ._core import BNLearnError, ci_test
from .bootstrap import CrossValidation, bn_cv, boot_strength
from .constraint import (gs, iamb, iamb_fdr, inter_iamb, mmpc, pc_stable,
                         si_hiton_pc)
from .fit import DiscreteNode, FittedNetwork, GaussianNode, fit, predict
from .hybrid import h2pc, mmhc, rsmax2
from .inference import cpdist, cpquery, rbn, set_seed
from .graph import (aracne, chow_liu, compare, cpdag, empty_graph, hamming,
                    model2network, moral, nparams, pdag2dag, shd, skeleton,
                    subgraph)
from .structure import BayesianNetwork, hc, score, tabu

__all__ = [
    "BNLearnError", "BayesianNetwork", "FittedNetwork",
    "CrossValidation", "DiscreteNode", "GaussianNode",
    # structure learning
    "aracne", "chow_liu", "gs", "hc", "iamb", "iamb_fdr", "inter_iamb",
    "mmhc", "mmpc", "pc_stable", "rsmax2", "si_hiton_pc", "tabu",
    # testing and scoring
    "ci_test", "score",
    # parameter learning
    "fit", "predict",
    # simulation and inference
    "cpdist", "cpquery", "rbn", "set_seed",
    # graphs
    "compare", "cpdag", "empty_graph", "hamming", "model2network", "moral",
    "nparams", "pdag2dag", "shd", "skeleton", "subgraph",
]

__version__ = "0.1.0.dev0"
