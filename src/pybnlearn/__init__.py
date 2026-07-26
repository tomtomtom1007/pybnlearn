"""pybnlearn: a Python port of the R package bnlearn.

The numerical core is bnlearn's own C code, compiled against a compatibility
layer that stands in for the R C API, so results agree with R rather than
merely resembling them.  See NOTICE for attribution and licensing.
"""

from ._core import BNLearnError, ci_test
from .bootstrap import CrossValidation, bn_cv, boot_strength
from .classifiers import classify, naive_bayes, tree_bayes
from .constraint import (gs, iamb, iamb_fdr, inter_iamb, mmpc, pc_stable,
                         si_hiton_pc)
from .exact import Factor, query
from .fit import (ConditionalGaussianNode, DiscreteNode, FittedNetwork,
                  GaussianNode, fit, predict)
from .hybrid import h2pc, mmhc, rsmax2
from .inference import cpdist, cpquery, rbn, set_seed
from .mvnorm import MultivariateNormal, gbn2mvnorm, mvnorm2gbn
from .graph import (aracne, chow_liu, compare, cpdag, empty_graph, hamming,
                    model2network, moral, nparams, pdag2dag, shd, skeleton,
                    subgraph)
from .structure import BayesianNetwork, hc, score, tabu

__all__ = [
    # types
    "BNLearnError", "BayesianNetwork", "ConditionalGaussianNode",
    "CrossValidation", "DiscreteNode",
    "Factor", "FittedNetwork", "GaussianNode", "MultivariateNormal",
    # structure learning: score-based, constraint-based, hybrid, pairwise
    "hc", "tabu",
    "gs", "iamb", "iamb_fdr", "inter_iamb", "mmpc", "pc_stable",
    "si_hiton_pc",
    "h2pc", "mmhc", "rsmax2",
    "aracne", "chow_liu",
    # classifiers
    "classify", "naive_bayes", "tree_bayes",
    # testing and scoring
    "ci_test", "score",
    # parameter learning and prediction
    "fit", "predict",
    # simulation and inference
    "cpdist", "cpquery", "query", "rbn", "set_seed",
    "gbn2mvnorm", "mvnorm2gbn",
    # resampling
    "bn_cv", "boot_strength",
    # graphs and comparison
    "compare", "cpdag", "empty_graph", "hamming", "model2network", "moral",
    "nparams", "pdag2dag", "shd", "skeleton", "subgraph",
]

__version__ = "0.1.0.dev0"
