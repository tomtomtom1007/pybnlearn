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
                  GaussianNode, bn_net, custom_fit, fit, predict)
from .hybrid import h2pc, mmhc, rsmax2
from .inference import cpdist, cpquery, rbn, set_seed
from .mvnorm import MultivariateNormal, gbn2mvnorm, mvnorm2gbn
from .graph import (acyclic, aracne, chow_liu, compare, connected_components,
                    cpdag, directed, empty_graph, hamming, leaf_nodes,
                    model2network, moral, node_ordering, nparams,
                    ordering2blacklist, path_exists, pdag2dag, root_nodes,
                    set2blacklist, shd, skeleton, subgraph, tiers2blacklist,
                    valid_cpdag, valid_dag, valid_ug)
from .nodes import (add_node, ancestors, arcs, children, compelled_arcs,
                    degree, descendants, directed_arcs, drop_arc, drop_edge,
                    in_degree, incident_arcs, incoming_arcs, isolated_nodes,
                    mb, narcs, nbr, nnodes, out_degree, outgoing_arcs,
                    parents, remove_node, rename_nodes, reverse_arc,
                    reversible_arcs, set_arc, set_edge, spouses,
                    undirected_arcs)
from .strength import (arc_strength, averaged_network,
                       custom_strength, inclusion_threshold)
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
    "fit", "custom_fit", "bn_net", "predict",
    # simulation and inference
    "cpdist", "cpquery", "query", "rbn", "set_seed",
    "gbn2mvnorm", "mvnorm2gbn",
    # resampling and arc strength
    "bn_cv", "boot_strength", "arc_strength", "custom_strength",
    "averaged_network", "inclusion_threshold",
    # graphs and comparison
    "compare", "cpdag", "empty_graph", "hamming", "model2network", "moral",
    "nparams", "pdag2dag", "shd", "skeleton", "subgraph",
    # graph properties
    "acyclic", "connected_components", "directed", "node_ordering",
    "path_exists", "valid_cpdag", "valid_dag", "valid_ug",
    "ordering2blacklist", "set2blacklist", "tiers2blacklist",
    # nodes and arcs
    "arcs", "narcs", "nnodes", "parents", "children", "mb", "nbr", "spouses",
    "ancestors", "descendants", "root_nodes", "leaf_nodes", "isolated_nodes",
    "degree", "in_degree", "out_degree",
    "directed_arcs", "undirected_arcs", "incoming_arcs", "outgoing_arcs",
    "incident_arcs", "compelled_arcs", "reversible_arcs",
    "set_arc", "drop_arc", "reverse_arc", "set_edge", "drop_edge",
    "add_node", "remove_node", "rename_nodes",
]

__version__ = "0.1.0.dev0"
