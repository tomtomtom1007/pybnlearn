"""pybnlearn: a Python port of the R package bnlearn.

The numerical core is bnlearn's own C code, compiled against a compatibility
layer that stands in for the R C API, so results agree with R rather than
merely resembling them.  See NOTICE for attribution and licensing.
"""

from ._core import (BNLearnError, ci_test, increment_test_counter,
                    reset_test_counter, test_counter)
from .bootstrap import CrossValidation, bn_boot, bn_cv, boot_strength, loss
from .classifiers import classify, naive_bayes, tree_bayes
from .causal import (StructuralCausalModel, as_bn, as_scm,
                     counterfactual, intervention, mutilated, twin)
from .constraint import (fast_iamb, gs, hpc, iamb, iamb_fdr, inter_iamb,
                         learn_mb, learn_nbr, mmpc, pc_stable, si_hiton_pc)
from .divergence import H, KL
from .exact import Factor, query
from .fit import (ConditionalGaussianNode, DiscreteNode, FittedNetwork,
                  GaussianNode, bn_net, custom_fit, fit, identifiable,
                  predict, singular)
from .foreign import (read_bif, read_dsc, read_net, write_bif,
                      write_dot, write_dsc, write_net)
from .hybrid import h2pc, mmhc, rsmax2
from .inference import cpdist, cpquery, rbn, set_seed
from .lingam import direct_lingam, lingam_ordering
from .missing import impute, structural_em
from .mvnorm import MultivariateNormal, gbn2mvnorm, mvnorm2gbn
from .graph import (acyclic, aracne, cextend, cextend_all, chow_liu,
                    colliders, compare, complete_graph, connected_components,
                    count_graphs, cpdag, directed, dsep, empty_graph, hamming,
                    leaf_nodes, perturb,
                    model2network, moral, node_ordering, nparams,
                    ordering2blacklist, path_exists, pdag2dag, root_nodes,
                    random_graph, set2blacklist, shd, shielded_colliders,
                    sid, skeleton, subgraph, tiers2blacklist,
                    unshielded_colliders, valid_cpdag, valid_dag, valid_ug,
                    vstructs)
from .nodes import (add_node, alst, ancestors, arcs, children,
                    compelled_arcs,
                    degree, descendants, directed_arcs, drop_arc, drop_edge,
                    in_degree, incident_arcs, incoming_arcs, isolated_nodes,
                    mb, narcs, nbr, nnodes, out_degree, outgoing_arcs,
                    parents, remove_node, rename_nodes, reverse_arc,
                    reversible_arcs, set_arc, set_edge, spouses,
                    undirected_arcs)
from .strength import (arc_strength, averaged_network, bf_strength,
                       custom_strength, inclusion_threshold)
from .preprocessing import configs, dedup, discretize
from .structure import (BF, BayesianNetwork, alpha_star, blacklist,
                        hc, ntests, score, tabu, whitelist)

__all__ = [
    # types
    "BNLearnError", "BayesianNetwork", "ConditionalGaussianNode",
    "CrossValidation", "DiscreteNode",
    "Factor", "FittedNetwork", "GaussianNode", "MultivariateNormal",
    "StructuralCausalModel",
    # structure learning: score-based, constraint-based, hybrid, pairwise
    "hc", "tabu",
    "gs", "iamb", "iamb_fdr", "inter_iamb", "fast_iamb", "mmpc", "pc_stable",
    "si_hiton_pc", "hpc", "learn_mb", "learn_nbr",
    "h2pc", "mmhc", "rsmax2",
    "direct_lingam", "lingam_ordering",
    "aracne", "chow_liu",
    # classifiers
    "classify", "naive_bayes", "tree_bayes",
    # testing and scoring
    "ci_test", "score", "alpha_star", "BF", "H", "KL",
    "whitelist", "blacklist", "ntests",
    # preprocessing
    "discretize", "configs", "dedup",
    # counters
    "test_counter", "reset_test_counter", "increment_test_counter",
    # parameter learning and prediction
    "fit", "custom_fit", "bn_net", "predict", "identifiable", "singular",
    # incomplete data
    "impute", "structural_em",
    # causal inference
    "as_scm", "as_bn", "intervention", "mutilated", "twin", "counterfactual",
    # simulation and inference
    "cpdist", "cpquery", "query", "rbn", "set_seed",
    "gbn2mvnorm", "mvnorm2gbn",
    # resampling and arc strength
    "bn_cv", "bn_boot", "loss", "boot_strength", "arc_strength",
    "custom_strength", "bf_strength", "averaged_network",
    "inclusion_threshold",
    # interchange formats
    "read_bif", "read_dsc", "read_net", "write_bif", "write_dsc",
    "write_net", "write_dot",
    # graphs and comparison
    "compare", "cpdag", "empty_graph", "hamming", "model2network", "moral",
    "nparams", "pdag2dag", "shd", "skeleton", "subgraph",
    # graph properties
    "acyclic", "connected_components", "directed", "node_ordering",
    "path_exists", "valid_cpdag", "valid_dag", "valid_ug",
    "dsep", "colliders", "unshielded_colliders", "shielded_colliders",
    "vstructs", "cextend", "cextend_all", "sid", "complete_graph",
    "random_graph", "perturb", "count_graphs",
    "ordering2blacklist", "set2blacklist", "tiers2blacklist",
    # nodes and arcs
    "arcs", "alst", "narcs", "nnodes", "parents", "children", "mb", "nbr", "spouses",
    "ancestors", "descendants", "root_nodes", "leaf_nodes", "isolated_nodes",
    "degree", "in_degree", "out_degree",
    "directed_arcs", "undirected_arcs", "incoming_arcs", "outgoing_arcs",
    "incident_arcs", "compelled_arcs", "reversible_arcs",
    "set_arc", "drop_arc", "reverse_arc", "set_edge", "drop_edge",
    "add_node", "remove_node", "rename_nodes",
]

__version__ = "0.1.0.dev0"
