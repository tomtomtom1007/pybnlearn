# pybnlearn

A Python port of [bnlearn](https://www.bnlearn.com/), Marco Scutari's R package
for Bayesian network structure learning, parameter learning and inference.

> **Status: early.** Conditional independence testing, network scoring, and
> both score-based and constraint-based structure learning work and are checked
> against R. Much of bnlearn's API is not ported yet — see
> [What works](#what-works).

## What makes this a port rather than a reimplementation

The numerical core *is* bnlearn's C code. The sources under `src/c/bnlearn/`
are byte-identical to the CRAN 5.2.1 tarball; what makes them build without R
is a compatibility layer (`src/c/compat/`) that reimplements the subset of the
R C API bnlearn uses — the SEXP object model, attributes, `allocVector`,
`error()` and about sixty other entry points. Forwarding headers let the
vendored sources keep including `<R.h>` and friends by their usual names, so
tracking a new bnlearn release means dropping in a new tarball.

One thing here is *not* a port: bnlearn has no exact inference over *discrete*
networks of its own, so `query()` implements the junction tree directly rather
than delegating to the gRain package as bnlearn does. Its tests compare numbers
against gRain, which is a weaker claim than the rest of the suite makes, and
`src/pybnlearn/exact.py` says why. Gaussian networks are a port: bnlearn turns
those into a multivariate normal itself, and `src/pybnlearn/mvnorm.py` follows
it.

R's own standalone maths library is vendored too (`src/c/nmath/`), along with
R's Mersenne-Twister, so distribution functions and random streams agree with R
bit for bit. That is what makes it meaningful to assert that results *match*
rather than *approximate*: the parity suite compares against numbers generated
by R itself, at a relative tolerance of 1e-12.

Because bnlearn is GPL-2 | GPL-3, pybnlearn is distributed under the GPL v3 or
later. See `NOTICE` for attribution.

This project is unrelated to the PyPI package called `bnlearn`, which is a
separate MIT-licensed project built on pgmpy.

## Install

```bash
pip install pybnlearn
```

## Use

```python
import pandas as pd
import pybnlearn

data = pd.read_csv("learning.test.csv", dtype="category")

# conditional independence testing
pybnlearn.ci_test(data, "A", "B", sx=["C"], test="mi")
# {'statistic': {'mi': 2347.258587894275}, 'parameter': {'df': 12.0}, ...}

# structure learning
net = pybnlearn.hc(data, score="bic")
net.modelstring()          # '[A][C][F][B|A][D|A:C][E|B:F]'
net.arcs                   # [('A', 'B'), ('A', 'D'), ...]
pybnlearn.score(net, data) # -24006.734232498

# constraint-based learning: these return a partially directed graph
pybnlearn.gs(data, test="mi", alpha=0.05).arcs
pybnlearn.iamb(data).arcs
pybnlearn.inter_iamb(data, undirected=True).arcs

# parameter learning
fitted = pybnlearn.fit(net, data)
fitted["B"].probabilities        # the conditional probability table
fitted["B"].as_frame()           # ... in long format

# inference: seeded sampling reproduces R's set.seed() exactly
pybnlearn.set_seed(1)
pybnlearn.cpquery(fitted, {"B": "a"}, {"A": "a"}, method="ls", n=10000)
pybnlearn.rbn(fitted, 100)       # sample from the network

# ... or compute it exactly, no sampling involved
pybnlearn.query(fitted, "B", {"A": "a"}).values

# resampling
pybnlearn.set_seed(1)
pybnlearn.boot_strength(data, algorithm="hc", replicates=200)
pybnlearn.bn_cv(data, "hc", k=10).mean

# constraints
pybnlearn.hc(data, whitelist=[("A", "F")], blacklist=[("A", "B")], maxp=3)

# graph priors: what you believe about the structure before seeing the data.
# A whitelist forces an arc; a prior only leans on it, so on data this
# decisive all three still land on the network above -- the scores differ.
pybnlearn.score(net, data, type="bde", prior="vsp", beta=0.1)
pybnlearn.score(net, data, type="bde", prior="marginal", beta=0.9)
pybnlearn.score(net, data, type="bde", prior="cs", beta=pd.DataFrame(
    {"from": ["A", "B"], "to": ["B", "C"], "prob": [0.9, 0.05]}))
pybnlearn.hc(data, score="bde", prior="vsp", beta=0.1)

# the whole graph at once, in and out
pybnlearn.amat(net)              # a labelled square frame of 0s and 1s
pybnlearn.set_amat(net, matrix)  # ... and back to a network
```

### Two notes on `pandas.read_csv` and categorical data

**Missing values.** `read_csv` treats `NA`, `None`, `N/A` and similar strings
as missing by default. bnlearn's own `insurance` data set has a category
literally called `"None"`, so the default reads 14681 of its rows as missing.
Pass `keep_default_na=False, na_values=[]` when the labels are meant to be
data. pybnlearn raises rather than letting missing values reach the C core, but
it cannot tell that a category was lost before it ever saw the frame.

**Level order.** `read_csv` orders categories alphabetically. Most of the
library does not care — mutual information and the network scores are invariant
to it — but a conditional probability table is *indexed* by level, so `fit()`
returns its axes in whatever order the categories are in. If the order matters
to you, set it explicitly:

```python
data["Species"] = data["Species"].cat.reorder_categories(["Sagrei", "Distichus"])
```

## What works

| | |
|---|---|
| Conditional independence tests | `mi`, `mi-adf`, `mi-sh`, `x2`, `x2-adf` (discrete); `cor`, `zf`, `mi-g`, `mi-g-sh` (Gaussian) |
| Scores | `loglik`, `aic`, `bic`, `bde`, `bds`, `bdj`, `k2`, `fnml`, `qnml`; `loglik-g`, `aic-g`, `bic-g`, `bge`; `loglik-cg`, `aic-cg`, `bic-cg`, `ebic-cg` |
| Structure learning (score-based) | `hc`, `tabu` — whitelists, blacklists, `maxp`, arbitrary starting networks, random restarts |
| Structure learning (constraint-based) | `pc_stable`, `gs`, `iamb`, `inter_iamb`, `iamb_fdr`, `fast_iamb`, `mmpc`, `si_hiton_pc`, `hpc` — whitelists, blacklists, `alpha`, `max_sx`, `undirected` |
| Structure learning (local) | `learn_mb`, `learn_nbr` — one node's Markov blanket or neighbourhood, without the whole network |
| Structure learning (hybrid) | `mmhc`, `h2pc`, `rsmax2` — any ported restrict/maximize pair |
| Structure learning (pairwise) | `chow_liu`, `aracne` |
| Causal discovery | `lingam_ordering` — the causal ordering non-Gaussianity identifies; `direct_lingam` |
| Causal inference | `as_scm`, `intervention` (`mutilated`), `twin`, `counterfactual` |
| Graphs | `cpdag`, `moral`, `skeleton`, `pdag2dag`, `subgraph`, `empty_graph`, `model2network`, topological ordering |
| Comparison | `shd`, `hamming`, `compare`, `nparams`, `sid` |
| Information | `H` (entropy), `KL` (divergence), `BF` (Bayes factor), `alpha_star` |
| Graph properties | `acyclic`, `directed`, `valid_dag`, `valid_cpdag`, `valid_ug`, `path_exists`, `connected_components`, `node_ordering`, `dsep` |
| Colliders and equivalence | `colliders`, `vstructs`, `shielded_colliders`, `unshielded_colliders`, `cextend`, `cextend_all` |
| Generating graphs | `random_graph` (ordered, ic-dag, melancon), `complete_graph`, `empty_graph`, `perturb`, `count_graphs` |
| Preprocessing | `discretize` (quantile, interval, Hartemink), `configs`, `dedup` |
| Nodes and arcs | `amat`, `set_amat`, `parents`, `children`, `mb`, `nbr`, `spouses`, `ancestors`, `descendants`, `root_nodes`, `leaf_nodes`, `isolated_nodes`, `degree`, `in_degree`, `out_degree`, `arcs`, `narcs`, `nnodes`, `directed_arcs`, `undirected_arcs`, `incoming_arcs`, `outgoing_arcs`, `incident_arcs`, `compelled_arcs`, `reversible_arcs` |
| Editing a graph | `set_amat` (every arc at once), `set_arc`, `drop_arc`, `reverse_arc`, `set_edge`, `drop_edge`, `add_node`, `remove_node`, `rename_nodes` |
| Constraints from orderings | `ordering2blacklist`, `tiers2blacklist`, `set2blacklist` |
| Parameter learning | `fit` — `mle` and `bayes` for discrete networks, `mle-g` for Gaussian ones, `mle-cg` for mixtures of the two; `custom_fit` to supply parameters by hand; `bn_net` to drop them again |
| Incomplete data | `impute` — from a node's parents, by likelihood weighting, or exactly; `structural_em` to learn a structure despite the gaps, latent variables included |
| Prediction | `predict` — from a node's parents, by likelihood weighting, or exactly |
| Exact inference | `query` — junction tree for discrete networks, multivariate normal for Gaussian ones; conditional and joint distributions, computed rather than sampled |
| Gaussian networks as distributions | `gbn2mvnorm`, `mvnorm2gbn` — the global mean and covariance, and the factorisation back |
| Classifiers | `naive_bayes`, `tree_bayes`, `classify` — exact class posteriors |
| Simulation and inference | `rbn`, `cpquery`, `cpdist` — logic sampling and likelihood weighting; `set_seed` reproduces R's `set.seed` |
| Resampling | `boot_strength`, `bn_boot`, `bn_cv`, `loss` — bootstrap arc strengths, a bootstrap distribution for any statistic, k-fold / hold-out / custom-fold cross-validation, all six losses |
| Arc strength | `arc_strength` (p-value or score difference), `bf_strength` (Bayes factors), `custom_strength`, `averaged_network`, `inclusion_threshold` |
| Interchange formats | `read_bif`, `read_dsc`, `read_net`, `write_bif`, `write_dsc`, `write_net`, `write_dot` |
| Graph priors | `uniform`, `vsp`, `marginal`, `cs` (Castelo & Siebes) — passed to any Bayesian score as `prior=` and `beta=` |
| Utilities | `score`, `modelstring`, `identifiable`, `singular`, `whitelist`, `blacklist`, `ntests` |

139 of bnlearn's 160 exported functions are covered, and nothing of
substance is left.  Fifteen of the remaining twenty-one are deliberately out
of scope -- the nine plotting functions and the six conversions to igraph,
graphNEL, gRain and `lm` -- because they would be rewrites rather than
ports, with nothing to check against.  The other six are R's assignment
forms (`nodes<-`, `arcs<-`, `parents<-`, `children<-`, `modelstring<-`,
`alst<-`), whose effect is available here as ordinary functions.

Three things are deliberately not reproduced, and each is documented where
it lives.  `direct_lingam` gives R's causal *ordering* exactly, but picks
the arcs with a score-based search rather than with glmnet's adaptive lasso,
which is not vendored.  `mi = "gkernel"` raises rather than returning a
number: bnlearn builds its Gram matrix with a matrix product where it means
an elementwise one, leaving a condition number around 1e20, so R's own
answer is floating-point noise and cannot be reproduced by anything.  And
`count_graphs("dags-with-r-arcs", r = 0)` returns one rather than R's *n*:
R's inner loop runs `seq(from = 1, to = 0)`, which counts downwards, and the
extra pass overwrites the count -- unlike the other two this one is
checkable, and the counts for a given number of nodes only add up to the
total with the corrected value.
Exact inference covers discrete and Gaussian networks, but not mixtures of
the two.

One inconsistency *is* reproduced rather than smoothed over, because
smoothing it over would be the divergence: `bdj` accepts `prior` and `beta`
and then ignores them, scoring as if the prior were uniform -- while still
being reported as non-decomposable under `cs` and `marginal`, which moves
the search even though the score stands still.  pybnlearn does the same, and
the fixtures pin down both halves of it.

## Verified against R

`pytest` runs 3854 checks, 3716 of which compare directly against values produced
by R 4.6.1 with bnlearn 5.2.1:

* 318 conditional independence tests across discrete and Gaussian data, each
  comparing the statistic, the degrees of freedom and the p-value;
* 82 hill-climbing runs across 8 data sets, 13 scores, non-default
  hyperparameters, whitelists, blacklists and parent limits, each comparing the
  arc set, the model string and the per-node scores;
* 231 constraint-based runs across `pc_stable`, `gs`, `iamb`, `inter_iamb`,
  `iamb_fdr`, `mmpc` and `si_hiton_pc`, 6 data sets, 7 independence tests,
  several significance levels, constraint sets and both directed and
  undirected output, comparing the arc set including direction;
* 64 hybrid runs across `mmhc` and `rsmax2`, covering every ported
  restrict/maximize pair and arguments passed through to each phase;
* 110 `hpc` and `h2pc` runs across 7 data sets -- including 37-node `alarm`,
  where the two superset stages earn their keep and an off-by-one in them
  would be least visible -- 6 independence tests, conditioning-set limits
  and constraints;
* 63 tabu searches across 8 data sets, 9 scores, tabu list sizes from 1 to 30,
  constraints and parent limits — 13 of which R's tabu resolves differently
  from R's hc, so the tabu-specific paths are actually covered rather than
  incidentally agreeing with hill climbing;
* 55 fitted networks, comparing every cell of every conditional probability
  table and every regression coefficient, over three structures per data set
  and both the maximum likelihood and Bayesian estimators;
* 61 seeded simulation and inference results -- generated data compared
  observation by observation, and conditional probabilities compared exactly
  rather than statistically, since both implementations draw the same numbers
  in the same order from R's Mersenne-Twister;
* 61 bootstrap and cross-validation results, again exact rather than
  statistical, including R's `sample()` itself -- everything here rests on
  drawing the same rows in the same order;
* 48 predictions and prediction-based losses, covering both prediction
  methods and all five predictive cross-validation losses;
* 42 classifier structures and class posteriors, over four data sets and
  several class variables, including the tree root that decides how a TAN's
  feature tree is oriented;
* 40 exact inference results for discrete networks checked against gRain --
  marginals, conditionals on up to three variables, joints, and exact
  prediction;
* 82 Gaussian exact inference results, which unlike the discrete ones are
  genuine parity fixtures: global means and covariances, the factorisation
  back into a network, conditional expectations, and exact prediction of
  every node of every structure -- including a network with a deterministic
  node, whose covariance matrix is singular and which therefore exercises
  the diagonal patching R does rather than failing;
* 70 conditional Gaussian results over two mixed data sets: structure
  learning with all four `-cg` scores under both `hc` and `tabu`, and
  parameter learning across all three of the estimators `mle-cg` dispatches
  to;
* 138 arc-strength and network-averaging results: p-values and score
  differences for six data sets and thirteen criteria, bootstrap strengths
  with the inclusion threshold they carry, and the averaged network at four
  thresholds -- plus thirteen strength vectors of awkward shapes checking
  that R's optimiser is reproduced rather than approximated, since the
  threshold it finds is used as a cutoff;
* 632 checks of the node and arc utilities, enumerated rather than chosen:
  nine graphs -- including one whose undirected part is not chordal and one
  with a real directed cycle -- crossed with every ordered pair of nodes and
  all five arc operations;
* 148 interchange-format results: every node of four networks written by R
  as BIF, DSC and NET and read back here, which is what pins down where each
  conditional distribution belongs -- NET does not record it at all;
* 24 hand-built networks from `custom_fit`, compared as parameters and then
  sampled from and queried, since a network with an axis in the wrong place
  reads back correctly and samples from a different distribution;
* 524 checks of d-separation, colliders, consistent extensions, structural
  intervention distance, random graph generation and discretization --
  d-separation enumerated over every pair of nodes in six graphs and three
  kinds of conditioning set, and the generated graphs compared arc for arc
  rather than statistically, since they come from R's own generator;
* 245 checks of fast.iamb, local structure learning, entropy and
  Kullback-Leibler divergence, the last two comparing pybnlearn's junction
  tree against gRain's;
* 40 incomplete-data results: parameters estimated from data with gaps, all
  three imputation methods, and structural EM under both maximisers -- plus
  a latent variable, never observed at all, which is the case EM cannot
  start from an empty network on;
* 183 causal results: interventions on every node of four graph shapes,
  twin networks, and counterfactuals with and without node merging -- plus
  the parameterised half, where a node's residual variance becomes a node of
  its own feeding both copies;
* 18 LiNGAM causal orderings, including a data set with genuinely
  non-Gaussian noise, which is the assumption the method rests on;
* 139 results for the remaining utilities: seeded perturbations and the
  hill-climbing restarts built on them, every extension of an equivalence
  class, exact graph counts, Bayes-factor arc strengths in extended
  precision, deduplication, and bootstrapping an arbitrary statistic;
* 22 adjacency-matrix results over ten graphs, including two partially
  directed ones -- an undirected arc is a symmetric pair of ones, so the
  matrix cannot tell it from two opposed directed arcs, and the round trip
  is recorded rather than assumed.  The row/column convention and the arc
  order coming back out are compared too: both are invisible in a
  same-shaped answer and decide how everything downstream reads;
* 235 results for the non-uniform graph priors: four Castelo & Siebes
  completions, and nine prior settings crossed with three data sets, five
  scores and both `hc` and `tabu` -- which is what pins down the
  non-decomposable path, since a prior that touches every node makes a
  move's effect wider than its cached delta.  `hc` and `tabu` are expected
  to disagree with each other under the marginal prior, and do, on the same
  pair of networks R's do;
* 27 checks of the graph utilities: CPDAG, moral graph and skeleton for six
  learned networks, `shd`/`hamming`/`compare` over five network pairs,
  `model2network` round trips, and `chow_liu` and `aracne` on six data sets;
* `set.seed(42)` reproduces R's uniform and normal streams to 15 digits.

Regenerate the fixtures (needs R with bnlearn installed):

```bash
Rscript tools/gen_r_fixtures.R
Rscript tools/gen_r_hc_fixtures.R
Rscript tools/gen_r_constraint_fixtures.R
Rscript tools/gen_r_graph_fixtures.R
Rscript tools/gen_r_tabu_fixtures.R
Rscript tools/gen_r_hybrid_fixtures.R
Rscript tools/gen_r_fit_fixtures.R
Rscript tools/gen_r_levels.R
Rscript tools/gen_r_inference_fixtures.R
Rscript tools/gen_r_resampling_fixtures.R
Rscript tools/gen_r_predict_fixtures.R
Rscript tools/gen_r_classifier_fixtures.R
Rscript tools/gen_r_exact_fixtures.R      # also needs gRain
Rscript tools/gen_r_cg_fixtures.R
Rscript tools/gen_r_mvnorm_fixtures.R
Rscript tools/gen_r_strength_fixtures.R
Rscript tools/gen_r_nodes_fixtures.R      # also needs igraph
Rscript tools/gen_r_custom_fixtures.R
Rscript tools/gen_r_foreign_fixtures.R
Rscript tools/gen_r_analysis_fixtures.R
Rscript tools/gen_r_local_fixtures.R     # also needs gRain
Rscript tools/gen_r_hpc_fixtures.R
Rscript tools/gen_r_missing_fixtures.R   # also needs gRain
Rscript tools/gen_r_causal_fixtures.R
Rscript tools/gen_r_lingam_fixtures.R
Rscript tools/gen_r_misc_fixtures.R      # also needs gmp and Rmpfr
Rscript tools/gen_r_priors_fixtures.R
Rscript tools/gen_r_amat_fixtures.R
```

## Performance

Roughly two to three times slower than R on `hc` — 0.61s vs 0.32s for the
37-node `alarm` data set — because the network is marshalled into R objects
afresh on each iteration of the search. Memory is flat across repeated runs.

## Building from source

Needs a C compiler and a BLAS/LAPACK. No Fortran: R's `dqrdc2` and `dqrsl`
are translated to C in `src/c/linpack/`, and `tools/check_linpack.sh` checks
the translation against the Fortran original bit for bit.

```bash
pip install -e . --no-build-isolation
```

To build and exercise the C core without Python in the loop — useful when
bisecting a numerical disagreement with R:

```bash
tools/build_core.sh
```

## Licence

GPL-3.0-or-later. See `LICENSE` and `NOTICE`.
