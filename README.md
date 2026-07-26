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

# constraints
pybnlearn.hc(data, whitelist=[("A", "F")], blacklist=[("A", "B")], maxp=3)
```

### A note on `pandas.read_csv` and categorical data

`read_csv` treats `NA`, `None`, `N/A` and similar strings as missing values by
default. bnlearn's own `insurance` data set has a category literally called
`"None"`, so the default reads 14681 of its rows as missing. Pass
`keep_default_na=False, na_values=[]` when the labels are meant to be data.
pybnlearn raises rather than letting missing values reach the C core, but it
cannot tell that a category was lost before it ever saw the frame.

## What works

| | |
|---|---|
| Conditional independence tests | `mi`, `mi-adf`, `mi-sh`, `x2`, `x2-adf` (discrete); `cor`, `zf`, `mi-g`, `mi-g-sh` (Gaussian) |
| Scores | `loglik`, `aic`, `bic`, `bde`, `bds`, `bdj`, `k2`, `fnml`, `qnml`; `loglik-g`, `aic-g`, `bic-g`, `bge` |
| Structure learning (score-based) | `hc`, `tabu` — whitelists, blacklists, `maxp`, arbitrary starting networks |
| Structure learning (constraint-based) | `pc_stable`, `gs`, `iamb`, `inter_iamb`, `iamb_fdr`, `mmpc`, `si_hiton_pc` — whitelists, blacklists, `alpha`, `max_sx`, `undirected` |
| Structure learning (hybrid) | `mmhc`, `rsmax2` — any ported restrict/maximize pair |
| Structure learning (pairwise) | `chow_liu`, `aracne` |
| Graphs | `cpdag`, `moral`, `skeleton`, `pdag2dag`, `subgraph`, `empty_graph`, `model2network`, topological ordering |
| Comparison | `shd`, `hamming`, `compare`, `nparams` |
| Utilities | `score`, `modelstring` |

Not yet ported: the remaining constraint-based and hybrid algorithms
(`fast.iamb`, `hpc`, and `h2pc`, which needs `hpc`), parameter learning
(`bn.fit`), inference
(`cpquery`, `rbn`), cross-validation, bootstrap, classifiers, conditional
Gaussian networks, incomplete data, non-uniform graph priors, and random
restarts for `hc`.

## Verified against R

`pytest` runs 816 checks, 785 of which compare directly against values produced
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
* 63 tabu searches across 8 data sets, 9 scores, tabu list sizes from 1 to 30,
  constraints and parent limits — 13 of which R's tabu resolves differently
  from R's hc, so the tabu-specific paths are actually covered rather than
  incidentally agreeing with hill climbing;
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
```

## Performance

Roughly two to three times slower than R on `hc` — 0.61s vs 0.32s for the
37-node `alarm` data set — because the network is marshalled into R objects
afresh on each iteration of the search. Memory is flat across repeated runs.

## Building from source

Needs a C compiler, gfortran (until R's LINPACK QR routines are translated to
C), and a BLAS/LAPACK.

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
