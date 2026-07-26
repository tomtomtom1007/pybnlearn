# pybnlearn

A Python port of [bnlearn](https://www.bnlearn.com/), Marco Scutari's R package
for Bayesian network structure learning, parameter learning and inference.

> **Status: pre-alpha.** The C core builds and matches R exactly on the tests
> implemented so far, but most of the API is not wired up yet. Not usable for
> real work.

## What makes this a port rather than a reimplementation

The numerical core *is* bnlearn's C code. The sources under `src/c/bnlearn/`
are byte-identical to the CRAN tarball; what makes them build without R is a
compatibility layer (`src/c/compat/`) that reimplements the subset of the R C
API bnlearn uses — the SEXP object model, attributes, `allocVector`, `error()`
and about sixty other entry points.

R's own standalone maths library is vendored too (`src/c/nmath/`), along with
R's Mersenne-Twister, so distribution functions and random streams agree with R
bit for bit. That is what makes it meaningful to assert that results *match*
rather than *approximate*.

Because bnlearn is GPL-2 | GPL-3, pybnlearn is distributed under the GPL v3 or
later. See `NOTICE` for attribution.

This project is unrelated to the PyPI package called `bnlearn`, which is a
separate MIT-licensed project built on pgmpy.

## Verified so far

Conditional independence tests on `learning.test`, against R's `ci.test()`:

| test | statistic | df | p-value |
|------|-----------|----|---------|
| `A ⟂ B`, mi | 2341.8278891943 | 4 | 0 |
| `A ⟂ B`, x2 | 2208.1970543590 | 4 | 0 |
| `A ⟂ B \| C`, mi | 2347.2585878943 | 12 | 0 |
| `A ⟂ F`, mi | 1.3008239105 | 2 | 0.5218307616 |
| `A ⟂ F \| B`, mi | 8.0618628872 | 6 | 0.2336060284 |

All agree with R to every digit printed.

## Building

Requires a C compiler, gfortran (until R's LINPACK QR routines are translated
to C), and a BLAS/LAPACK.

```bash
pip install -e .
```

To build and exercise the C core without Python in the loop — useful when
bisecting a numerical disagreement with R:

```bash
tools/build_core.sh
```

## Licence

GPL-3.0-or-later. See `LICENSE` and `NOTICE`.
