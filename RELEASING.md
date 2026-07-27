# Releasing pybnlearn

## Before the first release

1. ~~Push the repository to GitHub.~~ Done: it is at
   <https://github.com/tomtomtom1007/pybnlearn>, and `pyproject.toml` points
   there.

2. **Register the project on PyPI with Trusted Publishing**, so that no API
   token has to be stored anywhere. On
   <https://pypi.org/manage/account/publishing/>, add a *pending* publisher:

   | field | value |
   |---|---|
   | PyPI project name | `bnlearn-port` |
   | Owner | your GitHub account or org |
   | Repository name | `pybnlearn` |
   | Workflow name | `wheels.yml` |
   | Environment name | `pypi` |

   Create the matching `pypi` environment under the repository's
   Settings → Environments. Restrict it to tags if you want a second gate.

3. ~~Decide the version.~~ Done: `0.1.0a1`, in `pyproject.toml` and
   `src/pybnlearn/__init__.py`.  An alpha not because much is missing -- 139
   of bnlearn's 160 exports are ported and checked -- but because nobody
   outside this repository has run it.

   **A pre-release is only opt-in once a stable release exists.**  pip
   prefers stable versions, but falls back to pre-releases when a project
   has nothing else, so `pip install bnlearn-port` installs the alpha today
   -- verified, not assumed.  Cutting `0.1.0` is what makes `--pre` start
   meaning something.

   **The distribution is `bnlearn-port`; the import is still `pybnlearn`.**
   PyPI refuses names too close to an existing project and there is already a
   `bnlearn` there.  Only `[project] name` changed; `meson.build`, the source
   tree and every example are untouched.

## Cutting a release

```bash
# 1. update the version in pyproject.toml and src/pybnlearn/__init__.py
# 2. make sure the parity suite passes against a current R
Rscript tools/gen_r_fixtures.R
Rscript tools/gen_r_hc_fixtures.R
pytest tests -q

# 3. tag and push
git tag v0.1.0a1
git push origin v0.1.0a1
```

The `wheels` workflow builds Linux (x86-64 and aarch64, manylinux and
musllinux), macOS arm64 and Windows x86-64 wheels plus an sdist, runs the
whole parity suite inside each one, and — only for tags starting with `v` —
uploads everything to PyPI. A full matrix is thirty wheels and takes about
fifty minutes; Linux is the long pole, since it builds twice as many as the
others.

## Rehearsing on TestPyPI

Worth doing before the first release, and after any change to the release
machinery, because Trusted Publishing is the one step a tag cannot test
twice: it either authenticates or it does not, and the version number spent
finding out cannot be spent again.

The `publish-testpypi` job exists for this and runs on **workflow_dispatch**
only -- nothing to edit and nothing to revert, so the rehearsal exercises the
same workflow the release will.

Once, first:

1. Create a TestPyPI account at <https://test.pypi.org/account/register/> --
   it is a separate site from PyPI and shares no accounts.
2. Add a pending publisher at
   <https://test.pypi.org/manage/account/publishing/> with the same fields as
   PyPI except the environment:

   | field | value |
   |---|---|
   | PyPI project name | `bnlearn-port` |
   | Owner | `tomtomtom1007` |
   | Repository name | `pybnlearn` |
   | Workflow name | `wheels.yml` |
   | Environment name | `testpypi` |

3. Create a `testpypi` environment under the repository's
   Settings -> Environments, alongside `pypi`.

Then, any time: the repository's Actions tab -> `wheels` -> **Run workflow**.
It builds the full matrix and uploads to TestPyPI. `skip-existing` is set, so
repeating a rehearsal on the same version is free.

To check what arrived, install it in a clean environment:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ bnlearn-port
```

The extra index is needed because NumPy and pandas are not on TestPyPI.

## Building locally

```bash
pip install cibuildwheel
python -m cibuildwheel --only cp312-macosx_arm64   # or any identifier
```

`python -m cibuildwheel --print-build-identifiers` lists what the current
configuration would build.

## What has actually been verified

Every wheel below was built in CI and ran the whole parity suite — 8849
tests — before being kept. Thirty wheels, thirty suite runs.

| | |
|---|---|
| **Linux x86-64** | 10 wheels: manylinux_2_28 and musllinux, CPython 3.10–3.14 |
| **Linux aarch64** | 10 wheels, same spread, on a native ARM runner |
| **Windows x86-64** | 5 wheels, CPython 3.10–3.14, OpenBLAS vendored in by delvewheel |
| **macOS arm64** | 5 wheels, CPython 3.10–3.14, linked against Accelerate |
| **sdist** | builds on a clean Ubuntu with only `libopenblas-dev`, and the suite passes from it |
| macOS arm64 via `python -m build` | also built locally, installed into a clean venv, suite passes |
| `twine check` | passes for both artefacts |
| C translation of dqrdc2/dqrsl | agrees with the Fortran bit for bit over 60 cases including rank-deficient ones (`tools/check_linpack.sh`) |
| **macOS x86-64** | **not built, and not buildable here** — see below |
| Trusted Publishing | **exercised on TestPyPI**: 31 files uploaded by the OIDC exchange, installed from there into a clean virtualenv, and `hc` reproduced R's model string |

**macOS x86-64.** `macos-13` was the last Intel image GitHub hosted and it no
longer schedules: the job sat for fifty-eight minutes with no runner assigned
while every other job in the same run finished. It is out of the matrix.
Cross-compiling x86-64 on an arm64 runner would link — Accelerate is a
universal framework — but the tests cannot run on the wrong architecture, and
an untested wheel is not what this project ships. Intel Macs build from the
sdist, which is exercised above.

**Trusted Publishing has been rehearsed.** The `publish-testpypi` job ran the
same OIDC exchange against TestPyPI and uploaded all 31 artefacts -- one
sdist, five macOS arm64, ten manylinux, ten musllinux, five Windows --
which then installed from TestPyPI into a clean virtualenv and reproduced
R's answer on learning.test. What remains untested is only that the *PyPI*
registration is correct, which differs from the TestPyPI one by a single
field (`pypi` rather than `testpypi` as the environment).

## Known packaging limitations

* No x86-64 macOS wheels; see above. Intel Macs install from the sdist.
* Windows has no system BLAS, so the build takes `scipy-openblas32` — the
  LP64 build, not `-64`, because that one is ILP64 and its integer arguments
  are eight bytes wide where every declaration here passes `const int *`.
  Its symbols carry a `scipy_` prefix, which `src/c/compat/blas_names.h`
  applies; without that the link fails on `dcopy_`, `ddot_` and `daxpy_`.
* Installing from the sdist needs a C compiler and a BLAS/LAPACK. There is no
  Fortran anywhere any more — R's `dqrdc2` and `dqrsl` are translated to C in
  `src/c/linpack/` — so gfortran is not required.
