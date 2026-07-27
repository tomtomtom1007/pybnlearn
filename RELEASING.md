# Releasing pybnlearn

## Before the first release

1. **Push the repository to GitHub.** The URLs in `pyproject.toml` point at
   `github.com/tomtomtom1007/pybnlearn`; change them if the repository lands
   somewhere else.

2. **Register the project on PyPI with Trusted Publishing**, so that no API
   token has to be stored anywhere. On
   <https://pypi.org/manage/account/publishing/>, add a *pending* publisher:

   | field | value |
   |---|---|
   | PyPI project name | `pybnlearn` |
   | Owner | your GitHub account or org |
   | Repository name | `pybnlearn` |
   | Workflow name | `wheels.yml` |
   | Environment name | `pypi` |

   Create the matching `pypi` environment under the repository's
   Settings → Environments. Restrict it to tags if you want a second gate.

3. **Decide the version.** `pyproject.toml` currently says `0.1.0.dev0`.
   Given how much of bnlearn is not ported yet, `0.1.0a1` is a more honest
   first release than `0.1.0`: PyPI treats it as a pre-release, so `pip
   install pybnlearn` will not pick it up unless the user asks for it.

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

The `wheels` workflow builds Linux (x86-64, aarch64) and macOS (x86-64,
arm64) wheels plus an sdist, runs the test suite inside each wheel, and — only
for tags starting with `v` — uploads everything to PyPI.

## Trying it against TestPyPI first

Worth doing for the first release, because a version number on PyPI can never
be reused. Add a step before the real publish:

```yaml
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
```

then install from there in a clean environment:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ pybnlearn
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

Be aware of what is and is not tested, before tagging a release:

| | |
|---|---|
| macOS arm64 wheel via `python -m build` | built, installed into a clean venv, the whole suite passes |
| sdist contents | the vendored bnlearn, nmath and LINPACK sources, LICENSE and NOTICE present |
| `twine check` | passes for both artefacts |
| cibuildwheel configuration | parses; `before-all` runs; the build itself stops locally because cibuildwheel will not install python.org CPython outside CI |
| C translation of dqrdc2/dqrsl | agrees with the Fortran bit for bit over 60 cases including rank-deficient ones (`tools/check_linpack.sh`) |
| **Linux wheels** | **never built** — no container runtime was available on the development machine, so `manylinux_2_28` plus the `dnf install openblas-devel` step is unexercised |
| **macOS x86-64 wheels** | **never built** — no Intel machine |
| **Windows wheels** | **never built** — no Windows machine; the `scipy-openblas64` and `delvewheel` steps are unexercised |
| Trusted Publishing | configuration written, never run |

The first CI run is therefore the real test of the Linux path. Push a tag to a
scratch repository, or run the workflow with `workflow_dispatch`, before
tagging anything you intend to publish.

## Known packaging limitations

* Windows wheels are configured but have never been built. There is no
  Fortran anywhere any more — R's `dqrdc2` and `dqrsl` are translated to C in
  `src/c/linpack/` — but Windows has no system BLAS, so the build pulls in the
  one SciPy ships as a wheel. Expect that step to need iteration on the first
  CI run.
* Installing from the sdist needs a C compiler and a BLAS/LAPACK. gfortran is
  no longer required.
