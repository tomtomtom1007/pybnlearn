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
   | PyPI project name | `pybnlearn` |
   | Owner | your GitHub account or org |
   | Repository name | `pybnlearn` |
   | Workflow name | `wheels.yml` |
   | Environment name | `pypi` |

   Create the matching `pypi` environment under the repository's
   Settings → Environments. Restrict it to tags if you want a second gate.

3. **Decide the version.** `pyproject.toml` currently says `0.1.0.dev0`.
   `0.1.0a1` is still the more honest first release: not because much is
   missing -- 139 of bnlearn's 160 exports are ported and checked -- but
   because nobody outside this repository has run it yet, and PyPI treats an
   alpha as a pre-release, so `pip install pybnlearn` will not pick it up
   unless somebody asks for it.

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
| Trusted Publishing | configuration written, never run |

**macOS x86-64.** `macos-13` was the last Intel image GitHub hosted and it no
longer schedules: the job sat for fifty-eight minutes with no runner assigned
while every other job in the same run finished. It is out of the matrix.
Cross-compiling x86-64 on an arm64 runner would link — Accelerate is a
universal framework — but the tests cannot run on the wrong architecture, and
an untested wheel is not what this project ships. Intel Macs build from the
sdist, which is exercised above.

**Trusted Publishing is the one path still unexercised.** It only runs on a
`v*` tag, so nothing so far has touched it. Try TestPyPI first (below): a
version number on PyPI can never be reused, and a misconfigured publisher
fails after the wheels are built rather than before.

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
