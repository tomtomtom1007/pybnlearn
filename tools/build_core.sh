#!/bin/sh
# pybnlearn: build the C core and run the parity harness.
#
# This is the scaffold used while the compatibility layer was being brought up.
# The real build is meson-python (see pyproject.toml); this script stays because
# it builds the core without Python in the loop, which is what you want when
# bisecting a numerical disagreement with R.
#
# Usage: tools/build_core.sh [outdir]
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.

set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/build/core"}

mkdir -p "$OUT"
cd "$ROOT"

# The vendored bnlearn sources and the compatibility layer see the shim R
# headers; nothing under src/c/bnlearn/ is patched.
BN_INC="-Isrc/c/compat/rapi -Isrc/c/compat"
CFLAGS="-O2 -std=c11 -fPIC -w"

echo "building bnlearn + compat ..."
n=0
for f in $(find src/c/bnlearn -name '*.c') src/c/compat/*.c src/c/rsupport/qsort.c; do
  o="$OUT/$(echo "$f" | tr '/' '_').o"
  cc -c $CFLAGS $BN_INC -o "$o" "$f"
  n=$((n + 1))
done
echo "  $n objects"

# nmath is R's standalone maths library and is compiled against its own
# headers, not the shim: it *is* the reference implementation, so it must not
# see our reimplementation of the API.
echo "building nmath ..."
n=0
for f in src/c/nmath/*.c; do
  cc -c $CFLAGS -DMATHLIB_STANDALONE -Isrc/c/nmath -o "$OUT/nmath_$(basename "$f").o" "$f"
  n=$((n + 1))
done
echo "  $n objects"

# dqrdc2/dqrsl are still Fortran; translating them to C is what removes the
# gfortran dependency from the wheel build.
echo "building linpack ..."
for f in src/c/linpack/*.f; do
  gfortran -c -O2 -fPIC -o "$OUT/$(basename "$f").o" "$f"
done

case "$(uname -s)" in
  Darwin) BLAS="-framework Accelerate"; SO=dylib ;;
  *)      BLAS="-llapack -lblas";       SO=so    ;;
esac

echo "linking ..."
cc -shared -o "$OUT/libpybnlearn.$SO" "$OUT"/*.o $BLAS -lm
echo "  $OUT/libpybnlearn.$SO"

echo "building the parity harness ..."
cc $CFLAGS $BN_INC -o "$OUT/harness_citest" \
  tests/parity/harness_citest.c "$OUT"/*.o $BLAS -lm

if [ -f tests/parity/fixtures/learning.test.csv ]; then
  echo
  echo "conditional independence tests (compare with tools/gen_r_fixtures.R):"
  "$OUT/harness_citest" tests/parity/fixtures/learning.test.csv
else
  echo
  echo "fixtures missing; run tools/gen_r_fixtures.R first."
fi
