#!/bin/sh
# pybnlearn: verify the C translations of dqrdc2/dqrsl against the Fortran.
#
# Needs gfortran, which the build itself no longer does -- this is a
# development check, not part of building or installing the package.
#
# Copyright (C) 2026 the pybnlearn authors.
# Licensed under the GNU General Public License version 3 or later.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/build/linpack-check"}
mkdir -p "$OUT"
cd "$ROOT"

if ! command -v gfortran >/dev/null 2>&1; then
  echo "gfortran is needed to run this check; skipping." >&2
  exit 0
fi

# Rename the Fortran entry points so both implementations can be linked into
# one program and run side by side.
sed -e 's/dqrdc2/fdqrdc2/g' vendor/R-4.6.1/src/appl/dqrdc2.f > "$OUT/fdqrdc2.f"
sed -e 's/dqrsl/fdqrsl/g'   vendor/R-4.6.1/src/appl/dqrsl.f  > "$OUT/fdqrsl.f"

gfortran -c -O2 -o "$OUT/fdqrdc2.o" "$OUT/fdqrdc2.f"
gfortran -c -O2 -o "$OUT/fdqrsl.o"  "$OUT/fdqrsl.f"

cc -c -O2 -std=c11 -o "$OUT/dqrdc2.o" src/c/linpack/dqrdc2.c
cc -c -O2 -std=c11 -o "$OUT/dqrsl.o"  src/c/linpack/dqrsl.c

case "$(uname -s)" in
  Darwin) BLAS="-framework Accelerate" ;;
  *)      BLAS="-llapack -lblas" ;;
esac

cc -O2 -std=c11 -o "$OUT/check" tools/check_linpack.c "$OUT"/*.o $BLAS -lm \
   -lgfortran -L"$(dirname "$(gfortran -print-file-name=libgfortran.dylib 2>/dev/null || \
                               gfortran -print-file-name=libgfortran.so)")"

"$OUT/check"
