/* pybnlearn: the Fortran symbol names, with an optional prefix and suffix.
 *
 * A BLAS is usually linked under the plain Fortran names -- dgemm_, ddot_ --
 * and that is what R's F77_NAME does.  The OpenBLAS builds SciPy ships as
 * wheels are different: they are compiled with BLAS_SYMBOL_PREFIX=scipy_ so
 * that a program can link them without clashing with whatever other BLAS is
 * already loaded.  Their headers apply the prefix through macros, but this
 * project declares the handful of routines it needs itself rather than
 * including them, so the renaming has to happen here.
 *
 * That matters on Windows, where there is no system BLAS and the SciPy wheel
 * is the only one available: without this the link fails on dcopy_, ddot_
 * and daxpy_, which is exactly what the first Windows build did.
 *
 * Only the LP64 build is suitable.  scipy-openblas64 is ILP64 -- its integer
 * arguments are 64 bits wide -- while every declaration here, and every call
 * bnlearn makes, passes `const int *`.  Linking against it would compile,
 * link once the names matched, and then read garbage lengths.  Use
 * scipy-openblas32.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#ifndef PYBN_BLAS_NAMES_H
#define PYBN_BLAS_NAMES_H

/* Two levels of indirection: the inner one pastes, the outer one lets the
 * arguments expand first.  Without it BLAS_SYMBOL_PREFIX would be pasted
 * literally rather than as what it stands for. */
#define PYBN_PASTE4(a, b, c, d) a ## b ## c ## d
#define PYBN_JOIN4(a, b, c, d)  PYBN_PASTE4(a, b, c, d)

#ifndef BLAS_SYMBOL_PREFIX
#define BLAS_SYMBOL_PREFIX
#endif
#ifndef BLAS_SYMBOL_SUFFIX
#define BLAS_SYMBOL_SUFFIX
#endif

/* dgemm -> dgemm_, or scipy_dgemm_ when the prefix is set. */
#define PYBN_BLAS(x) PYBN_JOIN4(BLAS_SYMBOL_PREFIX, x, _, BLAS_SYMBOL_SUFFIX)

#endif /* PYBN_BLAS_NAMES_H */
