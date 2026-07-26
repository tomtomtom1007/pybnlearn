/* pybnlearn: a hand-written stand-in for the Rconfig.h that R's configure
 * script generates.
 *
 * The vendored nmath sources include this unconditionally.  Shipping R's own
 * generated copy would bake in whatever machine built it, which is no good for
 * wheels: the same source tree has to compile on Linux, macOS and Windows, for
 * x86-64 and arm64.  Everything below is therefore derived from macros the
 * compiler itself defines, not from a configure probe.
 *
 * Only the handful of settings nmath actually consults are defined.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#ifndef PYBNLEARN_RCONFIG_H
#define PYBNLEARN_RCONFIG_H

#include <float.h>

/* nmath assumes IEEE 754 doubles throughout: NaN payloads, signed zeros and
 * infinities all carry meaning in the distribution functions.  Every platform
 * pybnlearn targets provides them. */
#define IEEE_754 1

/* byte order, used when picking apart the NA payload. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
# if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#elif defined(__BIG_ENDIAN__) || defined(_BIG_ENDIAN)
# define WORDS_BIGENDIAN 1
#endif

/* nmath uses long double for intermediate accumulation where it is wider than
 * double.  On MSVC and on arm64 Windows long double is just double, so the
 * extra precision paths must be switched off there. */
#if LDBL_MANT_DIG > DBL_MANT_DIG
# define HAVE_LONG_DOUBLE 1
#endif

/* the Fortran symbol convention.  pybnlearn resolves BLAS/LAPACK against the
 * libraries SciPy already ships, which use the trailing-underscore names on
 * every platform SciPy builds for. */
#define HAVE_F77_UNDERSCORE 1

#define R_INLINE inline

#ifdef __SIZEOF_SIZE_T__
# define SIZEOF_SIZE_T __SIZEOF_SIZE_T__
#else
# define SIZEOF_SIZE_T 8
#endif

/* enum base types are C23; nmath only uses this to size Rboolean, and the
 * default enum is fine for that. */
/* #undef HAVE_ENUM_BASE_TYPE */

/* pybnlearn never builds the parts of R that need these, but the headers
 * reference them. */
#define HAVE_VISIBILITY_ATTRIBUTE 1
#define SUPPORT_UTF8 1

#endif
