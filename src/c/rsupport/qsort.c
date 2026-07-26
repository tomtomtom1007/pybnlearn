/* pybnlearn: vendored from R 4.6.1 src/main/qsort.c, with the four
 * R-internal includes replaced by the compatibility header.  The sort
 * itself is unmodified: bnlearn orders arcs and scores with it, and
 * tie handling has to agree with R for the parity tests to mean
 * anything.
 *   Copyright (C) 1998-2026 The R Core Team
 *   Licensed under the GNU General Public License version 2 or later. */
#include "../compat/rcompat.h"
#define R_SIZE_T size_t
#define attribute_hidden


#ifdef LONG_VECTOR_SUPPORT
static void R_qsort_R(double *v, double *I, size_t i, size_t j);
static void R_qsort_int_R(int *v, double *I, size_t i, size_t j);
#endif

/* pybnlearn: R's .Internal(qsort) entry point was removed here; only the
 * R_qsort_* helpers below are used, and do_qsort pulled in R's argument
 * matching and long-vector machinery. */


/* These are exposed in Utils.h and are misguidedly in the API */
void F77_SUB(qsort4)(double *v, int *indx, int *ii, int *jj)
{
    R_qsort_I(v, indx, *ii, *jj);
}

void F77_SUB(qsort3)(double *v, int *ii, int *jj)
{
    R_qsort(v, *ii, *jj);
}

//  sort with index : --------------------------
#define qsort_Index
#define INTt int
#define INDt int

#define NUMERIC double
void R_qsort_I(double *v, int *I, int i, int j)
#include "qsort-body.c"
#undef NUMERIC

#define NUMERIC int
void R_qsort_int_I(int *v, int *I, int i, int j)
#include "qsort-body.c"
#undef NUMERIC

#undef INTt
#undef INDt

#ifdef LONG_VECTOR_SUPPORT
#define INDt double
#define NUMERIC double
static void R_qsort_R(double *v, double *I, size_t i, size_t j)
#include "qsort-body.c"
#undef NUMERIC

#define NUMERIC int
static void R_qsort_int_R(int *v, double *I, size_t i, size_t j)
#include "qsort-body.c"
#undef NUMERIC
#undef INDt
#endif // LONG_VECTOR_SUPPORT

//  sort withOUT index : -----------------------
#undef qsort_Index

#define NUMERIC double
void R_qsort(double *v, size_t i, size_t j)
#include "qsort-body.c"
#undef NUMERIC

#define NUMERIC int
void R_qsort_int(int *v, size_t i, size_t j)
#include "qsort-body.c"
#undef NUMERIC
