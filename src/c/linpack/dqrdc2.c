/* pybnlearn: a C translation of R's src/appl/dqrdc2.f.
 *
 * dqrdc2 is R's modification of LINPACK's dqrdc, with the limited column
 * pivoting that moves near-zero-norm columns to the right and the rank
 * computation that lm() depends on.  LAPACK has nothing that pivots by the
 * same rule, so substituting dgeqp3 would change which columns a rank-deficient
 * fit drops -- hence a translation rather than a replacement.
 *
 * Translating it removes the only Fortran in the build, which is what stands
 * between pybnlearn and Windows wheels.
 *
 * Original (dqrdc.f) LINPACK version dated 08/14/78, G.W. Stewart.
 * Modified for R 22 August 1995 by Ross Ihaka; bug fixes 29 September 1999 by
 * B.D. Ripley; modernised to F77 2024-05 by B.D. Ripley.
 *   Copyright (C) 1995-2026 The R Core Team
 *   Licensed under the GNU General Public License version 2 or later.
 *
 * The one-based X()/WORK() macros are deliberate: they keep this line-for-line
 * comparable with the Fortran, which is the only practical way to audit it.
 * tools/check_linpack.sh compares the two implementations numerically.
 */

#include <math.h>
#include <stddef.h>
#include "../compat/blas_names.h"

#define X(i, j)     x[((j) - 1) * (*ldx) + ((i) - 1)]
#define WORK(j, k)  work[((k) - 1) * (*p) + ((j) - 1)]

extern double PYBN_BLAS(dnrm2)(const int *n, const double *x, const int *incx);
extern double PYBN_BLAS(ddot)(const int *n, const double *x, const int *incx,
    const double *y, const int *incy);
extern void PYBN_BLAS(dscal)(const int *n, const double *a, double *x, const int *incx);
extern void PYBN_BLAS(daxpy)(const int *n, const double *a, const double *x,
    const int *incx, double *y, const int *incy);

/* Fortran's SIGN(a, b): |a| carrying b's sign, with zero counting as
 * positive. */
static double fsign(double a, double b) {

  return (b >= 0.0) ? fabs(a) : -fabs(a);

}/*FSIGN*/

void PYBN_BLAS(dqrdc2)(double *x, const int *ldx, const int *n, const int *p,
    const double *tol, int *k, double *qraux, int *jpvt, double *work) {

int i = 0, j = 0, l = 0, lup = 0;
int one = 1, len = 0;
double tt = 0, ttt = 0, nrmxl = 0, t = 0;

  /* compute the norms of the columns of x. */
  if (*n > 0) {

    for (j = 1; j <= *p; j++) {

      qraux[j - 1] = PYBN_BLAS(dnrm2)(n, &X(1, j), &one);
      WORK(j, 1) = qraux[j - 1];
      WORK(j, 2) = qraux[j - 1];
      if (WORK(j, 2) == 0.0)
        WORK(j, 2) = 1.0;

    }/*FOR*/

  }/*THEN*/

  /* perform the householder reduction of x. */
  lup = (*n < *p) ? *n : *p;
  *k = *p + 1;

  for (l = 1; l <= lup; l++) {

    /* Cycle the columns from l to p left-to-right until one with a
     * non-negligible norm is located; a column has become negligible once its
     * norm has fallen below tol times its original norm.  The l >= k test is
     * what stops this cycling forever. */
    while (!(l >= *k || qraux[l - 1] >= WORK(l, 2) * (*tol))) {

      for (i = 1; i <= *n; i++) {

        t = X(i, l);
        for (j = l + 1; j <= *p; j++)
          X(i, j - 1) = X(i, j);
        X(i, *p) = t;

      }/*FOR*/

      i = jpvt[l - 1];
      t = qraux[l - 1];
      tt = WORK(l, 1);
      ttt = WORK(l, 2);

      for (j = l + 1; j <= *p; j++) {

        jpvt[j - 2] = jpvt[j - 1];
        qraux[j - 2] = qraux[j - 1];
        WORK(j - 1, 1) = WORK(j, 1);
        WORK(j - 1, 2) = WORK(j, 2);

      }/*FOR*/

      jpvt[*p - 1] = i;
      qraux[*p - 1] = t;
      WORK(*p, 1) = tt;
      WORK(*p, 2) = ttt;
      (*k)--;

    }/*WHILE*/

    if (l != *n) {

      /* compute the householder transformation for column l. */
      len = *n - l + 1;
      nrmxl = PYBN_BLAS(dnrm2)(&len, &X(l, l), &one);

      if (nrmxl != 0.0) {

        if (X(l, l) != 0.0)
          nrmxl = fsign(nrmxl, X(l, l));

        t = 1.0 / nrmxl;
        PYBN_BLAS(dscal)(&len, &t, &X(l, l), &one);
        X(l, l) = 1.0 + X(l, l);

        /* apply the transformation to the remaining columns, updating the
         * norms. */
        if (*p >= l + 1) {

          for (j = l + 1; j <= *p; j++) {

            t = -PYBN_BLAS(ddot)(&len, &X(l, l), &one, &X(l, j), &one) / X(l, l);
            PYBN_BLAS(daxpy)(&len, &t, &X(l, l), &one, &X(l, j), &one);

            if (qraux[j - 1] != 0.0) {

              tt = 1.0 - (fabs(X(l, j)) / qraux[j - 1])
                       * (fabs(X(l, j)) / qraux[j - 1]);
              tt = (tt > 0.0) ? tt : 0.0;
              t = tt;

              /* Modified 9/99 by BDR: recompute the norm outright when the
               * reduction is large, since this version needs accurate norms.
               * The tolerance is on the squared norm. */
              if (fabs(t) >= 1e-6) {

                qraux[j - 1] = qraux[j - 1] * sqrt(t);

              }/*THEN*/
              else {

                len = *n - l;
                qraux[j - 1] = PYBN_BLAS(dnrm2)(&len, &X(l + 1, j), &one);
                WORK(j, 1) = qraux[j - 1];
                len = *n - l + 1;

              }/*ELSE*/

            }/*THEN*/

          }/*FOR*/

        }/*THEN*/

        /* save the transformation. */
        qraux[l - 1] = X(l, l);
        X(l, l) = -nrmxl;

      }/*THEN*/

    }/*THEN*/

  }/*FOR*/

  *k = (*k - 1 < *n) ? *k - 1 : *n;

}/*DQRDC2_*/
