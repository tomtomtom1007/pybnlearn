/* pybnlearn: a C translation of R's src/appl/dqrsl.f.
 *
 * dqrsl applies the QR factorisation dqrdc2 produced: it forms Qy, Q'y, the
 * least-squares coefficients, the residuals and the fitted values, selected by
 * the digits of `job`.
 *
 * LINPACK version dated 08/14/78, G.W. Stewart, University of Maryland,
 * Argonne National Lab.
 *   Copyright (C) 1995-2026 The R Core Team
 *   Licensed under the GNU General Public License version 2 or later.
 *
 * As with dqrdc2.c the one-based X() macro is deliberate, so this stays
 * line-for-line comparable with the Fortran.
 *
 * Note that bnlearn calls this with qty, xb and the fitted-value buffer all
 * pointing at the same array.  Fortran would call that aliasing undefined; the
 * translation keeps the original statement order so that it does whatever the
 * Fortran did, and tools/check_linpack.sh checks the two against each other on
 * exactly the call shape bnlearn uses.
 */

#include <math.h>
#include <stddef.h>

#define X(i, j)  x[((j) - 1) * (*ldx) + ((i) - 1)]

extern double ddot_(const int *n, const double *x, const int *incx,
    const double *y, const int *incy);
extern void daxpy_(const int *n, const double *a, const double *x,
    const int *incx, double *y, const int *incy);
extern void dcopy_(const int *n, const double *x, const int *incx, double *y,
    const int *incy);

void dqrsl_(double *x, const int *ldx, const int *n, const int *k,
    double *qraux, double *y, double *qy, double *qty, double *b, double *rsd,
    double *xb, const int *job, int *info) {

int i = 0, j = 0, jj = 0, ju = 0, kp1 = 0;
int one = 1, len = 0;
int cqy = 0, cqty = 0, cb = 0, cr = 0, cxb = 0;
double t = 0, temp = 0;

  *info = 0;

  /* determine what is to be computed. */
  cqy = (*job / 10000) != 0;
  cqty = (*job % 10000) != 0;
  cb = ((*job % 1000) / 100) != 0;
  cr = ((*job % 100) / 10) != 0;
  cxb = (*job % 10) != 0;
  ju = (*k < *n - 1) ? *k : *n - 1;

  /* special action when n = 1. */
  if (ju == 0) {

    if (cqy)
      qy[0] = y[0];
    if (cqty)
      qty[0] = y[0];
    if (cxb)
      xb[0] = y[0];

    if (cb) {

      if (X(1, 1) == 0.0)
        *info = 1;
      else
        b[0] = y[0] / X(1, 1);

    }/*THEN*/

    if (cr)
      rsd[0] = 0.0;

    return;

  }/*THEN*/

  /* set up to compute qy or qty. */
  if (cqy)
    dcopy_(n, y, &one, qy, &one);
  if (cqty)
    dcopy_(n, y, &one, qty, &one);

  /* compute qy. */
  if (cqy) {

    for (jj = 1; jj <= ju; jj++) {

      j = ju - jj + 1;

      if (qraux[j - 1] != 0.0) {

        temp = X(j, j);
        X(j, j) = qraux[j - 1];
        len = *n - j + 1;
        t = -ddot_(&len, &X(j, j), &one, &qy[j - 1], &one) / X(j, j);
        daxpy_(&len, &t, &X(j, j), &one, &qy[j - 1], &one);
        X(j, j) = temp;

      }/*THEN*/

    }/*FOR*/

  }/*THEN*/

  /* compute trans(q)*y. */
  if (cqty) {

    for (j = 1; j <= ju; j++) {

      if (qraux[j - 1] != 0.0) {

        temp = X(j, j);
        X(j, j) = qraux[j - 1];
        len = *n - j + 1;
        t = -ddot_(&len, &X(j, j), &one, &qty[j - 1], &one) / X(j, j);
        daxpy_(&len, &t, &X(j, j), &one, &qty[j - 1], &one);
        X(j, j) = temp;

      }/*THEN*/

    }/*FOR*/

  }/*THEN*/

  /* set up to compute b, rsd or xb. */
  if (cb)
    dcopy_(k, qty, &one, b, &one);

  kp1 = *k + 1;

  if (cxb)
    dcopy_(k, qty, &one, xb, &one);

  if (cr && *k < *n) {

    len = *n - *k;
    dcopy_(&len, &qty[kp1 - 1], &one, &rsd[kp1 - 1], &one);

  }/*THEN*/

  if (cxb && kp1 <= *n) {

    for (i = kp1; i <= *n; i++)
      xb[i - 1] = 0.0;

  }/*THEN*/

  if (cr) {

    for (i = 1; i <= *k; i++)
      rsd[i - 1] = 0.0;

  }/*THEN*/

  /* compute b. */
  if (cb) {

    for (jj = 1; jj <= *k; jj++) {

      j = *k - jj + 1;

      if (X(j, j) == 0.0) {

        *info = j;
        break;

      }/*THEN*/

      b[j - 1] = b[j - 1] / X(j, j);

      if (j != 1) {

        t = -b[j - 1];
        len = j - 1;
        daxpy_(&len, &t, &X(1, j), &one, b, &one);

      }/*THEN*/

    }/*FOR*/

  }/*THEN*/

  /* compute rsd or xb as required. */
  if (cr || cxb) {

    for (jj = 1; jj <= ju; jj++) {

      j = ju - jj + 1;

      if (qraux[j - 1] != 0.0) {

        temp = X(j, j);
        X(j, j) = qraux[j - 1];
        len = *n - j + 1;

        if (cr) {

          t = -ddot_(&len, &X(j, j), &one, &rsd[j - 1], &one) / X(j, j);
          daxpy_(&len, &t, &X(j, j), &one, &rsd[j - 1], &one);

        }/*THEN*/

        if (cxb) {

          t = -ddot_(&len, &X(j, j), &one, &xb[j - 1], &one) / X(j, j);
          daxpy_(&len, &t, &X(j, j), &one, &xb[j - 1], &one);

        }/*THEN*/

        X(j, j) = temp;

      }/*THEN*/

    }/*FOR*/

  }/*THEN*/

}/*DQRSL_*/
