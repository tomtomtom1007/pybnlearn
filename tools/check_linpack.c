/* pybnlearn: check the C translations of dqrdc2 and dqrsl against the Fortran
 * they were translated from.
 *
 * Both are compiled into this program, the Fortran under renamed symbols, and
 * run on the same inputs.  The comparison is bit-for-bit: these routines are
 * deterministic and call the same BLAS, so anything short of an exact match
 * means the translation changed the arithmetic.
 *
 * The cases deliberately include rank-deficient matrices -- duplicated
 * columns, zero columns, more columns than rows -- because that is where
 * dqrdc2 differs from stock LINPACK, and where a translation is most likely
 * to go wrong unnoticed.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void dqrdc2_(double *x, const int *ldx, const int *n, const int *p,
    const double *tol, int *k, double *qraux, int *jpvt, double *work);
void dqrsl_(double *x, const int *ldx, const int *n, const int *k,
    double *qraux, double *y, double *qy, double *qty, double *b, double *rsd,
    double *xb, const int *job, int *info);

/* the Fortran originals, renamed by tools/check_linpack.sh */
void fdqrdc2_(double *x, const int *ldx, const int *n, const int *p,
    const double *tol, int *k, double *qraux, int *jpvt, double *work);
void fdqrsl_(double *x, const int *ldx, const int *n, const int *k,
    double *qraux, double *y, double *qy, double *qty, double *b, double *rsd,
    double *xb, const int *job, int *info);

static unsigned long long seed = 88172645463325252ULL;

static double next_double(void) {

  seed ^= seed << 13;
  seed ^= seed >> 7;
  seed ^= seed << 17;

  return (double)((seed >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53);

}/*NEXT_DOUBLE*/

static int failures = 0;

static int same_doubles(const char *what, const double *a, const double *b,
    int n) {

  for (int i = 0; i < n; i++) {

    /* compare the bits: NaN must match NaN, and -0.0 must match -0.0. */
    if (memcmp(&a[i], &b[i], sizeof(double)) != 0) {

      printf("    %s[%d] differs: C %.17g, Fortran %.17g\n", what, i, a[i],
        b[i]);
      return 0;

    }/*THEN*/

  }/*FOR*/

  return 1;

}/*SAME_DOUBLES*/

static int same_ints(const char *what, const int *a, const int *b, int n) {

  for (int i = 0; i < n; i++) {

    if (a[i] != b[i]) {

      printf("    %s[%d] differs: C %d, Fortran %d\n", what, i, a[i], b[i]);
      return 0;

    }/*THEN*/

  }/*FOR*/

  return 1;

}/*SAME_INTS*/

/* `kind` selects how the matrix is degenerate, if at all. */
static void run_case(int n, int p, int kind) {

int ldx = n, i, j, ok = 1;
double tol = 1e-7;
size_t xs = (size_t)ldx * p;

double *xc = calloc(xs, sizeof(double)), *xf = calloc(xs, sizeof(double));
double *qrauxc = calloc(p, sizeof(double)), *qrauxf = calloc(p, sizeof(double));
double *workc = calloc(2 * p, sizeof(double)), *workf = calloc(2 * p, sizeof(double));
int *pivc = calloc(p, sizeof(int)), *pivf = calloc(p, sizeof(int));
double *y = calloc(n, sizeof(double));
double *bc = calloc(p, sizeof(double)), *bf = calloc(p, sizeof(double));
double *rsdc = calloc(n, sizeof(double)), *rsdf = calloc(n, sizeof(double));
double *fttc = calloc(n, sizeof(double)), *fttf = calloc(n, sizeof(double));
int kc = 0, kf = 0, infoc = 0, infof = 0, job = 100 + 10 + 1;

  for (j = 0; j < p; j++)
    for (i = 0; i < n; i++)
      xc[j * ldx + i] = next_double() * 2.0 - 1.0;

  if (kind == 1 && p >= 2)          /* an exactly duplicated column */
    for (i = 0; i < n; i++)
      xc[1 * ldx + i] = xc[0 * ldx + i];
  else if (kind == 2 && p >= 3)     /* a zero column */
    for (i = 0; i < n; i++)
      xc[2 * ldx + i] = 0.0;
  else if (kind == 3 && p >= 3)     /* a linear combination */
    for (i = 0; i < n; i++)
      xc[2 * ldx + i] = 2.0 * xc[0 * ldx + i] - 3.0 * xc[1 * ldx + i];
  else if (kind == 4)               /* an intercept column */
    for (i = 0; i < n; i++)
      xc[0 * ldx + i] = 1.0;

  for (i = 0; i < n; i++)
    y[i] = next_double() * 2.0 - 1.0;

  memcpy(xf, xc, xs * sizeof(double));
  for (j = 0; j < p; j++)
    pivc[j] = pivf[j] = j + 1;

  dqrdc2_(xc, &ldx, &n, &p, &tol, &kc, qrauxc, pivc, workc);
  fdqrdc2_(xf, &ldx, &n, &p, &tol, &kf, qrauxf, pivf, workf);

  ok &= same_doubles("x", xc, xf, (int)xs);
  ok &= same_doubles("qraux", qrauxc, qrauxf, p);
  ok &= same_ints("jpvt", pivc, pivf, p);
  if (kc != kf) {
    printf("    rank differs: C %d, Fortran %d\n", kc, kf);
    ok = 0;
  }

  /* the same aliasing bnlearn uses: qty, xb and the fitted values share a
   * buffer. */
  memcpy(fttc, y, n * sizeof(double));
  memcpy(fttf, y, n * sizeof(double));

  dqrsl_(xc, &ldx, &n, &kc, qrauxc, fttc, NULL, fttc, bc, rsdc, fttc, &job,
    &infoc);
  fdqrsl_(xf, &ldx, &n, &kf, qrauxf, fttf, NULL, fttf, bf, rsdf, fttf, &job,
    &infof);

  ok &= same_doubles("coefficients", bc, bf, p);
  ok &= same_doubles("residuals", rsdc, rsdf, n);
  ok &= same_doubles("fitted", fttc, fttf, n);
  if (infoc != infof) {
    printf("    info differs: C %d, Fortran %d\n", infoc, infof);
    ok = 0;
  }

  printf("  n=%-4d p=%-3d kind=%d rank=%-3d %s\n", n, p, kind, kc,
    ok ? "ok" : "MISMATCH");
  if (!ok)
    failures++;

  free(xc); free(xf); free(qrauxc); free(qrauxf); free(workc); free(workf);
  free(pivc); free(pivf); free(y); free(bc); free(bf);
  free(rsdc); free(rsdf); free(fttc); free(fttf);

}/*RUN_CASE*/

int main(void) {

int sizes[][2] = {{1, 1}, {2, 1}, {5, 3}, {10, 4}, {20, 7}, {50, 10},
                  {100, 25}, {3, 5}, {4, 9}, {200, 3}, {6, 6}, {17, 17}};

  printf("comparing the C translations against the Fortran originals\n");

  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++)
    for (int kind = 0; kind <= 4; kind++)
      run_case(sizes[s][0], sizes[s][1], kind);

  if (failures == 0)
    printf("\nall cases agree bit for bit.\n");
  else
    printf("\n%d case(s) DIFFER.\n", failures);

  return failures != 0;

}/*MAIN*/
