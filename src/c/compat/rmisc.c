/* pybnlearn: the remaining odds and ends of the R C API that bnlearn calls.
 *
 * revsort() below is taken from R's src/main/sort.c, and is
 *   Copyright (C) 1995-2026 The R Core Team
 *   Copyright (C) 1998 Ross Ihaka
 * It is reproduced rather than reimplemented because bnlearn uses it to order
 * the probabilities in likelihood weighting, where the handling of ties
 * decides which observations get drawn; anything but R's exact heapsort would
 * silently desynchronise the sampling.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include "rcompat.h"
#include "rcompat_internal.h"

/* -------------------------------------------------------------------------
 * match().
 *
 * R's semantics, which bnlearn depends on throughout: for each element of `x`,
 * the one-based position of its first occurrence in `table`, or `nomatch`
 * (always 0 at bnlearn's call sites) when it does not occur.
 * ------------------------------------------------------------------------- */

static int elements_equal(SEXP a, int i, SEXP b, int j) {

  if (TYPEOF(a) != TYPEOF(b))
    return 0;

  switch (TYPEOF(a)) {

    case STRSXP:
      return strcmp(CHAR(STRING_ELT(a, i)), CHAR(STRING_ELT(b, j))) == 0;

    case INTSXP:
    case LGLSXP:
      return INTEGER(a)[i] == INTEGER(b)[j];

    case REALSXP:
      /* NA and NaN match themselves in R's match(), unlike in ==. */
      if (ISNAN(REAL(a)[i]) && ISNAN(REAL(b)[j]))
        return 1;
      return REAL(a)[i] == REAL(b)[j];

    default:
      Rf_error("match() is not implemented for vectors of type %s.",
        Rf_type2char(TYPEOF(a)));

  }/*SWITCH*/

}/*ELEMENTS_EQUAL*/

SEXP Rf_match(SEXP table, SEXP x, int nomatch) {

int nx = Rf_length(x), ntable = Rf_length(table);
SEXP out = Rf_allocVector(INTSXP, nx);

  for (int i = 0; i < nx; i++) {

    INTEGER(out)[i] = nomatch;

    for (int j = 0; j < ntable; j++) {

      if (elements_equal(table, j, x, i)) {

        INTEGER(out)[i] = j + 1;
        break;

      }/*THEN*/

    }/*FOR*/

  }/*FOR*/

  return out;

}/*RF_MATCH*/

/* -------------------------------------------------------------------------
 * duplicated().
 * ------------------------------------------------------------------------- */

SEXP Rf_duplicated(SEXP x, int from_last) {

int n = Rf_length(x);
SEXP out = Rf_allocVector(LGLSXP, n);

  for (int i = 0; i < n; i++)
    LOGICAL(out)[i] = FALSE;

  if (from_last) {

    for (int i = n - 1; i >= 0; i--)
      for (int j = n - 1; j > i; j--)
        if (elements_equal(x, j, x, i)) {
          LOGICAL(out)[i] = TRUE;
          break;
        }/*THEN*/

  }/*THEN*/
  else {

    for (int i = 0; i < n; i++)
      for (int j = 0; j < i; j++)
        if (elements_equal(x, j, x, i)) {
          LOGICAL(out)[i] = TRUE;
          break;
        }/*THEN*/

  }/*ELSE*/

  return out;

}/*RF_DUPLICATED*/

/* -------------------------------------------------------------------------
 * type2char(), used only to build error messages.
 * ------------------------------------------------------------------------- */

const char *Rf_type2char(SEXPTYPE type) {

  switch (type) {
    case NILSXP:  return "NULL";
    case SYMSXP:  return "symbol";
    case LISTSXP: return "pairlist";
    case CLOSXP:  return "closure";
    case LANGSXP: return "language";
    case CHARSXP: return "char";
    case LGLSXP:  return "logical";
    case INTSXP:  return "integer";
    case REALSXP: return "double";
    case STRSXP:  return "character";
    case VECSXP:  return "list";
    default:      return "unknown";
  }/*SWITCH*/

}/*RF_TYPE2CHAR*/

/* -------------------------------------------------------------------------
 * revsort(), from R's src/main/sort.c.
 * ------------------------------------------------------------------------- */

void revsort(double *a, int *ib, int n) {

/* Sort a[] into descending order by "heapsort";
 * sort ib[] alongside;
 * if initially, ib[] = 1...n, it will contain the permutation finally
 */

int l, j, ir, i;
double ra;
int ii;

    if (n <= 1) return;

    a--; ib--;

    l = (n >> 1) + 1;
    ir = n;

    for (;;) {
	if (l > 1) {
	    l = l - 1;
	    ra = a[l];
	    ii = ib[l];
	}
	else {
	    ra = a[ir];
	    ii = ib[ir];
	    a[ir] = a[1];
	    ib[ir] = ib[1];
	    if (--ir == 1) {
		a[1] = ra;
		ib[1] = ii;
		return;
	    }
	}
	i = l;
	j = l << 1;
	while (j <= ir) {
	    if (j < ir && a[j] > a[j + 1]) ++j;
	    if (ra > a[j]) {
		a[i] = a[j];
		ib[i] = ib[j];
		j += (i = j);
	    }
	    else
		j = ir + 1;
	}
	a[i] = ra;
	ib[i] = ii;
    }

}/*REVSORT*/
