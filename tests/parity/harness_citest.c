/* pybnlearn: a standalone check that the compatibility layer reproduces R.
 *
 * This exercises the whole stack below Python -- the SEXP model, attributes,
 * factor levels, data-frame column lookup and the discrete test kernels -- by
 * running conditional independence tests on bnlearn's own learning.test data
 * and printing the statistics for comparison against R's ci.test().
 *
 * Compiling and linking prove nothing about semantics; this is what actually
 * shows the shim behaves like R.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include <setjmp.h>
#include "../../src/c/compat/rcompat.h"
#include "../../src/c/compat/rcompat_internal.h"

extern SEXP onLoad(void);
extern SEXP indep_test(SEXP x, SEXP y, SEXP sx, SEXP data, SEXP test,
    SEXP alpha, SEXP extra_args, SEXP learning, SEXP complete);

#define MAX_OBS   10000
#define MAX_COLS  16
#define MAX_LVLS  32

static char colname[MAX_COLS][64];
static char levname[MAX_COLS][MAX_LVLS][64];
static int  nlev[MAX_COLS];
static int  codes[MAX_COLS][MAX_OBS];
static int  n_cols = 0, n_obs = 0;

/* map a level label to its one-based code, registering it on first sight.
 * R's read.csv would sort the labels; learning.test's are "a","b","c" in
 * order of appearance anyway, and the tests are invariant to the labelling. */
static int level_code(int col, const char *s) {

  for (int i = 0; i < nlev[col]; i++)
    if (strcmp(levname[col][i], s) == 0)
      return i + 1;

  strcpy(levname[col][nlev[col]], s);
  nlev[col]++;

  return nlev[col];

}/*LEVEL_CODE*/

static void read_csv(const char *path) {

char line[4096];
FILE *f = fopen(path, "r");

  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(1);
  }/*THEN*/

  /* header */
  if (fgets(line, sizeof(line), f)) {

    char *p = strtok(line, ",\r\n");

    while (p) {
      strcpy(colname[n_cols++], p);
      p = strtok(NULL, ",\r\n");
    }/*WHILE*/

  }/*THEN*/

  while (fgets(line, sizeof(line), f) && n_obs < MAX_OBS) {

    char *p = strtok(line, ",\r\n");

    for (int c = 0; p && c < n_cols; c++) {
      codes[c][n_obs] = level_code(c, p);
      p = strtok(NULL, ",\r\n");
    }/*FOR*/

    n_obs++;

  }/*WHILE*/

  fclose(f);

}/*READ_CSV*/

/* build the data.frame of factors that bnlearn expects. */
static SEXP build_dataframe(void) {

SEXP df = Rf_allocVector(VECSXP, n_cols);
SEXP names = Rf_allocVector(STRSXP, n_cols);
SEXP klass = Rf_mkString("data.frame");
SEXP rownames = Rf_allocVector(INTSXP, n_obs);

  for (int c = 0; c < n_cols; c++) {

    SEXP col = Rf_allocVector(INTSXP, n_obs);
    SEXP levels = Rf_allocVector(STRSXP, nlev[c]);

    for (int i = 0; i < n_obs; i++)
      INTEGER(col)[i] = codes[c][i];

    for (int l = 0; l < nlev[c]; l++)
      SET_STRING_ELT(levels, l, Rf_mkChar(levname[c][l]));

    Rf_setAttrib(col, R_LevelsSymbol, levels);
    Rf_setAttrib(col, R_ClassSymbol, Rf_mkString("factor"));

    SET_VECTOR_ELT(df, c, col);
    SET_STRING_ELT(names, c, Rf_mkChar(colname[c]));

  }/*FOR*/

  for (int i = 0; i < n_obs; i++)
    INTEGER(rownames)[i] = i + 1;

  Rf_setAttrib(df, R_NamesSymbol, names);
  Rf_setAttrib(df, R_ClassSymbol, klass);
  Rf_setAttrib(df, R_RowNamesSymbol, rownames);

  return df;

}/*BUILD_DATAFRAME*/

static SEXP build_complete(void) {

SEXP cc = Rf_allocVector(LGLSXP, n_cols);
SEXP names = Rf_allocVector(STRSXP, n_cols);

  for (int c = 0; c < n_cols; c++) {
    LOGICAL(cc)[c] = TRUE;
    SET_STRING_ELT(names, c, Rf_mkChar(colname[c]));
  }/*FOR*/

  Rf_setAttrib(cc, R_NamesSymbol, names);

  return cc;

}/*BUILD_COMPLETE*/

static SEXP get_element(SEXP list, const char *want) {

SEXP names = Rf_getAttrib(list, R_NamesSymbol);

  for (int i = 0; i < Rf_length(list); i++)
    if (strcmp(CHAR(STRING_ELT(names, i)), want) == 0)
      return VECTOR_ELT(list, i);

  return R_NilValue;

}/*GET_ELEMENT*/

static void run(SEXP df, SEXP cc, const char *xn, const char *yn,
    const char *zn, const char *testname) {

SEXP x = Rf_mkString(xn);
SEXP y = Rf_mkString(yn);
SEXP sx = zn ? Rf_mkString(zn) : R_NilValue;
SEXP test = Rf_mkString(testname);
SEXP alpha = Rf_ScalarReal(1.0);
SEXP extra = Rf_allocVector(VECSXP, 0);
SEXP learning = Rf_ScalarLogical(FALSE);
SEXP res = NULL, stat = NULL, df_ = NULL, pv = NULL;

  res = indep_test(x, y, sx, df, test, alpha, extra, learning, cc);

  stat = get_element(res, "statistic");
  df_  = get_element(res, "parameter");
  pv   = get_element(res, "p.value");

  printf("%-6s %-7s stat=%.10f df=%.0f p=%.10f\n",
    zn ? "cond" : "uncond", testname,
    Rf_isNull(stat) ? -1.0 : REAL(stat)[0],
    Rf_isNull(df_)  ? -1.0 : REAL(df_)[0],
    Rf_isNull(pv)   ? -1.0 : REAL(pv)[0]);

}/*RUN*/

int main(int argc, char **argv) {

  if (argc < 2) {
    fprintf(stderr, "usage: %s learning.test.csv\n", argv[0]);
    return 1;
  }/*THEN*/

  read_csv(argv[1]);
  printf("read %d observations, %d columns, levels:", n_obs, n_cols);
  for (int c = 0; c < n_cols; c++)
    printf(" %s=%d", colname[c], nlev[c]);
  printf("\n\n");

  pybn_init_constants();
  pybn_arena_push();

  if (setjmp(*pybn_error_target()) != 0) {
    fprintf(stderr, "error from the C core: %s\n", pybn_error_message());
    return 1;
  }/*THEN*/

  onLoad();

  {
    SEXP df = build_dataframe();
    SEXP cc = build_complete();

    run(df, cc, "A", "B", NULL, "mi");
    run(df, cc, "A", "B", NULL, "x2");
    run(df, cc, "A", "B", NULL, "mi-adf");
    run(df, cc, "A", "B", NULL, "x2-adf");
    run(df, cc, "A", "B", "C",  "mi");
    run(df, cc, "A", "B", "C",  "x2");

    /* pairs whose p-values are not saturated at zero, so that pchisq() is
     * actually exercised rather than just the test statistics. */
    run(df, cc, "A", "F", NULL, "mi");
    run(df, cc, "B", "F", NULL, "mi");
    run(df, cc, "C", "E", NULL, "mi");
    run(df, cc, "A", "F", "B",  "mi");
    run(df, cc, "D", "E", "A",  "mi");
  }

  pybn_arena_pop();

  return 0;

}/*MAIN*/
