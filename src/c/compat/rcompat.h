/* pybnlearn: the R C API, reimplemented so that bnlearn builds without R.
 *
 * The vendored bnlearn sources under src/c/bnlearn/ are byte-identical to
 * upstream and keep including <R.h>, <Rinternals.h>, <Rmath.h> and the
 * R_ext headers
 * exactly as they do on CRAN.  Those names are satisfied by the forwarding
 * headers in compat/rapi/, which all resolve to this file; nothing in the
 * vendored tree is patched, so tracking a new bnlearn release is a matter of
 * dropping in the new tarball.
 *
 * What follows is the (surprisingly small) subset of the R C API that bnlearn
 * actually uses:
 *
 *   - the SEXP object model (NULL/logical/integer/real/string/list/pairlist),
 *     including the attributes bnlearn relies on: names, class, levels, dim,
 *     dimnames and row.names;
 *   - allocVector/allocMatrix/allocLang and the PROTECT stack;
 *   - error/warning/Rprintf;
 *   - the R_qsort family from R_ext/Utils.h;
 *   - the handful of Rmath entry points bnlearn calls (see nmath/).
 *
 * Identifiers are remapped exactly the way R's Rinternals.h does it
 * (Rf_length -> length, and so on).  bnlearn declares function *parameters*
 * called `length`, `nrows` and `ncols`; that is legal under R's remapping and
 * stays legal here, which is why the remapping is reproduced faithfully rather
 * than replaced with macros.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#ifndef PYBNLEARN_RCOMPAT_H
#define PYBNLEARN_RCOMPAT_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdarg.h>

/* -------------------------------------------------------------------------
 * the object model.
 * ------------------------------------------------------------------------- */

/* type tags: the numeric values match R's so that any stray comparison against
 * a literal keeps working. */
typedef enum {
  NILSXP  =  0,
  SYMSXP  =  1,
  LISTSXP =  2,
  CLOSXP  =  3,   /* wraps a Python callable. */
  LANGSXP =  6,
  CHARSXP =  9,
  LGLSXP  = 10,
  INTSXP  = 13,
  REALSXP = 14,
  STRSXP  = 16,
  VECSXP  = 19
} SEXPTYPE;

typedef struct SEXPREC *SEXP;

typedef struct SEXPATTRIB {
  SEXP name;                  /* a SYMSXP. */
  SEXP value;
  struct SEXPATTRIB *next;
} SEXPATTRIB;

struct SEXPREC {

  int sxptype;
  int len;

  union {
    int    *i;                /* LGLSXP, INTSXP. */
    double *d;                /* REALSXP. */
    char   *c;                /* CHARSXP, SYMSXP. */
    SEXP   *v;                /* STRSXP (of CHARSXP), VECSXP. */
    struct { SEXP car, cdr; } cons;   /* LISTSXP, LANGSXP. */
    void   *ptr;              /* CLOSXP: a borrowed PyObject *. */
  } u;

  SEXPATTRIB *attrib;

  /* every object is linked into a per-call arena and released wholesale when
   * the call returns; `preserved` opts an object out of that. */
  struct SEXPREC *arena_next;
  int preserved;

};

/* -------------------------------------------------------------------------
 * constants.
 * ------------------------------------------------------------------------- */

extern SEXP R_NilValue;

extern SEXP R_NamesSymbol;
extern SEXP R_ClassSymbol;
extern SEXP R_LevelsSymbol;
extern SEXP R_DimSymbol;
extern SEXP R_DimNamesSymbol;
extern SEXP R_RowNamesSymbol;

#define NA_INTEGER  INT_MIN
#define NA_LOGICAL  INT_MIN
extern double R_NaReal;
extern SEXP   R_NaString;
#define NA_REAL   R_NaReal
#define NA_STRING R_NaString

#define R_NaN      (0.0/0.0)
#define R_PosInf   (1.0/0.0)
#define R_NegInf   (-1.0/0.0)

#define TRUE  1
#define FALSE 0

#include <limits.h>

/* -------------------------------------------------------------------------
 * accessors.  these are macros for the same reason they are in R: they sit in
 * the innermost loops of the scoring and independence-test code.
 * ------------------------------------------------------------------------- */

#define TYPEOF(x)   ((x)->sxptype)
#define LENGTH(x)   Rf_length(x)
#define INTEGER(x)  ((x)->u.i)
#define LOGICAL(x)  ((x)->u.i)
#define REAL(x)     ((x)->u.d)
#define CHAR(x)     ((const char *)((x)->u.c))

SEXP Rf_allocVector(SEXPTYPE type, int n);
SEXP Rf_allocMatrix(SEXPTYPE type, int nrow, int ncol);
SEXP Rf_allocLang(int n);
int  Rf_length(SEXP x);
int  Rf_nrows(SEXP x);
int  Rf_ncols(SEXP x);

SEXP Rf_STRING_ELT(SEXP x, int i);
void Rf_SET_STRING_ELT(SEXP x, int i, SEXP v);
SEXP Rf_VECTOR_ELT(SEXP x, int i);
void Rf_SET_VECTOR_ELT(SEXP x, int i, SEXP v);

#define STRING_ELT(x, i)      Rf_STRING_ELT((x), (i))
#define SET_STRING_ELT(x,i,v) Rf_SET_STRING_ELT((x), (i), (v))
#define VECTOR_ELT(x, i)      Rf_VECTOR_ELT((x), (i))
#define SET_VECTOR_ELT(x,i,v) Rf_SET_VECTOR_ELT((x), (i), (v))

/* pairlist access, used only by the three call-back sites. */
SEXP Rf_CAR(SEXP x);
SEXP Rf_CDR(SEXP x);
void Rf_SETCAR(SEXP x, SEXP v);

#define CAR(x)       Rf_CAR(x)
#define CDR(x)       Rf_CDR(x)
#define SETCAR(x, v) Rf_SETCAR((x), (v))

/* -------------------------------------------------------------------------
 * construction, coercion and type predicates.
 * ------------------------------------------------------------------------- */

SEXP Rf_mkChar(const char *s);
SEXP Rf_mkString(const char *s);
SEXP Rf_ScalarReal(double x);
SEXP Rf_ScalarInteger(int x);
SEXP Rf_ScalarLogical(int x);
SEXP Rf_coerceVector(SEXP x, SEXPTYPE type);
SEXP Rf_duplicate(SEXP x);

int    Rf_asInteger(SEXP x);
double Rf_asReal(SEXP x);
int    Rf_asLogical(SEXP x);

int Rf_isNull(SEXP x);
int Rf_isReal(SEXP x);
int Rf_isInteger(SEXP x);
int Rf_isString(SEXP x);
int Rf_isLogical(SEXP x);
int Rf_isMatrix(SEXP x);
int Rf_isFactor(SEXP x);
int Rf_isVector(SEXP x);
int Rf_isFrame(SEXP x);
int Rf_isFunction(SEXP x);

/* -------------------------------------------------------------------------
 * attributes and symbols.
 * ------------------------------------------------------------------------- */

SEXP Rf_getAttrib(SEXP x, SEXP name);
SEXP Rf_setAttrib(SEXP x, SEXP name, SEXP value);
SEXP Rf_install(const char *name);

/* -------------------------------------------------------------------------
 * vector utilities.
 *
 * bnlearn leans on match() heavily to map node labels onto column positions,
 * so it has to reproduce R's semantics exactly: one-based positions, and
 * `nomatch` (always 0 at bnlearn's call sites) for absent elements.
 * ------------------------------------------------------------------------- */

SEXP Rf_match(SEXP table, SEXP x, int nomatch);
SEXP Rf_duplicated(SEXP x, int from_last);
const char *Rf_type2char(SEXPTYPE type);

/* -------------------------------------------------------------------------
 * memory management.
 *
 * bnlearn allocates its own working memory with calloc()/free() through
 * Calloc1D()/Free1D(), so the only thing that needs managing here are the
 * SEXPs themselves.  Rather than reproduce R's generational collector and its
 * subtle PROTECT invariants, every SEXP is linked into a per-call arena and
 * the whole arena is freed when the call returns; the result is deep-copied
 * into Python objects before that happens.  PROTECT/UNPROTECT therefore have
 * nothing to do and compile away.
 * ------------------------------------------------------------------------- */

#define PROTECT(x)      (x)
#define UNPROTECT(n)    do { } while (0)

void R_PreserveObject(SEXP x);
void R_ReleaseObject(SEXP x);

/* arena control, called from the Cython layer around each entry point. */
void pybn_arena_push(void);
void pybn_arena_pop(void);

/* -------------------------------------------------------------------------
 * conditions and output.
 *
 * error() unwinds to the arena boundary with longjmp(), mirroring what R does;
 * the Cython layer turns that into a Python exception.  Rprintf() feeds
 * bnlearn's debugging output into Python's sys.stdout.
 * ------------------------------------------------------------------------- */

void Rf_error(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
void Rf_warning(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void Rprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void REprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/* R's print(), reached from one debugging path in the naive Bayes predictor.
 * The rendering only approximates R's deparse, which is fine: it is debugging
 * output, not anything the parity suite compares. */
void Rf_PrintValue(SEXP x);
#define PrintValue Rf_PrintValue

/* -------------------------------------------------------------------------
 * evaluation of user-supplied callbacks (custom scores, custom tests and the
 * modelstring() helper used for debugging output).
 * ------------------------------------------------------------------------- */

extern SEXP R_GlobalEnv;
SEXP Rf_eval(SEXP call, SEXP env);

/* -------------------------------------------------------------------------
 * numerics.
 * ------------------------------------------------------------------------- */

#define ISNAN(x)     (isnan(x) != 0)
#define R_FINITE(x)  (isfinite(x) != 0)
int R_IsNA(double x);

/* the thirteen Rmath entry points bnlearn calls, vendored from R's src/nmath
 * so that results agree with R bit for bit. */
double pchisq(double x, double df, int lower_tail, int log_p);
double pt(double x, double n, int lower_tail, int log_p);

/* the normal density and distribution function carry their arity in the
 * symbol name, and Rmath.h aliases them; nmath is compiled here in standalone
 * mode, so the aliases have to be repeated rather than inherited. */
double pnorm5(double x, double mu, double sigma, int lower_tail, int log_p);
double dnorm4(double x, double mu, double sigma, int give_log);
#define pnorm pnorm5
#define dnorm dnorm4
double dpois(double x, double lambda, int give_log);
double rpois(double mu);
double rgamma(double a, double scale);
double lgammafn(double x);
double digamma(double x);
double trigamma(double x);
double tetragamma(double x);
double logspace_add(double lx, double ly);
double fmax2(double x, double y);
double fmin2(double x, double y);

/* in standalone mode nmath exports choose() under its plain name, so unlike
 * the rest of the R API it is not remapped through an Rf_ prefix. */
double choose(double n, double k);

/* R's headers guarantee these to the packages that include them, and the
 * vendored bnlearn sources use them without defining them.  Two are R's own
 * (M_LN_SQRT_PI, M_LN_SQRT_2PI) and never come from anywhere else; the rest
 * are BSD extensions that <math.h> exposes only outside strict ANSI mode.
 * Building at c_std=c11 puts glibc in exactly that mode, so on Linux they
 * are absent -- which is why this list has to be complete rather than
 * whatever the development machine happened to be missing.  macOS defines
 * them regardless, which is why M_SQRT1_2 went unnoticed until the first
 * Linux build.
 *
 * Adding one here is the right place: the vendored sources under
 * src/c/bnlearn/ stay byte-identical to the CRAN tarball, and this header
 * is the shim that stands in for R's.  Keep it in step with what
 *   grep -rhoE '\bM_[A-Za-z0-9_]+' src/c/bnlearn/ src/c/linpack/
 * turns up.
 */
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375
#endif
#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121458176568
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.414213562373095048801688724209698079
#endif
#ifndef M_LN_SQRT_PI
#define M_LN_SQRT_PI 0.572364942924700087071713675677
#endif
#ifndef M_LN_SQRT_2PI
#define M_LN_SQRT_2PI 0.918938533204672741780329736406
#endif

/* the random number generator, also vendored from R so that seeding behaves
 * identically. */
void   GetRNGstate(void);
void   PutRNGstate(void);
double unif_rand(void);
double norm_rand(void);
void   pybn_set_seed(unsigned int seed);

/* R's sampling primitive: a uniform integer in [0, dn), drawn by rejection the
 * way R has done since 3.6.0.  sample() is built on it, and so is the
 * bootstrap's resampling. */
double R_unif_index(double dn);

/* sorting helpers from R_ext/Utils.h. */
void R_qsort(double *v, size_t i, size_t j);
void R_qsort_I(double *v, int *II, int i, int j);
void R_qsort_int(int *iv, size_t i, size_t j);
void R_qsort_int_I(int *iv, int *II, int i, int j);
void revsort(double *a, int *ib, int n);

/* LAPACK/BLAS.  bnlearn only reaches for dgesdd, dgesvd, dgetrf and dgemm;
 * they are resolved against the BLAS/LAPACK that SciPy already ships, so
 * pybnlearn needs no BLAS of its own.  Declared with the usual F77 mangling. */
#define F77_CALL(x) x ## _
#define F77_NAME(x) x ## _
#define F77_SUB(x)  x ## _

/* Fortran hidden character-length arguments.  bnlearn's rcore.h defines
 * USE_FC_LEN_T before including <R_ext/Lapack.h>, so FCONE has to expand to a
 * trailing length argument and the prototypes below have to accept them. */
#define FC_LEN_T size_t
#define FCONE , (FC_LEN_T)1

void F77_NAME(dgesdd)(const char *jobz, const int *m, const int *n, double *a,
    const int *lda, double *s, double *u, const int *ldu, double *vt,
    const int *ldvt, double *work, const int *lwork, int *iwork, int *info,
    FC_LEN_T jobz_len);
void F77_NAME(dgetrf)(const int *m, const int *n, double *a, const int *lda,
    int *ipiv, int *info);
void F77_NAME(dgemm)(const char *transa, const char *transb, const int *m,
    const int *n, const int *k, const double *alpha, const double *a,
    const int *lda, const double *b, const int *ldb, const double *beta,
    double *c, const int *ldc, FC_LEN_T transa_len, FC_LEN_T transb_len);

/* dqrdc2 and dqrsl are the LINPACK-derived QR routines that R ships in
 * src/appl/; they are not part of any LAPACK, so pybnlearn carries its own C
 * translations of R's Fortran (see compat/linpack.c).  dqrdc2 in particular is
 * R's own variant of dqrdc, with the column-pivoting tolerance that makes
 * lm() rank-deficiency handling behave the way bnlearn expects. */
void F77_NAME(dqrdc2)(double *x, const int *ldx, const int *n, const int *p,
    const double *tol, int *rank, double *qraux, int *pivot, double *work);
void F77_NAME(dqrsl)(const double *x, const int *ldx, const int *n,
    const int *k, const double *qraux, const double *y, double *qy,
    double *qty, double *b, double *rsd, double *xb, const int *job,
    int *info);

/* -------------------------------------------------------------------------
 * the remapping, kept identical to R's so that bnlearn's sources compile
 * exactly as they do under R.
 *
 * Gated on R_NO_REMAP, as R's own headers are, and for the reason R offers
 * it: these are unprefixed common words, and a translation unit that also
 * includes somebody else's headers will collide.  `length` is the one that
 * bites -- CPython's PyASCIIObject has a member of that name, so the macro
 * silently rewrites `obj->length` into `obj->Rf_length` and the compiler
 * reports a missing member in a struct this project never wrote.  It shows
 * up only on some CPython versions, because whether the accessor is inlined
 * into the generated C depends on the version, which is why it survived
 * every local build and appeared on the first CI run.
 *
 * The binding in src/pybnlearn/ defines R_NO_REMAP and calls Rf_* directly;
 * the vendored bnlearn sources do not, and must not.
 * ------------------------------------------------------------------------- */

#ifndef R_NO_REMAP

#define allocVector   Rf_allocVector
#define match         Rf_match
#define duplicated    Rf_duplicated
#define type2char     Rf_type2char
#define allocMatrix   Rf_allocMatrix
#define allocLang     Rf_allocLang
#define length        Rf_length
#define nrows         Rf_nrows
#define ncols         Rf_ncols
#define mkChar        Rf_mkChar
#define mkString      Rf_mkString
#define ScalarReal    Rf_ScalarReal
#define ScalarInteger Rf_ScalarInteger
#define ScalarLogical Rf_ScalarLogical
#define coerceVector  Rf_coerceVector
#define duplicate     Rf_duplicate
#define asInteger     Rf_asInteger
#define asReal        Rf_asReal
#define asLogical     Rf_asLogical
#define isNull        Rf_isNull
#define isReal        Rf_isReal
#define isInteger     Rf_isInteger
#define isString      Rf_isString
#define isLogical     Rf_isLogical
#define isMatrix      Rf_isMatrix
#define isFactor      Rf_isFactor
#define isVector      Rf_isVector
#define isFrame       Rf_isFrame
#define isFunction    Rf_isFunction
#define getAttrib     Rf_getAttrib
#define setAttrib     Rf_setAttrib
#define install       Rf_install
#define error         Rf_error
#define warning       Rf_warning
#define eval          Rf_eval

#endif /* R_NO_REMAP */

/* bnlearn's own convenience macros (isTRUE, INT, NUM, NODE, MIN, MAX) are not
 * defined here: they come from the vendored src/include/rcore.h, which is left
 * untouched.  MAYBE_REFERENCED is, though, because that header's fallback
 * definition reaches for NAMED() and R's reference-counting semantics.  Since
 * nothing here is ever shared between callers, reporting objects as possibly
 * referenced is the conservative answer: it makes bnlearn duplicate before
 * modifying, which is always safe. */
#define MAYBE_REFERENCED(x) 1

#endif
