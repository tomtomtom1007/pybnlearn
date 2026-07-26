/* pybnlearn: the R object model, reimplemented for use without R.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include "rcompat.h"
#include "rcompat_internal.h"

/* -------------------------------------------------------------------------
 * the arena.
 *
 * Every SEXP created while a call is in flight is linked into the arena that
 * is current at the time; pybn_arena_pop() frees the lot.  Nesting is
 * supported so that a callback into Python which itself re-enters the C core
 * does not free its caller's objects.
 * ------------------------------------------------------------------------- */

typedef struct arena {
  SEXP head;
  struct arena *parent;
} arena;

static arena *current_arena = NULL;

/* objects handed to R_PreserveObject outlive their arena and live here. */
static SEXP preserved_head = NULL;

void pybn_arena_push(void) {

arena *a = calloc(1, sizeof(arena));

  if (!a) {
    fprintf(stderr, "pybnlearn: out of memory allocating an arena.\n");
    abort();
  }/*THEN*/

  a->parent = current_arena;
  current_arena = a;

}/*PYBN_ARENA_PUSH*/

static void free_one(SEXP s) {

SEXPATTRIB *at = NULL, *next = NULL;

  for (at = s->attrib; at; at = next) {

    next = at->next;
    free(at);

  }/*FOR*/

  switch (s->sxptype) {

    case LGLSXP:
    case INTSXP:
      free(s->u.i);
      break;

    case REALSXP:
      free(s->u.d);
      break;

    case CHARSXP:
    case SYMSXP:
      free(s->u.c);
      break;

    case STRSXP:
    case VECSXP:
      free(s->u.v);
      break;

    default:
      break;

  }/*SWITCH*/

  free(s);

}/*FREE_ONE*/

void pybn_arena_pop(void) {

arena *a = current_arena;
SEXP s = NULL, next = NULL;

  if (!a)
    return;

  for (s = a->head; s; s = next) {

    next = s->arena_next;

    /* preserved objects have been moved off the arena list already. */
    if (!s->preserved)
      free_one(s);

  }/*FOR*/

  current_arena = a->parent;
  free(a);

}/*PYBN_ARENA_POP*/

static SEXP new_sexp(SEXPTYPE type, int n) {

SEXP s = calloc(1, sizeof(struct SEXPREC));

  if (!s)
    Rf_error("unable to allocate a %d-element vector.", n);

  s->sxptype = type;
  s->len = n;

  if (current_arena) {

    s->arena_next = current_arena->head;
    current_arena->head = s;

  }/*THEN*/

  return s;

}/*NEW_SEXP*/

void R_PreserveObject(SEXP x) {

  if (!x || x == R_NilValue)
    return;

  x->preserved = 1;
  x->arena_next = preserved_head;
  preserved_head = x;

}/*R_PRESERVEOBJECT*/

void R_ReleaseObject(SEXP x) {

  if (x)
    x->preserved = 0;

}/*R_RELEASEOBJECT*/

/* -------------------------------------------------------------------------
 * the singletons.
 * ------------------------------------------------------------------------- */

static struct SEXPREC nil_value = { .sxptype = NILSXP, .len = 0 };
SEXP R_NilValue = &nil_value;

static struct SEXPREC global_env = { .sxptype = NILSXP, .len = 0 };
SEXP R_GlobalEnv = &global_env;

double R_NaReal = 0.0;
SEXP   R_NaString = NULL;

SEXP R_NamesSymbol    = NULL;
SEXP R_ClassSymbol    = NULL;
SEXP R_LevelsSymbol   = NULL;
SEXP R_DimSymbol      = NULL;
SEXP R_DimNamesSymbol = NULL;
SEXP R_RowNamesSymbol = NULL;

/* symbols are interned in a small open list; bnlearn only ever installs a
 * couple of dozen of them. */
#define MAX_SYMBOLS 256
static SEXP symbol_table[MAX_SYMBOLS];
static int nsymbols = 0;

SEXP Rf_install(const char *name) {

SEXP s = NULL;

  for (int i = 0; i < nsymbols; i++)
    if (strcmp(symbol_table[i]->u.c, name) == 0)
      return symbol_table[i];

  if (nsymbols >= MAX_SYMBOLS)
    Rf_error("the symbol table is full, cannot install '%s'.", name);

  /* symbols are permanent: allocate them outside any arena. */
  s = calloc(1, sizeof(struct SEXPREC));
  if (!s)
    Rf_error("unable to install the symbol '%s'.", name);

  s->sxptype = SYMSXP;
  s->len = 1;
  s->preserved = 1;
  s->u.c = strdup(name);

  symbol_table[nsymbols++] = s;

  return s;

}/*RF_INSTALL*/

void pybn_init_constants(void) {

static int done = 0;

  if (done)
    return;

  /* R's NA_real_ is a quiet NaN carrying the payload 1954. */
  {
    volatile union { double d; unsigned int w[2]; } na;
    int big_endian = (*(const char *)(const int[]){1} == 0);

    na.d = 0.0/0.0;
    if (big_endian)
      na.w[1] = 1954;
    else
      na.w[0] = 1954;
    R_NaReal = na.d;
  }

  R_NamesSymbol    = Rf_install("names");
  R_ClassSymbol    = Rf_install("class");
  R_LevelsSymbol   = Rf_install("levels");
  R_DimSymbol      = Rf_install("dim");
  R_DimNamesSymbol = Rf_install("dimnames");
  R_RowNamesSymbol = Rf_install("row.names");

  R_NaString = calloc(1, sizeof(struct SEXPREC));
  R_NaString->sxptype = CHARSXP;
  R_NaString->len = 2;
  R_NaString->preserved = 1;
  R_NaString->u.c = strdup("NA");

  done = 1;

}/*PYBN_INIT_CONSTANTS*/

int R_IsNA(double x) {

volatile union { double d; unsigned int w[2]; } na;
int big_endian = (*(const char *)(const int[]){1} == 0);

  if (!isnan(x))
    return 0;

  na.d = x;

  return (big_endian ? na.w[1] : na.w[0]) == 1954;

}/*R_ISNA*/

/* -------------------------------------------------------------------------
 * allocation.
 * ------------------------------------------------------------------------- */

SEXP Rf_allocVector(SEXPTYPE type, int n) {

SEXP s = new_sexp(type, n);

  if (n < 0)
    Rf_error("negative vector length (%d).", n);

  switch (type) {

    case LGLSXP:
    case INTSXP:
      s->u.i = n ? calloc((size_t)n, sizeof(int)) : NULL;
      if (n && !s->u.i)
        Rf_error("unable to allocate an integer vector of length %d.", n);
      break;

    case REALSXP:
      s->u.d = n ? calloc((size_t)n, sizeof(double)) : NULL;
      if (n && !s->u.d)
        Rf_error("unable to allocate a real vector of length %d.", n);
      break;

    case STRSXP:
    case VECSXP:
      s->u.v = n ? calloc((size_t)n, sizeof(SEXP)) : NULL;
      if (n && !s->u.v)
        Rf_error("unable to allocate a list of length %d.", n);
      /* R initialises string vectors to "" and lists to NULL; both are
       * represented by R_NilValue slots that the accessors resolve. */
      for (int i = 0; i < n; i++)
        s->u.v[i] = R_NilValue;
      break;

    case NILSXP:
      return R_NilValue;

    default:
      Rf_error("cannot allocate a vector of type %d.", (int)type);

  }/*SWITCH*/

  return s;

}/*RF_ALLOCVECTOR*/

SEXP Rf_allocMatrix(SEXPTYPE type, int nrow, int ncol) {

SEXP s = Rf_allocVector(type, nrow * ncol);
SEXP dim = Rf_allocVector(INTSXP, 2);

  INTEGER(dim)[0] = nrow;
  INTEGER(dim)[1] = ncol;
  Rf_setAttrib(s, R_DimSymbol, dim);

  return s;

}/*RF_ALLOCMATRIX*/

SEXP Rf_allocLang(int n) {

SEXP head = NULL, tail = NULL;

  if (n <= 0)
    return R_NilValue;

  for (int i = 0; i < n; i++) {

    SEXP cell = new_sexp(i == 0 ? LANGSXP : LISTSXP, 1);

    cell->u.cons.car = R_NilValue;
    cell->u.cons.cdr = R_NilValue;

    if (!head)
      head = tail = cell;
    else {
      tail->u.cons.cdr = cell;
      tail = cell;
    }/*ELSE*/

  }/*FOR*/

  return head;

}/*RF_ALLOCLANG*/

SEXP Rf_CAR(SEXP x) { return (x && x != R_NilValue) ? x->u.cons.car : R_NilValue; }
SEXP Rf_CDR(SEXP x) { return (x && x != R_NilValue) ? x->u.cons.cdr : R_NilValue; }
void Rf_SETCAR(SEXP x, SEXP v) { if (x && x != R_NilValue) x->u.cons.car = v; }

/* -------------------------------------------------------------------------
 * lengths and element access.
 * ------------------------------------------------------------------------- */

int Rf_length(SEXP x) {

  if (!x || x == R_NilValue)
    return 0;

  if (x->sxptype == LANGSXP || x->sxptype == LISTSXP) {

    int n = 0;

    for (SEXP p = x; p && p != R_NilValue; p = p->u.cons.cdr)
      n++;

    return n;

  }/*THEN*/

  return x->len;

}/*RF_LENGTH*/

int Rf_nrows(SEXP x) {

SEXP dim = Rf_getAttrib(x, R_DimSymbol);

  if (dim != R_NilValue && Rf_length(dim) >= 1)
    return INTEGER(dim)[0];

  /* data frames are lists of equal-length columns. */
  if (x->sxptype == VECSXP)
    return x->len ? Rf_length(VECTOR_ELT(x, 0)) : 0;

  return Rf_length(x);

}/*RF_NROWS*/

int Rf_ncols(SEXP x) {

SEXP dim = Rf_getAttrib(x, R_DimSymbol);

  if (dim != R_NilValue && Rf_length(dim) >= 2)
    return INTEGER(dim)[1];

  if (x->sxptype == VECSXP)
    return x->len;

  return 1;

}/*RF_NCOLS*/

SEXP Rf_STRING_ELT(SEXP x, int i) {

  if (!x || x == R_NilValue || i < 0 || i >= x->len)
    Rf_error("string subscript out of bounds.");

  return x->u.v[i] ? x->u.v[i] : R_NilValue;

}/*RF_STRING_ELT*/

void Rf_SET_STRING_ELT(SEXP x, int i, SEXP v) {

  if (!x || x == R_NilValue || i < 0 || i >= x->len)
    Rf_error("string subscript out of bounds.");

  x->u.v[i] = v;

}/*RF_SET_STRING_ELT*/

SEXP Rf_VECTOR_ELT(SEXP x, int i) {

  if (!x || x == R_NilValue || i < 0 || i >= x->len)
    Rf_error("list subscript out of bounds.");

  return x->u.v[i] ? x->u.v[i] : R_NilValue;

}/*RF_VECTOR_ELT*/

void Rf_SET_VECTOR_ELT(SEXP x, int i, SEXP v) {

  if (!x || x == R_NilValue || i < 0 || i >= x->len)
    Rf_error("list subscript out of bounds.");

  x->u.v[i] = v;

}/*RF_SET_VECTOR_ELT*/

/* -------------------------------------------------------------------------
 * construction and coercion.
 * ------------------------------------------------------------------------- */

SEXP Rf_mkChar(const char *s) {

SEXP c = new_sexp(CHARSXP, s ? (int)strlen(s) : 0);

  c->u.c = strdup(s ? s : "");

  if (!c->u.c)
    Rf_error("unable to allocate a string.");

  return c;

}/*RF_MKCHAR*/

SEXP Rf_mkString(const char *s) {

SEXP v = Rf_allocVector(STRSXP, 1);

  SET_STRING_ELT(v, 0, Rf_mkChar(s));

  return v;

}/*RF_MKSTRING*/

SEXP Rf_ScalarReal(double x) {

SEXP v = Rf_allocVector(REALSXP, 1);

  REAL(v)[0] = x;

  return v;

}/*RF_SCALARREAL*/

SEXP Rf_ScalarInteger(int x) {

SEXP v = Rf_allocVector(INTSXP, 1);

  INTEGER(v)[0] = x;

  return v;

}/*RF_SCALARINTEGER*/

SEXP Rf_ScalarLogical(int x) {

SEXP v = Rf_allocVector(LGLSXP, 1);

  LOGICAL(v)[0] = x;

  return v;

}/*RF_SCALARLOGICAL*/

SEXP Rf_coerceVector(SEXP x, SEXPTYPE type) {

int n = Rf_length(x);
SEXP out = NULL;

  if (x->sxptype == type)
    return x;

  out = Rf_allocVector(type, n);

  for (int i = 0; i < n; i++) {

    switch (type) {

      case REALSXP:
        if (x->sxptype == INTSXP || x->sxptype == LGLSXP)
          REAL(out)[i] = (INTEGER(x)[i] == NA_INTEGER) ?
                           NA_REAL : (double)INTEGER(x)[i];
        else
          Rf_error("unsupported coercion to a real vector.");
        break;

      case INTSXP:
        if (x->sxptype == REALSXP)
          INTEGER(out)[i] = ISNAN(REAL(x)[i]) ?
                              NA_INTEGER : (int)REAL(x)[i];
        else if (x->sxptype == LGLSXP)
          INTEGER(out)[i] = LOGICAL(x)[i];
        else
          Rf_error("unsupported coercion to an integer vector.");
        break;

      default:
        Rf_error("unsupported coercion to type %d.", (int)type);

    }/*SWITCH*/

  }/*FOR*/

  return out;

}/*RF_COERCEVECTOR*/

SEXP Rf_duplicate(SEXP x) {

SEXP out = NULL;
int n = 0;

  if (!x || x == R_NilValue)
    return R_NilValue;

  n = x->len;

  switch (x->sxptype) {

    case LGLSXP:
    case INTSXP:
      out = Rf_allocVector(x->sxptype, n);
      memcpy(INTEGER(out), INTEGER(x), (size_t)n * sizeof(int));
      break;

    case REALSXP:
      out = Rf_allocVector(REALSXP, n);
      memcpy(REAL(out), REAL(x), (size_t)n * sizeof(double));
      break;

    case STRSXP:
      out = Rf_allocVector(STRSXP, n);
      for (int i = 0; i < n; i++)
        SET_STRING_ELT(out, i, Rf_mkChar(CHAR(STRING_ELT(x, i))));
      break;

    case VECSXP:
      out = Rf_allocVector(VECSXP, n);
      for (int i = 0; i < n; i++)
        SET_VECTOR_ELT(out, i, Rf_duplicate(VECTOR_ELT(x, i)));
      break;

    case CHARSXP:
      return Rf_mkChar(CHAR(x));

    default:
      return x;

  }/*SWITCH*/

  /* attributes are copied too, as R does for vectors. */
  for (SEXPATTRIB *at = x->attrib; at; at = at->next)
    Rf_setAttrib(out, at->name, Rf_duplicate(at->value));

  return out;

}/*RF_DUPLICATE*/

int Rf_asInteger(SEXP x) {

  if (Rf_length(x) < 1)
    return NA_INTEGER;

  switch (x->sxptype) {
    case INTSXP:
    case LGLSXP:  return INTEGER(x)[0];
    case REALSXP: return ISNAN(REAL(x)[0]) ? NA_INTEGER : (int)REAL(x)[0];
    default:      return NA_INTEGER;
  }/*SWITCH*/

}/*RF_ASINTEGER*/

double Rf_asReal(SEXP x) {

  if (Rf_length(x) < 1)
    return NA_REAL;

  switch (x->sxptype) {
    case REALSXP: return REAL(x)[0];
    case INTSXP:
    case LGLSXP:  return (INTEGER(x)[0] == NA_INTEGER) ?
                           NA_REAL : (double)INTEGER(x)[0];
    default:      return NA_REAL;
  }/*SWITCH*/

}/*RF_ASREAL*/

int Rf_asLogical(SEXP x) {

  return Rf_asInteger(x);

}/*RF_ASLOGICAL*/

/* -------------------------------------------------------------------------
 * type predicates.
 * ------------------------------------------------------------------------- */

int Rf_isNull(SEXP x)     { return !x || x == R_NilValue; }
int Rf_isReal(SEXP x)     { return x && x->sxptype == REALSXP; }
int Rf_isInteger(SEXP x)  { return x && x->sxptype == INTSXP; }
int Rf_isString(SEXP x)   { return x && x->sxptype == STRSXP; }
int Rf_isLogical(SEXP x)  { return x && x->sxptype == LGLSXP; }
int Rf_isFunction(SEXP x) { return x && x->sxptype == CLOSXP; }

int Rf_isMatrix(SEXP x) {

SEXP dim = Rf_getAttrib(x, R_DimSymbol);

  return dim != R_NilValue && Rf_length(dim) == 2;

}/*RF_ISMATRIX*/

int Rf_isFactor(SEXP x) {

  return x && x->sxptype == INTSXP &&
         Rf_getAttrib(x, R_LevelsSymbol) != R_NilValue;

}/*RF_ISFACTOR*/

int Rf_isVector(SEXP x) {

  if (!x)
    return 0;

  switch (x->sxptype) {
    case LGLSXP: case INTSXP: case REALSXP: case STRSXP: case VECSXP:
      return 1;
    default:
      return 0;
  }/*SWITCH*/

}/*RF_ISVECTOR*/

int Rf_isFrame(SEXP x) {

SEXP klass = Rf_getAttrib(x, R_ClassSymbol);

  if (klass == R_NilValue)
    return 0;

  for (int i = 0; i < Rf_length(klass); i++)
    if (strcmp(CHAR(STRING_ELT(klass, i)), "data.frame") == 0)
      return 1;

  return 0;

}/*RF_ISFRAME*/

/* -------------------------------------------------------------------------
 * attributes.
 * ------------------------------------------------------------------------- */

SEXP Rf_getAttrib(SEXP x, SEXP name) {

  if (!x || x == R_NilValue)
    return R_NilValue;

  for (SEXPATTRIB *at = x->attrib; at; at = at->next)
    if (at->name == name)
      return at->value;

  return R_NilValue;

}/*RF_GETATTRIB*/

SEXP Rf_setAttrib(SEXP x, SEXP name, SEXP value) {

SEXPATTRIB *at = NULL;

  if (!x || x == R_NilValue)
    return value;

  for (at = x->attrib; at; at = at->next) {

    if (at->name == name) {

      at->value = value;

      return value;

    }/*THEN*/

  }/*FOR*/

  at = calloc(1, sizeof(SEXPATTRIB));

  if (!at)
    Rf_error("unable to allocate an attribute.");

  at->name = name;
  at->value = value;
  at->next = x->attrib;
  x->attrib = at;

  return value;

}/*RF_SETATTRIB*/
