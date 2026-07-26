/* pybnlearn: error(), warning(), Rprintf() and the eval() bridge.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include "rcompat.h"
#include "rcompat_internal.h"

/* -------------------------------------------------------------------------
 * errors.
 * ------------------------------------------------------------------------- */

#define ERRBUF_SIZE 4096

static jmp_buf error_jmp;
static char error_msg[ERRBUF_SIZE];

jmp_buf *pybn_error_target(void) {

  return &error_jmp;

}/*PYBN_ERROR_TARGET*/

const char *pybn_error_message(void) {

  return error_msg;

}/*PYBN_ERROR_MESSAGE*/

void pybn_clear_error(void) {

  error_msg[0] = '\0';

}/*PYBN_CLEAR_ERROR*/

void Rf_error(const char *fmt, ...) {

va_list ap;

  va_start(ap, fmt);
  vsnprintf(error_msg, ERRBUF_SIZE, fmt, ap);
  va_end(ap);

  /* unwind to the entry point, exactly as R's error() unwinds to the top of
   * the .Call().  The binding layer pops the arena and raises. */
  longjmp(error_jmp, 1);

}/*RF_ERROR*/

/* -------------------------------------------------------------------------
 * warnings.
 *
 * R defers warnings to the end of the top-level call, so collecting them and
 * letting the binding layer re-emit them as Python warnings matches the
 * original behaviour more closely than printing them immediately would.
 * ------------------------------------------------------------------------- */

#define MAX_WARNINGS 64
#define WARNBUF_SIZE 1024

static char warnings[MAX_WARNINGS][WARNBUF_SIZE];
static int nwarnings = 0;

void Rf_warning(const char *fmt, ...) {

va_list ap;

  if (nwarnings >= MAX_WARNINGS)
    return;

  va_start(ap, fmt);
  vsnprintf(warnings[nwarnings], WARNBUF_SIZE, fmt, ap);
  va_end(ap);

  nwarnings++;

}/*RF_WARNING*/

int pybn_warning_count(void) { return nwarnings; }
const char *pybn_warning_at(int i) {

  return (i >= 0 && i < nwarnings) ? warnings[i] : "";

}/*PYBN_WARNING_AT*/

void pybn_clear_warnings(void) { nwarnings = 0; }

/* -------------------------------------------------------------------------
 * printed output.
 *
 * bnlearn's debugging output goes through Rprintf(); it is buffered so that it
 * can be handed to Python's sys.stdout rather than written straight to the
 * process' stdout, which would bypass redirection and notebook capture.
 * ------------------------------------------------------------------------- */

static char *outbuf = NULL;
static size_t outbuf_len = 0, outbuf_cap = 0;

static void out_append(const char *s) {

size_t n = strlen(s);

  if (outbuf_len + n + 1 > outbuf_cap) {

    size_t want = (outbuf_cap ? outbuf_cap * 2 : 4096);

    while (want < outbuf_len + n + 1)
      want *= 2;

    outbuf = realloc(outbuf, want);

    if (!outbuf) {
      /* dropping debugging output is preferable to aborting the run. */
      outbuf_cap = outbuf_len = 0;
      return;
    }/*THEN*/

    outbuf_cap = want;

  }/*THEN*/

  memcpy(outbuf + outbuf_len, s, n + 1);
  outbuf_len += n;

}/*OUT_APPEND*/

void Rprintf(const char *fmt, ...) {

va_list ap;
char buf[WARNBUF_SIZE];

  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  out_append(buf);

}/*RPRINTF*/

/* REprintf() is not defined here: nmath's mlutils.c already provides one in
 * standalone mode, writing to stderr, which is exactly what this would do. */

/* R's print(), used by one debugging branch of the naive Bayes predictor. */
void Rf_PrintValue(SEXP x) {

  if (Rf_isNull(x)) {

    Rprintf("NULL\n");

    return;

  }/*THEN*/

  switch (TYPEOF(x)) {

    case LGLSXP:
    case INTSXP:
      for (int i = 0; i < Rf_length(x); i++)
        Rprintf("%s%d", i ? " " : "", INTEGER(x)[i]);
      Rprintf("\n");
      break;

    case REALSXP:
      for (int i = 0; i < Rf_length(x); i++)
        Rprintf("%s%g", i ? " " : "", REAL(x)[i]);
      Rprintf("\n");
      break;

    case STRSXP:
      for (int i = 0; i < Rf_length(x); i++)
        Rprintf("%s\"%s\"", i ? " " : "", CHAR(STRING_ELT(x, i)));
      Rprintf("\n");
      break;

    case VECSXP:
      Rprintf("[list of %d]\n", Rf_length(x));
      for (int i = 0; i < Rf_length(x); i++)
        Rf_PrintValue(VECTOR_ELT(x, i));
      break;

    default:
      Rprintf("<%s>\n", Rf_type2char(TYPEOF(x)));
      break;

  }/*SWITCH*/

}/*RF_PRINTVALUE*/

const char *pybn_output_buffer(void) {

  return outbuf ? outbuf : "";

}/*PYBN_OUTPUT_BUFFER*/

void pybn_clear_output(void) {

  outbuf_len = 0;

  if (outbuf)
    outbuf[0] = '\0';

}/*PYBN_CLEAR_OUTPUT*/

/* -------------------------------------------------------------------------
 * eval().
 *
 * Three call sites in bnlearn build a pairlist and evaluate it: the custom
 * score, the custom independence test, and the modelstring() call used when
 * printing debugging output.  All three become calls into Python.
 * ------------------------------------------------------------------------- */

static pybn_call_handler call_handler = NULL;

void pybn_set_call_handler(pybn_call_handler handler) {

  call_handler = handler;

}/*PYBN_SET_CALL_HANDLER*/

SEXP pybn_wrap_callable(void *pyobj) {

SEXP s = Rf_allocVector(LGLSXP, 1);

  /* reuse the vector allocation for its arena bookkeeping, then retag. */
  s->sxptype = CLOSXP;
  s->u.ptr = pyobj;

  return s;

}/*PYBN_WRAP_CALLABLE*/

void *pybn_unwrap_callable(SEXP x) {

  return (x && x->sxptype == CLOSXP) ? x->u.ptr : NULL;

}/*PYBN_UNWRAP_CALLABLE*/

SEXP Rf_eval(SEXP call, SEXP env) {

SEXP args[8];
SEXP fn = NULL, p = NULL;
int nargs = 0;

  (void)env;

  if (!call_handler)
    Rf_error("no callback handler is installed; custom scores, custom tests "
             "and debugging output are unavailable.");

  fn = CAR(call);

  for (p = CDR(call); p != R_NilValue && nargs < 8; p = CDR(p))
    args[nargs++] = CAR(p);

  return call_handler(fn, args, nargs);

}/*RF_EVAL*/
