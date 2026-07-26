/* pybnlearn: internals shared between the compatibility translation units and
 * the Cython binding layer.  Not included by any vendored bnlearn source.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#ifndef PYBNLEARN_RCOMPAT_INTERNAL_H
#define PYBNLEARN_RCOMPAT_INTERNAL_H

#include <setjmp.h>
#include "rcompat.h"

/* set up the constants and the symbol table; idempotent. */
void pybn_init_constants(void);

/* the unwind target for error().  The binding layer wraps every entry point
 * in setjmp(*pybn_error_target()) and, on a non-zero return, raises a Python
 * exception carrying pybn_error_message(). */
jmp_buf    *pybn_error_target(void);
const char *pybn_error_message(void);
void        pybn_clear_error(void);

/* warnings are collected rather than emitted, so the binding layer can turn
 * them into Python warnings once the call has returned. */
int         pybn_warning_count(void);
const char *pybn_warning_at(int i);
void        pybn_clear_warnings(void);

/* Rprintf() output is buffered here and flushed to Python's sys.stdout by the
 * binding layer; bnlearn writes a lot of it when debug = TRUE. */
const char *pybn_output_buffer(void);
void        pybn_clear_output(void);

/* the callback used by Rf_eval(): the binding layer installs a function that
 * calls a Python object with the arguments of the pairlist.  `fn` is the CAR
 * of the call, `args` the remaining elements, `nargs` their count. */
typedef SEXP (*pybn_call_handler)(SEXP fn, SEXP *args, int nargs);
void pybn_set_call_handler(pybn_call_handler handler);

/* wrap a borrowed Python callable so it can travel through the C core as a
 * CLOSXP and come back out again at the eval() site. */
SEXP pybn_wrap_callable(void *pyobj);
void *pybn_unwrap_callable(SEXP x);

#endif
