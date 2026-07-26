/* pybnlearn: the guarded entry into the C core.
 *
 * bnlearn signals failure the way R does, by calling error(), which unwinds
 * with longjmp().  That unwinding must not cross Cython-generated code: Cython
 * emits reference-counting and cleanup around every block, and a longjmp past
 * it would leak or corrupt Python objects.  So setjmp() lives here, in plain
 * C, and Cython only ever sees a return code.
 *
 * The arena is deliberately *not* pushed or popped here.  The result of a call
 * is a SEXP that lives in the arena, and it has to be copied into Python
 * objects before the arena goes away -- which is work that belongs on the
 * Cython side.  The caller therefore brackets this with pybn_arena_push() and
 * pybn_arena_pop(), and the pop happens on the error path too.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include <setjmp.h>
#include "rcompat.h"
#include "rcompat_internal.h"

typedef SEXP (*fn1)(SEXP);
typedef SEXP (*fn2)(SEXP, SEXP);
typedef SEXP (*fn3)(SEXP, SEXP, SEXP);
typedef SEXP (*fn4)(SEXP, SEXP, SEXP, SEXP);
typedef SEXP (*fn5)(SEXP, SEXP, SEXP, SEXP, SEXP);
typedef SEXP (*fn6)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
typedef SEXP (*fn7)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
typedef SEXP (*fn8)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
typedef SEXP (*fn9)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
typedef SEXP (*fn10)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
    SEXP);
typedef SEXP (*fn11)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
    SEXP, SEXP);
typedef SEXP (*fn12)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
    SEXP, SEXP, SEXP);

#define A(i) args[i]

/* returns 0 and sets *out on success; returns 1 on error, with the message
 * available from pybn_error_message(). */
int pybn_protected_call(void *fn, SEXP *args, int nargs, SEXP *out) {

  pybn_clear_error();

  if (setjmp(*pybn_error_target()) != 0)
    return 1;

  switch (nargs) {

    case  1: *out = ((fn1)fn)(A(0)); break;
    case  2: *out = ((fn2)fn)(A(0), A(1)); break;
    case  3: *out = ((fn3)fn)(A(0), A(1), A(2)); break;
    case  4: *out = ((fn4)fn)(A(0), A(1), A(2), A(3)); break;
    case  5: *out = ((fn5)fn)(A(0), A(1), A(2), A(3), A(4)); break;
    case  6: *out = ((fn6)fn)(A(0), A(1), A(2), A(3), A(4), A(5)); break;
    case  7: *out = ((fn7)fn)(A(0), A(1), A(2), A(3), A(4), A(5), A(6)); break;
    case  8: *out = ((fn8)fn)(A(0), A(1), A(2), A(3), A(4), A(5), A(6),
               A(7)); break;
    case  9: *out = ((fn9)fn)(A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7),
               A(8)); break;
    case 10: *out = ((fn10)fn)(A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7),
               A(8), A(9)); break;
    case 11: *out = ((fn11)fn)(A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7),
               A(8), A(9), A(10)); break;
    case 12: *out = ((fn12)fn)(A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7),
               A(8), A(9), A(10), A(11)); break;

    default:
      Rf_error("entry points take between 1 and 12 arguments, got %d.", nargs);

  }/*SWITCH*/

  return 0;

}/*PYBN_PROTECTED_CALL*/
