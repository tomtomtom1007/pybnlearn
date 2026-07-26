/* pybnlearn: no-op stand-ins for R's dynamic-registration machinery.
 *
 * bnlearn's globals.c ends with R_init_bnlearn(), which hands R a table of the
 * .Call() entry points.  pybnlearn reaches those entry points directly from
 * Cython, so the table is never consulted -- but globals.c also defines the
 * BN_*Symbol globals and onLoad(), which are needed, so the file is compiled
 * whole and these three symbols are supplied to satisfy the linker.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include "rcompat.h"
#include "rapi/R_ext/Rdynload.h"

int R_registerRoutines(DllInfo *info, const R_CMethodDef *croutines,
    const R_CallMethodDef *callRoutines, const void *fortranRoutines,
    const R_ExternalMethodDef *externalRoutines) {

  (void)info; (void)croutines; (void)callRoutines;
  (void)fortranRoutines; (void)externalRoutines;

  return 1;

}/*R_REGISTERROUTINES*/

void R_useDynamicSymbols(DllInfo *info, int onoff) {

  (void)info; (void)onoff;

}/*R_USEDYNAMICSYMBOLS*/

void R_forceSymbols(DllInfo *info, int onoff) {

  (void)info; (void)onoff;

}/*R_FORCESYMBOLS*/
