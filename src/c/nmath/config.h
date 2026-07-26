/* pybnlearn: a minimal stand-in for the config.h that R's configure script
 * generates.  Three nmath sources (fprec.c, fround.c, pcauchy.c) include it
 * unconditionally, which is why a standalone Rmath build needs one at all.
 *
 * As with Rconfig.h, nothing here is probed at build time: the settings are
 * either universally true on the platforms pybnlearn targets, or deliberately
 * left undefined so that nmath takes its portable fallback path.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#ifndef PYBNLEARN_NMATH_CONFIG_H
#define PYBNLEARN_NMATH_CONFIG_H

#include "Rconfig.h"

/* atanpi() is C23 and is not yet universally available; leaving it undefined
 * makes pcauchy.c compute atan(x)/pi itself, which is what R does on every
 * platform that lacks it. */
/* #undef HAVE_ATANPI */

#endif
