/* pybnlearn: exercise arcs2amat() the way the binding does, under a
 * sanitizer.
 *
 * The suite segfaulted inside amat() the first time it ran on Linux, and
 * nowhere on macOS.  A crash that appears on one libc and not another is
 * almost always a write just outside an allocation -- benign wherever the
 * allocator happens to leave slack, fatal where it does not -- and the way
 * to find one is a sanitizer rather than a hypothesis.
 *
 * This reproduces exactly what _core.pyx's arcs_to_amat() does: open an
 * arena, build the flat "from...to..." character vector and the node vector,
 * call arcs2amat, read back dims*dims integers, close the arena.  Build it
 * with -fsanitize=address and any out-of-bounds access is reported with the
 * offending line rather than a stack trace from the crash site.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include <stdio.h>
#include <string.h>
#include "../../src/c/compat/rcompat.h"
#include "../../src/c/compat/rcompat_internal.h"

extern SEXP onLoad(void);
extern SEXP arcs2amat(SEXP arcs, SEXP nodes);

/* the flat vector the binding builds: every "from", then every "to". */
static SEXP arc_vector(const char **from, const char **to, int narcs) {

SEXP out = Rf_allocVector(STRSXP, 2 * narcs);

  for (int i = 0; i < narcs; i++) {
    Rf_SET_STRING_ELT(out, i, Rf_mkChar(from[i]));
    Rf_SET_STRING_ELT(out, i + narcs, Rf_mkChar(to[i]));
  }

  return out;

}/*ARC_VECTOR*/

static SEXP node_vector(const char **names, int n) {

SEXP out = Rf_allocVector(STRSXP, n);

  for (int i = 0; i < n; i++)
    Rf_SET_STRING_ELT(out, i, Rf_mkChar(names[i]));

  return out;

}/*NODE_VECTOR*/

static int check(const char *label, const char **nodes, int nnodes,
    const char **from, const char **to, int narcs) {

SEXP amat, arcs, nodeset;
int total = 0;

  pybn_arena_push();

  arcs = arc_vector(from, to, narcs);
  nodeset = node_vector(nodes, nnodes);
  amat = arcs2amat(arcs, nodeset);

  /* read every cell, exactly as _read_int_matrix() does. */
  for (int j = 0; j < nnodes; j++)
    for (int i = 0; i < nnodes; i++)
      total += INTEGER(amat)[j * nnodes + i];

  pybn_arena_pop();

  printf("%-18s %d nodes, %d arcs -> %d ones\n", label, nnodes, narcs, total);
  return total;

}/*CHECK*/

int main(void) {

  pybn_init_constants();
  onLoad();

  {
  const char *nodes[] = {"A", "B", "C", "D"};
  const char *from[]  = {"A", "B", "C"};
  const char *to[]    = {"B", "C", "D"};
  if (check("chain", nodes, 4, from, to, 3) != 3)
    return fprintf(stderr, "chain: wrong number of arcs\n"), 1;
  }

  {
  const char *nodes[] = {"A", "B", "C"};
  const char *from[]  = {"A", "B"};
  const char *to[]    = {"C", "C"};
  if (check("collider", nodes, 3, from, to, 2) != 2)
    return fprintf(stderr, "collider: wrong number of arcs\n"), 1;
  }

  {
  /* an undirected arc is held as both directions */
  const char *nodes[] = {"A", "B", "C", "D"};
  const char *from[]  = {"A", "B", "B", "C", "C", "D"};
  const char *to[]    = {"B", "A", "C", "B", "D", "C"};
  if (check("cpdag-chain", nodes, 4, from, to, 6) != 6)
    return fprintf(stderr, "cpdag-chain: wrong number of arcs\n"), 1;
  }

  {
  const char *nodes[] = {"A", "B", "C", "D"};
  const char *from[]  = {NULL};
  const char *to[]    = {NULL};
  if (check("empty", nodes, 4, from, to, 0) != 0)
    return fprintf(stderr, "empty: expected no arcs\n"), 1;
  }

  {
  const char *nodes[] = {"A"};
  const char *from[]  = {NULL};
  const char *to[]    = {NULL};
  if (check("single", nodes, 1, from, to, 0) != 0)
    return fprintf(stderr, "single: expected no arcs\n"), 1;
  }

  {
  /* every ordered pair, which is where an off-by-one in the indexing shows */
  const char *nodes[] = {"A", "B", "C", "D"};
  const char *from[]  = {"A", "A", "A", "B", "B", "C"};
  const char *to[]    = {"B", "C", "D", "C", "D", "D"};
  if (check("complete", nodes, 4, from, to, 6) != 6)
    return fprintf(stderr, "complete: wrong number of arcs\n"), 1;
  }

  printf("all cases passed\n");
  return 0;

}/*MAIN*/
