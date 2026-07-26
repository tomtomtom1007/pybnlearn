/* pybnlearn: R's default random number generator.
 *
 * The standalone Rmath library ships its own unif_rand() based on
 * Marsaglia-multicarry, which is *not* what R uses: R defaults to the
 * Mersenne-Twister, seeded through a specific scrambling schedule, and draws
 * normals by inversion.  Anything else would make rbn(), cpquery(),
 * boot.strength() and random.graph() produce different numbers from R for the
 * same seed, and the parity suite could then only compare distributions
 * instead of values.  So std_unif.c is left out of the build and this file
 * takes its place.
 *
 * The generator, the seeding schedule and R_unif_index() are taken from R's
 * src/main/RNG.c, and are
 *   Copyright (C) 1995-2026 The R Core Team
 *   Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura
 * norm_rand() is not reimplemented here: nmath's snorm.c already defaults to
 * INVERSION, which is R's default too, so it agrees once unif_rand() does.
 *
 * Copyright (C) 2026 the pybnlearn authors.
 * Licensed under the GNU General Public License version 3 or later.
 */

#include <math.h>
#include <limits.h>
#include <stdint.h>

typedef unsigned int Int32;

#define i2_32m1 2.328306437080797e-10  /* 1/(2^32 - 1) */

/* -------------------------------------------------------------------------
 * the Mersenne-Twister.
 *
 * dummy[0] holds mti and dummy[1..624] the state, which is the layout R uses
 * for .Random.seed; keeping it means a future version can read and write R's
 * seed vector directly.
 * ------------------------------------------------------------------------- */

#define N 624
#define M 397
#define MATRIX_A   0x9908b0df
#define UPPER_MASK 0x80000000
#define LOWER_MASK 0x7fffffff

#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static Int32 dummy[628];
static Int32 *mt = dummy + 1;
static int mti = N + 1;

static void MT_sgenrand(Int32 seed) {

int i;

    for (i = 0; i < N; i++) {
	mt[i] = seed & 0xffff0000;
	seed = 69069 * seed + 1;
	mt[i] |= (seed & 0xffff0000) >> 16;
	seed = 69069 * seed + 1;
    }

    mti = N;

}/*MT_SGENRAND*/

static double MT_genrand(void) {

Int32 y;
static Int32 mag01[2] = { 0x0, MATRIX_A };

    mti = dummy[0];

    if (mti >= N) {

	int kk;

	if (mti == N + 1)
	    MT_sgenrand(4357);

	for (kk = 0; kk < N - M; kk++) {
	    y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
	    mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
	}
	for (; kk < N - 1; kk++) {
	    y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
	    mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
	}
	y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
	mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

	mti = 0;

    }/*THEN*/

    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);
    dummy[0] = mti;

    return ((double)y * 2.3283064365386963e-10);

}/*MT_GENRAND*/

/* -------------------------------------------------------------------------
 * seeding.
 *
 * R's set.seed() scrambles the user's seed fifty times before filling the
 * state, then fixes mti up to 624 so that the first draw regenerates the
 * block.  Both details are load-bearing for reproducing R's streams.
 * ------------------------------------------------------------------------- */

void pybn_set_seed(unsigned int seed) {

    for (int j = 0; j < 50; j++)
	seed = (69069 * seed + 1);

    for (int j = 0; j < N + 1; j++) {
	seed = (69069 * seed + 1);
	dummy[j] = seed;
    }

    /* FixupSeeds(MERSENNE_TWISTER, 1). */
    dummy[0] = 624;
    mti = 624;

}/*PYBN_SET_SEED*/

/* the standalone Rmath entry points, kept so that anything linking against
 * this expecting libRmath's interface still works. */
void set_seed(unsigned int i1, unsigned int i2) {

    (void)i2;
    pybn_set_seed(i1);

}/*SET_SEED*/

void get_seed(unsigned int *i1, unsigned int *i2) {

    *i1 = dummy[0];
    *i2 = dummy[1];

}/*GET_SEED*/

/* R saves and restores .Random.seed around each .Call(); pybnlearn keeps the
 * state in this process for the lifetime of the interpreter, so there is
 * nothing to synchronise. */
void GetRNGstate(void) { }
void PutRNGstate(void) { }

/* -------------------------------------------------------------------------
 * the uniform stream.
 * ------------------------------------------------------------------------- */

static double fixup(double x) {

    /* ensure 0 and 1 are never returned */
    if (x <= 0.0) return 0.5 * i2_32m1;
    if ((1.0 - x) <= 0.0) return 1.0 - 0.5 * i2_32m1;

    return x;

}/*FIXUP*/

double unif_rand(void) {

    if (mti == N + 1 && dummy[0] == 0)
	pybn_set_seed(4357);

    return fixup(MT_genrand());

}/*UNIF_RAND*/

/* generate a random non-negative integer < 2^bits in 16 bit chunks. */
static double rbits(int bits) {

uint_least64_t v = 0;

    for (int n = 0; n <= bits; n += 16) {
	int v1 = (int) floor(unif_rand() * 65536);
	v = 65536 * v + v1;
    }

    const uint_least64_t one64 = 1L;

    return (double) (v & ((one64 << bits) - 1));

}/*RBITS*/

/* R has defaulted to rejection sampling here since 3.6.0; the older rounding
 * method is what changed sample()'s results in that release, so matching the
 * current default is what parity with a current R means. */
double R_unif_index(double dn) {

    if (dn <= 0)
	return 0.0;

    int bits = (int) ceil(log2(dn));
    double dv;

    do { dv = rbits(bits); } while (dn <= dv);

    return dv;

}/*R_UNIF_INDEX*/
