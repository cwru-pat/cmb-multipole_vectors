#ifndef MPD_DECOMP_H_
#define MPD_DECOMP_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MPD_VERSION "1.30"

/* Information used internally by mpd_decomp_fit to do the decomposition.
 * Use mpd_decomp_create to initially allocate and build it from a list of
 * alm.
 */
typedef struct {
  unsigned int L; /* L for *alm */
  double *alm; /* Size: 2L+1 */
  double *alminus1re, *alminus1im; /* Size: L+2 each.  The extra entry
				      makes the algorithm easier. */
  double v[3]; /* Vector we construct */
} mpd_decomp_t;

/* Structure for returning the full decomposition of a rank L symmetric,
 * traceless tensor.  The unit vectors are returned in vector and the
 * overall normalization in norm.
 */
typedef struct {
  unsigned int L;
  double **vector; /* Size [L][3] */
  double norm;
} mpd_decomp_vector_t;

/*
 * Do the actual decomposition.  One vector and a rank l-1 tensor are 
 * returned.  Initial values are sent in with alm and v.  These
 * are over written with the best fit (assuming we find one).  Return
 * status is from gsl.  In particular, 0 == success.
 */
int mpd_decomp_fit (mpd_decomp_t *mpd, double *alm, double *v);

/*
 * Do the full decomposition returning all the vectors. Status returned as
 * in mpd_decomp_fit.  Note that this makes a number of assumptions to get
 * things working, such as randomly picking starting values.  These can
 * easily fail.  The code for this routine can be used as a starting point
 * for building your own decomposition loop using mpd_decomp_fit for each
 * decomposition.
 */
int mpd_decomp_full_fit (unsigned int L, double *alm, mpd_decomp_vector_t *v);


/* Routines to properly create and destroy the above structures. */

/* 
 * alm should be of size 2L+1.  alm[0]=al0, alm[1]=Re(al1), alm[2]=Im(al1), etc.
 * Returns NULL on failure.
 */
mpd_decomp_t *mpd_decomp_create (unsigned int L, double *alm);
void mpd_decomp_destroy (mpd_decomp_t *mpd);

mpd_decomp_vector_t *mpd_decomp_vector_create (unsigned int L);
void mpd_decomp_vector_destroy (mpd_decomp_vector_t *v);

#ifdef __cplusplus
}
#endif

#endif
