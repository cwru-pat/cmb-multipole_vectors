#ifndef MPD_DECOMP_H_
#define MPD_DECOMP_H_

/* $Id: mpd_decomp.h,v 1.3 2003/05/09 03:45:36 copi Exp $ */

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t L; /* L for *alm */
  double *alm; /* Size: 2L+1 */
  double *alminus1re, *alminus1im; /* Size: L+2 each.  The extra entry
				      makes the algorithm easier. */
  double *bre, *bim; /* Size: L each.  The extra entry makes things easier. */
  double v[3]; /* Vector we construct */
} mpd_decomp_t;

typedef struct {
  size_t L;
  double **vector;
} mpd_decomp_vector_t;

/* 
 * alm should be of size 2L+1.  alm[0]=al0, alm[1]=Re(al1), alm[2]=Im(al1), etc.
 * Returns NULL on failure.
 */
mpd_decomp_t *mpd_decomp_create (size_t L, double *alm);

/* Frees memory */
void mpd_decomp_destroy (mpd_decomp_t *mpd);

/*
 * Do the actual decomposition.  One vector and a rank l-1 tensor are 
 * returned.  Initial values are sent in with alm and v.  These
 * are over written with the best fit (assuming we find one).  Return
 * status is from gsl.  In particular, 0 == success.
 */
int mpd_decomp_fit (mpd_decomp_t *mpd, double *alm, double *v);

/*
 * Do the full decomposition returning all the vectors. Status returned as
 * in mpd_decomp_fit.
 */
int mpd_decomp_full_fit (size_t L, double *alm, mpd_decomp_vector_t *v);

mpd_decomp_vector_t *mpd_decomp_vector_create (size_t L);
void mpd_decomp_vector_destroy (mpd_decomp_vector_t *v);

#ifdef __cplusplus
}
#endif

#endif
