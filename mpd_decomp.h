#ifndef MPD_DECOMP_H_
#define MPD_DECOMP_H_

/* $Id$ */


typedef struct {
  int L; /* L for *alm */
  double *alm; /* Size: 2L+1 */
  double *alminus1re, *alminus1im; /* Size: L+2 each.  The extra makes the
				      algorithm easier. */
  double *bre, *bim; /* Size: L each.  The extra makes things easier. */
  double v[3]; /* Vector we construct */
} mpd_decomp_t;

typedef struct {
  int L;
  double **vector;
} mpd_decomp_vector_t;

/* 
 * alm should be of size 2L+1.  alm[0]=al0, alm[1]=Re(al1), alm[2]=Im(al1), etc.
 * Returns NULL on failure.
 */
mpd_decomp_t *mpd_decomp_create (int L, double *alm);

/* Frees memory */
void mpd_decomp_destroy (mpd_decomp_t *mpd);

/*
 * Do the actual decomposition.  One vector and a rank l-1 tenor are left
 * created.  Initial values are sent in with alm and v.  These
 * are over written with the best fit (assuming we find one).  Return
 * status is from gsl.  In particular, 0 == success.
 */
int mpd_decomp_fit (mpd_decomp_t *mpd, double *alm, double *v);

/*
 * Do the full decomposition. Status returned as in mpd_decomp_fit.
 */
int mpd_decomp_full_fit (int L, double *alm, mpd_decomp_vector_t *v);

mpd_decomp_vector_t *mpd_decomp_vector_create (int L);
void mpd_decomp_vector_destroy (mpd_decomp_vector_t *v);

#endif
