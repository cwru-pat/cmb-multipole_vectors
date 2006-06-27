/*
 * Copyright (c) 2003 Craig J Copi
 * All rights reserved.
  
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
  
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_multiroots.h>
#include "mpd_decomp.h"


static char const rcsid []
  = "$Id: mpd_decomp.c,v 1.11 2006/06/27 17:34:04 copi Exp $";


/* Internal functions needed for the fitting */
static int internal_mpd_decomp_f (const gsl_vector *x, void *params,
				  gsl_vector *f);
#if 0
/* These do not appear to be necessary.  We are "only" solving quadratic
 * equations and routines without derivative information seem sufficient. */
static int internal_mpd_decomp_df (const gsl_vector *x, void *params,
				   gsl_matrix *J);
static int internal_mpd_decomp_fdf (const gsl_vector *x, void *params,
				   gsl_vector *f, gsl_matrix *J);
#endif
static void internal_set_mpd_params (mpd_decomp_t *mpd, const gsl_vector *x);
static double internal_C0 (unsigned int L, int M);
static double internal_Cp1 (unsigned int L, int M);
static double internal_Cm1 (unsigned int L, int M);
static double internal_D0 (unsigned int L, int M);
static double internal_Dp1 (unsigned int L, int M);
static double internal_Dm1 (unsigned int L, int M);
#ifdef MPD_DEBUG
static void internal_print_state (unsigned int iter, mpd_decomp_t *mpd,
				  gsl_multiroot_fsolver *s);
#endif

/* 
 * alm should be of size 2L+1.  alm[0]=al0, alm[1]=Re(al1), alm[2]=Im(al1), etc.
 * Returns NULL on failure.
 */
mpd_decomp_t *mpd_decomp_create (unsigned int L, double *alm)
{
  mpd_decomp_t *mpd;

  mpd = (mpd_decomp_t *) malloc (sizeof (mpd_decomp_t));
  if (mpd == NULL) return NULL;

  mpd->L = L;
  mpd->alm = mpd->alminus1re = mpd->alminus1im = NULL;
  mpd->bre = mpd->bim = NULL;

  mpd->alm = (double *) malloc ((2*L+1)*sizeof (double));

  /*
   * Note: We make the sizes of these L+2 so we can access [0] ... [L+1]
   * without errors.  We initialize them to 0 for extra safety.  The
   * algorithm handles all of this on its own too (only multiplying [L]
   * times 0, etc) but it costs us little to be safe here.
   */
  mpd->alminus1re = (double *) calloc ((L+2), sizeof (double));
  mpd->alminus1im = (double *) calloc ((L+2), sizeof (double));

  if ((mpd->alm == NULL) || (mpd->alminus1re == NULL)
      || (mpd->alminus1im == NULL)) {
    mpd_decomp_destroy (mpd);
    return NULL;
  }

  /*
   * Note: We make the sizes of these L-1 and are careful to ONLY access these!
   */
  mpd->bre = (double *) calloc (L-1, sizeof (double));
  mpd->bim = (double *) calloc (L-1, sizeof (double));

  if ((mpd->bre == NULL) || (mpd->bim == NULL)) {
    mpd_decomp_destroy (mpd);
    return NULL;
  }

  (void)memcpy (mpd->alm, alm, (2*L+1)*sizeof (alm[0]));

  return mpd;
}

void mpd_decomp_destroy (mpd_decomp_t *mpd)
{
  if (mpd == NULL) return;
  if (mpd->alm != NULL) free (mpd->alm);
  if (mpd->alminus1re != NULL) free (mpd->alminus1re);
  if (mpd->alminus1im != NULL) free (mpd->alminus1im);
  if (mpd->bre != NULL) free (mpd->bre);
  if (mpd->bim != NULL) free (mpd->bim);

  free (mpd);
}

/*
 * Do the actual fit.  Initial values are sent in with alm and v.  These
 * are over written with the best fit (assuming we find one).  Return
 * status is from gsl.  In particular, 0 == success.
 */
int mpd_decomp_fit (mpd_decomp_t *mpd, double *alm, double *v)
{
  const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_dnewton;
  gsl_multiroot_fsolver *s;
  int status;
  unsigned int i, offset, iter=0;
  unsigned int N = 4*mpd->L - 1; /* Number of parameters to fit */

  gsl_multiroot_function f = { &internal_mpd_decomp_f, N, mpd };
  gsl_vector *x = gsl_vector_alloc (N);

  /* Initialize */
  for (i=0; i < 2*mpd->L-1; ++i) gsl_vector_set (x, i, alm[i]);
  /* Totally arbitrarily set */
  offset = 2*mpd->L-1;
  for (i=0; i < 2*mpd->L-3; ++i) gsl_vector_set (x, i+offset, alm[offset-1-i]);
  for (i=0; i < 3; ++i) gsl_vector_set (x, N-3+i, v[i]);

  s = gsl_multiroot_fsolver_alloc (T, N);
  gsl_multiroot_fsolver_set (s, &f, x);

#ifdef MPD_DEBUG
  internal_print_state (iter, mpd, s);
#endif

  do {
    ++iter;
    status = gsl_multiroot_fsolver_iterate (s);
#ifdef MPD_DEBUG
    internal_print_state (iter, mpd, s);
#endif
    if (status) break;

    status = gsl_multiroot_test_residual (s->f, 1e-10);
  } while ((status == GSL_CONTINUE) && (iter < 1000));

#ifdef MPD_DEBUG
  printf ("status = %d, %s\n", status, gsl_strerror(status));
#endif

  if (status == GSL_SUCCESS) { /* Over write input arrays */
    for (i=0; i < 2*mpd->L-1; ++i) alm[i] = gsl_vector_get (s->x, i);
    /* Ignore bre, bim since we don't need them!! */
    for (i=0; i < 3; ++i) v[i] = gsl_vector_get (s->x, N-3+i);
  }

  gsl_multiroot_fsolver_free (s);
  gsl_vector_free (x);
  return status;
}

/* For the full fits/all the vectors. */
mpd_decomp_vector_t *mpd_decomp_vector_create (unsigned int L)
{
  mpd_decomp_vector_t *v;
  unsigned int l;

  v = (mpd_decomp_vector_t *) malloc (sizeof (mpd_decomp_vector_t));
  if (v == NULL) return NULL;

  v->L = L;

  v->vector = (double **) calloc (L, sizeof (double *));
  if (v->vector == NULL) goto ERROR;

  for (l=0; l < L; ++l) {
    v->vector[l] = (double *) calloc (3, sizeof (double));
    if (v->vector[l] == NULL) goto ERROR;
  }

  return v;

 ERROR:
  mpd_decomp_vector_destroy (v);
  return NULL;
}    

  
void mpd_decomp_vector_destroy (mpd_decomp_vector_t *v)
{
  if (v == NULL) return;
  if (v->vector != NULL) {
    unsigned int l;
    for (l=0; l < v->L; ++l) {
      if (v->vector[l] != NULL) free (v->vector[l]);
    }
    free (v->vector);
  }
  free (v);
}

/* Do fit for all vectors.
 * Note that we have hard coded a number of things in like randomly picking
 * the starting values between [-1,1] and doing this 3 times at each level
 * before giving up.  If you don't like this, write your own driver!
 * This routine serves as a fine example for writing your own.
 */
int mpd_decomp_full_fit (unsigned int L, double *alm, mpd_decomp_vector_t *mpd_v)
{
  mpd_decomp_t *mpd;
  double *a1m, v[3];
  unsigned int m, l, try;
  int status=0;

  if (mpd_v->L < L) return GSL_EINVAL;

  mpd = NULL;
  a1m = NULL;

  a1m = (double *)malloc ((2*L+1)*sizeof (double));
  if (a1m == NULL) {
    status = GSL_ENOMEM;
    goto DONE;
  }

  srand ((unsigned int)time(NULL));
  mpd_v->norm = 1.0;

  /* Get things started by copying alm into a1m (hence the size for a1m) */
  (void) memcpy (a1m, alm, (2*L+1)*sizeof (alm[0]));

  for (l=L; l >= 2; --l) {
    double a1mmax = -1;

    /* Renormalize the alm so they fall in the interval [-1,1].  This
     * allows us to pick random values from this interval as our initial
     * guess.  We include this factor in the normalization.
     */
    for (m=0; m < 2*l+1; ++m) {
      if (fabs (a1m[m]) > a1mmax) a1mmax = fabs (a1m[m]);
    }
    mpd_v->norm *= a1mmax;
    a1mmax = 1.0 / a1mmax;
    for (m=0; m < 2*l+1; ++m) a1m[m] *= a1mmax;

    mpd = mpd_decomp_create (l, a1m);
    if (mpd == NULL) {
      status = GSL_ENOMEM;
      goto DONE;
    }

    /* Repeat our try, starting from different random points */
    status = 1;
    for (try=0; (status != 0) && (try < 3); ++try) {
      for (m=0; m < 3; ++m) v[m] = 2*(rand()/(RAND_MAX+1.0)-0.5);
      for (m=0; m < 2*l-1; ++m) a1m[m] = 2*(rand()/(RAND_MAX+1.0)-0.5);
      status = mpd_decomp_fit (mpd, a1m, v);
    }

    if (try >= 3) {
      status = GSL_CONTINUE;
      goto DONE;
    }

    /* Add vector to the list. */
    (void) memcpy (mpd_v->vector[L-l], v, 3*sizeof(v[0]));
  }

  /*
   * For the quadrupole, a1m is the other vector, in some funny form.  So
   * pull this apart, normalize it, and store it.  Note the extra factor of
   * 3/4Pi that comes from Y1m.
   */
  {
    double norm = 0;

    v[0] = -sqrt(2.0) * a1m[1];
    v[1] = sqrt(2.0) * a1m[2];
    v[2] = a1m[0];
    for (m=0; m < 3; ++m) norm += v[m]*v[m];
    norm = sqrt (norm);
    for (m=0; m < 3; ++m) mpd_v->vector[L-1][m] = v[m] / norm;
    mpd_v->norm *= norm * (0.25*3.0/M_PI);
  }

 DONE:
  if (a1m != NULL) free (a1m);
  if (mpd != NULL) mpd_decomp_destroy (mpd);

  return status;
}

/* Internal routines: */

/* Pull the parameters out of the vector x. */
static void internal_set_mpd_params (mpd_decomp_t *mpd, const gsl_vector *x)
{
  unsigned int offset, m;
  /* Though this seems silly, it explicitly shows the order with which we
   * store the coefficients and makes writting the formulae easier.  We do
   * NOT set all the unused elements to zero.  This is done elsewhere (or
   * it better be) or it doesn't matter.
   */
  mpd->alminus1re[0] = gsl_vector_get (x, 0);
  for (m=1; m <= mpd->L-1; ++m) {
    mpd->alminus1re[m] = gsl_vector_get (x, 2*m-1);
    mpd->alminus1im[m] = gsl_vector_get (x, 2*m);
  }

  offset = 2*mpd->L-1;
  mpd->bre[0] = gsl_vector_get (x, 0+offset);
  for (m=1; m <= mpd->L-2; ++m) {
    mpd->bre[m] = gsl_vector_get (x, 2*m-1+offset);
    mpd->bim[m] = gsl_vector_get (x, 2*m+offset);
  }

  offset += 2*mpd->L-3;
  for (m=0; m < 3; ++m) mpd->v[m] = gsl_vector_get (x, m+offset);
}

#if 0 /* Not needed!! */
static int internal_mpd_decomp_fdf (const gsl_vector *x, void *params, 
			     gsl_vector *f, gsl_matrix *J)
{
  int status;

  status = internal_mpd_decomp_f (x, params, f);
  if (status != GSL_SUCCESS) return status;

  status = internal_mpd_decomp_df (x, params, J);
  return status;
}
#endif

static int internal_mpd_decomp_f (const gsl_vector *x, void *params, gsl_vector *f)
{
  mpd_decomp_t *mpd = (mpd_decomp_t *)params;
  double val; /* Used for computations. */
  double *v, *am1re, *am1im, *bre, *bim; /* Shorthand to save some typing */
  unsigned int L, m, offset;

  internal_set_mpd_params (mpd, x);
  v = mpd->v;
  am1re = mpd->alminus1re;
  am1im = mpd->alminus1im;
  bre = mpd->bre;
  bim = mpd->bim;
  L = mpd->L;
  
  /* Treat m=0 separately.  Here we ONLY have an equation for the real part
     (as al0 is purely real). */
  val = internal_C0(L,0)*v[2]*am1re[0]
    + sqrt(2.0)*internal_Cp1(L,0)*(am1re[1]*v[0] - am1im[1]*v[1]);
  gsl_vector_set (f, 0, val-mpd->alm[0]);
  
  /* The am1re, am1im are set up to handle going up to L! */
  for (m=1; m <= L; ++m) {
    /* Re(alm) */
    val = internal_C0(L,m)*v[2]*am1re[m]
      - internal_Cp1(L,m)/sqrt(2.0)*(v[0]*am1re[m-1] + v[1]*am1im[m-1])
      + internal_Cm1(L,m)/sqrt(2.0)*(v[0]*am1re[m+1] - v[1]*am1im[m+1]);
    gsl_vector_set (f, 2*m-1, val - mpd->alm[2*m-1]);
    /* Im(alm) */
    val = internal_C0(L,m)*v[2]*am1im[m]
      - internal_Cp1(L,m)/sqrt(2.0)*(v[0]*am1im[m-1] - v[1]*am1re[m-1])
      + internal_Cm1(L,m)/sqrt(2.0)*(v[0]*am1im[m+1] + v[1]*am1re[m+1]);
    gsl_vector_set (f, 2*m, val - mpd->alm[2*m]);
  }

  /* Internal parameters */
  offset = 2*L+1;

  /* Treat m=0 separately.  Here we ONLY have an equation for the real part
     (as b0 is purely real). */
  val = internal_D0(L,0)*v[2]*am1re[0]
    + sqrt(2.0)*internal_Dp1(L,0)*(am1re[1]*v[0] - am1im[1]*v[1]);
  gsl_vector_set (f, 0+offset, val - bre[0]);
  
  /* The L-2 pieces. */
  for (m=1; m <= L-2; ++m) {
    /* Re(bm) */
    val = internal_D0(L,m)*v[2]*am1re[m]
      - internal_Dp1(L,m)/sqrt(2.0)*(v[0]*am1re[m-1] + v[1]*am1im[m-1])
      + internal_Dm1(L,m)/sqrt(2.0)*(v[0]*am1re[m+1] - v[1]*am1im[m+1]);
    gsl_vector_set (f, 2*m-1+offset, val - bre[m]);
    /* Im(bm) */
    val = internal_D0(L,m)*v[2]*am1im[m]
      - internal_Dp1(L,m)/sqrt(2.0)*(v[0]*am1im[m-1] - v[1]*am1re[m-1])
      + internal_Dm1(L,m)/sqrt(2.0)*(v[0]*am1im[m+1] + v[1]*am1re[m+1]);
    gsl_vector_set (f, 2*m+offset, val - bim[m]);
  }


  /* Normalization condition */
  offset += 2*L-3;
  gsl_vector_set (f, offset, v[0]*v[0]+v[1]*v[1]+v[2]*v[2] - 1);

  return GSL_SUCCESS;
}

#if 0 /* Not needed */
static int internal_mpd_decomp_df (const gsl_vector *x, void *params, gsl_matrix *J)
{
  return GSL_FAILURE;
}
#endif


/*
 * Coefficients from the YLM = Y1a Y(L-1)m decomposition.
 */
static double internal_C0 (unsigned int L, int M)
{
  const double norm = sqrt (3.0/(4.0*M_PI));
  double val;

  if (abs (M) >= L) return 0;

  val = norm * sqrt ((L-M)*(L+M) / ((2*L-1.0)*(2*L+1.0)));
  return val;
}

static double internal_Cp1 (unsigned int L, int M)
{
  const double norm = sqrt (3.0/(8.0*M_PI));
  double val;

  if (abs (M) > L) return 0;

  val = norm * sqrt ((L+M-1.0)*(L+M) / ((2*L-1.0)*(2*L+1.0)));
  return val;
}

static double internal_Cm1 (unsigned int L, int M)
{
  return internal_Cp1 (L, -M);
}

/*
 * Coefficients from the YL-2,M piece of decomposition.  Normalized as above.
 */
static double internal_D0 (unsigned int L, int M)
{
  const double norm = sqrt (3.0/(4.0*M_PI));
  double val;

  if (abs (M) > L-2) return 0;

  val = norm * sqrt ((L-M-1.0)*(L+M-1.0) / ((2*L-3.0)*(2*L-1.0)));
  return val;
}

static double internal_Dp1 (unsigned int L, int M)
{
  const double norm = -sqrt (3.0/(8.0*M_PI));
  double val;

  if (abs (M) > L-2) return 0;

  val = norm * sqrt ((L-M-1.0)*(L-M) / ((2*L-3.0)*(2*L-1.0)));
  return val;
}

static double internal_Dm1 (unsigned int L, int M)
{
  return internal_Dp1 (L, -M);
}

#ifdef MPD_DEBUG
static void internal_print_state (unsigned int iter, mpd_decomp_t *mpd,
				  gsl_multiroot_fsolver *s)
{
  unsigned int m, offset;
  unsigned int N = 4*mpd->L-1;

  printf ("\nMPD_DECOMP version %s\n", MPD_VERSION);
  printf ("iter = %3u:\n", iter);
  printf ("\t 0 (%g, 0)\n", gsl_vector_get (s->x, 0));
  for (m=1; m < mpd->L; ++m)
    printf ("\t%2d (%g, %g)\n", m, gsl_vector_get (s->x, 2*m-1),
	    gsl_vector_get (s->x, 2*m));
  printf ("\tbm\n");
  offset = 2*mpd->L-1;
  printf ("\t 0 (%g, 0)\n", gsl_vector_get (s->x, 0+offset));
  for (m=1; m < mpd->L-1; ++m)
    printf ("\t%2d (%g, %g)\n", m, gsl_vector_get (s->x, 2*m-1+offset),
	    gsl_vector_get (s->x, 2*m+offset));
  printf ("\t v = (%g, %g, %g)\n",
	  gsl_vector_get (s->x, N-3), gsl_vector_get (s->x, N-2),
	  gsl_vector_get (s->x, N-1));
  printf ("\tf: ");
  for (m=0; m < 4*mpd->L-1; ++m) printf (" %g", gsl_vector_get (s->f, m));
  printf ("\n");
}
#endif
