#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include "mpd_decomp.h"


static char const rcsid[]
  = "$Id$";


/*
 * Read in 2L+1 coefficients from the command line.  Returns a vector and
 * the remaining 2L-1 coefficients.
 */


static void usage (const char *progname);


int main (int argc, char *argv[])
{
  mpd_decomp_vector_t *v;
  double *alm, tmp;
  int status, m, l, L;

  tmp = 0.5*argc-1;
  L = (int)tmp;
  if (((int)(tmp+0.6) != L) || (L < 2)) {
    usage (argv[0]);
    exit (EXIT_FAILURE);
  }

  alm = (double *)malloc ((2*L+1)*sizeof (double));
  v = mpd_decomp_vector_create (L);
  if ((alm == NULL) || (v == NULL)) {
    fprintf (stderr, "Error allocating memory\n");
    exit (EXIT_FAILURE);
  }

  for (m=0; m < 2*L+1; ++m) alm[m] = atof (argv[m+1]);

  status = mpd_decomp_full_fit (L, alm, v);
  printf ("status = %s\n", gsl_strerror (status));
  if (status != 0) {
    fprintf (stderr, "Error fitting\n");
    exit (EXIT_FAILURE);
  }

  for (l=0; l < L; ++l) {
    for (m=0; m < 3; ++m) printf ("%g ", v->vector[l][m]);
    printf ("\n");
  }

  /* Clean up memory */
  mpd_decomp_vector_destroy (v);
  mpd_decomp_destroy (mpd);

  exit (EXIT_SUCCESS);
}


void usage (const char *progname)
{
  fprintf (stderr, "Usage: %s al0 a1mre a1mim a2mre a2mim a3mre a3mim [...]\n",
	   progname);
}
