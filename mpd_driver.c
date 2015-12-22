/*
 * Copyright (c) 2003-2007 Craig J Copi
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

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include "mpd_decomp.h"


/*
 * A simple driver routine for mpd_decomp_full_fit.  This program reads in
 * 2L+1 coefficients from the command line, decomposes them into L vectors
 * and a normalization, and prints the result to stdout.  This serves as
 * an example of using mpd_decomp_full_fit.  The source code for that
 * routine is also valuable if you want to write your own routine from the
 * ground up.
 */


static void usage (const char *progname);


int main (int argc, char *argv[])
{
  mpd_decomp_vector_t *v;
  double *alm, tmp;
  unsigned int m, l, L;
  int status;

  tmp = 0.5*argc-1;
  L = (unsigned int)tmp;
  if (((unsigned int)(tmp+0.6) != L) || (L < 2)) {
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

  printf ("Normalization %g\n", v->norm);
  for (l=0; l < L; ++l) {
    for (m=0; m < 3; ++m) printf ("%g ", v->vector[l][m]);
    printf ("\n");
  }

  /* Clean up memory */
  mpd_decomp_vector_destroy (v);
  free (alm);

  exit (EXIT_SUCCESS);
}


static void usage (const char *progname)
{
  fprintf (stderr, "Usage: %s al0 al1re al1im al2re al2im al3re al3im [...]\n",
	   progname);
}
