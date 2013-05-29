import numpy as np
import scipy.optimize as opt
from numpy import pi, sqrt # Shorthand

# Number of tries to solve the system of equations
Ntries = 5

# Coefficients needed in the system equations to be solved.  These are
# intended for internal use only.
def _C0 (l, m) :
    return sqrt(3*(l-m)*(l+m) / (4*pi*(2*l-1.)*(2*l+1.)))
def _cp1 (l, m) :
    if abs(m) > l : return 0
    else : return sqrt(3*(l+m-1.)*(l+m) / (8*pi*(2*l-1.)*(2*l+1.)))
_Cp1 = np.vectorize(_cp1)
def _Cm1 (l, m) :
    return _Cp1(l,-m)

def _D0 (l, m) :
    return sqrt(3*(l-m-1.)*(l+m-1.)/(4*pi*(2*l-3.)*(2*l-1.)))
def _dp1 (l, m) :
    if abs(m) > l-2 : return 0
    else : return - sqrt(3*(l-m-1.)*(l-m)/(8*pi*(2*l-3.)*(2*l-1.)))
_Dp1 = np.vectorize(_dp1)
def _Dm1 (l, m) :
    return _Dp1(l,-m)

def norm (v) :
    """Trivial implementation of the L2 norm."""
    return sqrt(np.sum(v**2))

def _extract_pieces (L, x) :
    # Extra space in alm set to zero to allow easier equations in mpd_equation_system
    alm = np.zeros(L+2, dtype=np.complex)
    blm = np.zeros(L-1, dtype=np.complex)
    v = np.zeros(3)
    alm[0] = x[0]
    alm.real[1:L] = x[1:2*L-1:2]
    alm.imag[1:L] = x[2:2*L:2]
    offset = 2*(L-1)+1
    blm[0] = x[offset]
    blm.real[1:] = x[offset+1:offset+2*(L-1)-1:2]
    blm.imag[1:] = x[offset+2:offset+2*(L-1):2]
    offset += 2*(L-2)+1
    v = x[offset:]
    return (alm, blm, v)

def mpd_equation_system (x, L, alm) :
    """System of coupled quadratic equations to solve for the the
    multipole vectors.  We do this by "peeling off" a vector.  Produced is
    the set of new alm for L-1, the extra coefficients blm, and the vector
    v.  These are stored in x in the format:
      x[:2*L-1] : new alm as a_{l-1,0}, a_{l-1,1} real, a_{l-1,1} imag, ...
      x[2*L-1:2*L-1+2*L-3] : blm in same order as new alm
      x[4*L-4:4*L-1] : x,y,z components of vector, v.
    The input alm are provided in the same order as the new alm."""

    (afit, bfit, v) = _extract_pieces(L, x)
    f = np.zeros_like(x)

    # al0
    f[0] = _C0(L,0)*afit[0].real*v[2] \
        + sqrt(2)*_Cp1(L,0)*(afit[1].real*v[0] - afit[1].imag*v[1]) - alm[0]
    m = np.arange(1,L+1)
    # Real part of alm
    f[1:2*L:2] = _C0(L,m)*afit[m].real*v[2] \
        + _Cm1(L,m)/sqrt(2.)*(afit[m+1].real*v[0] - afit[m+1].imag*v[1]) \
        - _Cp1(L,m)/sqrt(2.)*(afit[m-1].real*v[0] + afit[m-1].imag*v[1]) \
        - alm[1:2*L:2]
    # Imaginary part of alm
    f[2:2*L+1:2] = _C0(L,m)*afit[m].imag*v[2] \
        + _Cm1(L,m)/sqrt(2.)*(afit[m+1].imag*v[0] + afit[m+1].real*v[1]) \
        - _Cp1(L,m)/sqrt(2.)*(afit[m-1].imag*v[0] - afit[m-1].real*v[1]) \
        - alm[2:2*L+1:2]
     
    offset = 2*L+1
    # bl0
    f[offset] = _D0(L,0)*afit[0].real*v[2] \
        + sqrt(2)*_Dp1(L,0)*(afit[1].real*v[0] - afit[1].imag*v[1]) - bfit[0].real
    if L > 2 :
        m = np.arange(1,L-1)
        # Real part of alm
        f[offset+1:offset+2*(L-2):2] = _D0(L,m)*afit[m].real*v[2] \
            + _Dm1(L,m)/sqrt(2.)*(afit[m+1].real*v[0] - afit[m+1].imag*v[1]) \
            - _Dp1(L,m)/sqrt(2.)*(afit[m-1].real*v[0] + afit[m-1].imag*v[1]) \
            - bfit[1:2*(L-2)+1].real
        # Imaginary part of alm
        f[offset+2:offset+2*(L-2)+1:2] = _D0(L,m)*afit[m].imag*v[2] \
            + _Dm1(L,m)/sqrt(2.)*(afit[m+1].imag*v[0] + afit[m+1].real*v[1]) \
            - _Dp1(L,m)/sqrt(2.)*(afit[m-1].imag*v[0] - afit[m-1].real*v[1]) \
            - bfit[1:2*(L-2)+1].imag
    f[offset+2*L-3] = np.sum(v**2) - 1.
    return f

def mpd_decomp_fit (alm, almguess, vguess) :
    """Given input alm and initial guesses for the new alm and vector v a
    fit is performed.  The format for the input is
      alm: al0, al1 real, al1 imag, al2 real, al2 imag, ...
      almguess: same as alm but for l-1
      vguess: x, y, z components.
    Returned is the new alm and vector v in the same format as above.  Also
    information from scipy.optimize.fsolve are returned.  More explicitly
    the tuple
    (almnew, v, ierr, msg)
    is returned.  If ierr != 1 then msg contains the error from fsolve."""
    L = int((len(alm)-1)/2)
    x = np.zeros(4*L-1)
    x[:2*L-1] = almguess
    offset = 2*L-1
    x[offset:offset+2*L-3] = almguess[:2*L-3] # Arbitrarily set
    offset += 2*L-3
    x[offset:] = vguess
    (xnew, info, ierr, msg) = opt.fsolve (mpd_equation_system, x,
                                          args=(L,alm), xtol=1.e-12,
                                          full_output=True)
    return (xnew[:2*L-1], xnew[offset:offset+3], ierr, msg)

def mpd_decomp_full_fit (alm, v0=None) :
    """Fit for all multipole vectors for a given set of alm.
    The format for the input is
      alm: al0, al1 real, al1 imag, al2 real, al2 imag, ...
    Returned is a tuple
      (v, normalization)
    where v is a Lx3 array, each row being a multipole vector.
    On failure None is returned for v and 0 for the normalization.
    An initial guess for the vectors, v0, can be provided.  This is useful
    in MC studies where a good guess at a vector is known.  v0 MUST be l
    vectors, that is, a guess for each of the vectors.
    """
    L = int((len(alm)-1)/2)
    varr = np.zeros((L,3))
    vnorm = 1.0
    a1m = alm.copy() # initialize

    # Get all the vectors
    for l in xrange(L, 1, -1) :
        # Renormalize the a1m.  We do this since we will pick random
        # starting points for our fits.  This just makes it a little
        # easier.
        n = norm(a1m)
        a1m /= n
        vnorm *= n
        # Allow a number of tries to find a solution before giving up
        for _ in xrange(Ntries) :
            if v0 is None :
                vguess = np.random.rand(3)*2 - 1
                vguess /= norm(vguess)
            else :
                vguess = v0[l-1].copy()
            almguess = np.random.rand(2*l-1)*2 - 1
            res = mpd_decomp_fit (a1m, almguess, vguess)
            if res[2] == 1 : break
        if res[2] != 1 : # Failed to find a solution
            return (None, 0)
        a1m = res[0].copy()
        varr[l-1] = res[1].copy()
    
    # For the quadrupole the a1m is now ther other vector stored in a funny
    # format.  Pull this apart, normalize it, and store it.  Note the
    # factor (3/4pi)^(L/2) comes from the Y1m factors that were peeled off
    # during the decomposition.
    varr[0,0] = -sqrt(2) * a1m[1]
    varr[0,1] =  sqrt(2) * a1m[2]
    varr[0,2] =  a1m[0]
    n = norm(varr[0])
    varr[0] /= n
    vnorm *= n * sqrt((0.75/pi)**L)
    return (varr, vnorm)
