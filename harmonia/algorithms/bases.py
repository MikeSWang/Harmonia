"""
Fourier bases (:mod:`~harmonia.algorithms.bases`)
===============================================================================

Evaluate Fourier basis functions.

"""
import numpy as np
from mpmath import besseljzero
from scipy.special import spherical_jn, sph_harm

from harmonia.collections.utils import bisect_roots


def sph_besselj(ell, x, deriv=False):
    """Spherical Bessel function of the first kind or its derivative.

    This function relies on :func:`scipy.special.spherical_jn`.

    Parameters
    ----------
    ell : int, array_like
        Order of the spherical Bessel function (``ell >= 0``).
    x : float, array_like
        Positive argument (``x > 0``).
    deriv : bool, optional
        If `True` (default is `False`), evaluate the derivative instead.

    Returns
    -------
    float, array_like
        Function value.

    """
    return spherical_jn(ell, x, derivative=deriv)


def sph_besselj_root(ell, nmax, only=True, deriv=False):
    r"""Compute up to some number of positive zero of spherical Bessel
    function of the first kind or its derivative.

    The determination of roots of the spherical Bessel function:math:`j_\nu(x)`
    relies on the identity

    .. math:: j_\nu(x) = \sqrt{\frac{\pi}{2x}} J_{\nu+1/2}(x)

    where :math:`J_\nu(x)` is the Bessel funcion of the first kind.  For roots
    of the derivative function, a built-in bisection root finder is employed.

    This function relies on :func:`mpmath.besseljzero`.

    Parameters
    ----------
    ell : int
        Order of the spherical Bessel function (``ell >= 0``).
    nmax : int
        Maximum number of the zero (``nmax >= 1``).
    only : bool, optional
        If `True` (default), return the `nmax`-th root only.
    deriv : bool, optional
        If `True` (default is `False`), compute the root of its derivative
        instead.

    Returns
    -------
    u_ell : float, array_like
        Positive zero(s).

    """
    if not deriv:
        if only:
            u_ell = float(besseljzero(ell+1/2, nmax, derivative=0))
        else:
            u_ell = []
            for n in range(1, nmax+1):
                u_ell = np.append(
                    u_ell, float(besseljzero(ell+1/2, n, derivative=0))
                    )
    else:
        # Define auxiliary function.
        def _deriv(x):
            return sph_besselj(ell, x, deriv=True)

        # Empirically set root range in [`ell`+1, max{4, `ell`}].
        u_ell = bisect_roots(_deriv, ell+1, nmax*max(4, ell), maxnum=nmax)

        if only:
            u_ell = u_ell[-1]

    return u_ell


def sph_harmonic(ell, m, theta, phi):
    """Spherical harmonic function.

    This function relies on :func:`scipy.special.sph_harm`.

    Parameters
    ----------
    ell : int, array_like
        Degree of the spherical harmonic function (``ell >= 0``).
    m : int, array_like
        Order of the spherical harmonic function (``-ell <= m <= ell``).
    theta : float, array_like
        Polar angle (``0 <= theta <= np.pi``).
    phi: float, array_like
        Azimuthal angle (``0 <= phi < 2*np.pi``).

    Returns
    -------
    complex, array_like
        Function value.

    Warnings
    --------
    The ordering of the arguments differ from that of
    :func:`scipy.special.sph_harm`.

    """
    return sph_harm(m, ell, phi, theta)
