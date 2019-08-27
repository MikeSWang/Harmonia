"""
Fourier bases (:mod:`~harmonia.algorithms.bases`)
===============================================================================

Evaluate Fourier basis functions.

.. topic:: Important note

    Function evaluation relies on :func:`scipy.special.spherical_jn`,
    :func:`scipy.special.sph_harm` and :func:`mpmath.besseljzero`.  Note the
    types and ordering of arguments may differ here.

"""
import numpy as np
from scipy.special import spherical_jn, sph_harm
from mpmath import besseljzero

from harmonia.collections.utils import bisect_roots


def sph_harmonic(ell, m, theta, phi):
    """Spherical harmonic function.

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

    """
    return sph_harm(m, ell, phi, theta)


def sph_besselj(ell, x, deriv=False):
    """Spherical Bessel function of the first kind or its derivative.

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
    r"""Compute up to a maximum number of positive zeros of spherical Bessel
    function of the first kind or its derivative.

    The determination of roots of the spherical Bessel function:math:`j_\nu(x)`
    relies on the identity

    .. math:: j_\nu(x) = \sqrt{\frac{\pi}{2x}} J_{\nu+1/2}(x)

    where :math:`J_\nu(x)` is the Bessel funcion of the first kind.  For roots
    of the derivative function, a built-in bisection root finder is employed.

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
            u_ell = np.array(
                [float(besseljzero(ell+1/2, n, derivative=0))
                 for n in range(1, nmax+1)]
                )
    else:
        def _deriv_func(x):
            return sph_besselj(ell, x, deriv=True)

        # Empirically set root search range [`ell`+1, `nmax`*max{4, `ell`}].
        u_ell = bisect_roots(_deriv_func, ell+1, nmax*max(4, ell), maxnum=nmax)
        if only:
            u_ell = u_ell[-1]

    return u_ell
