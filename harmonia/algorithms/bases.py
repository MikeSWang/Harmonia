"""
Fourier bases (:mod:`~harmonia.algorithms.bases`)
===============================================================================

Evaluations and solutions of Fourier basis functions.

Function evaluation relies on :func:`scipy.special.spherical_jn`,
:func:`scipy.special.sph_harm` and :func:`mpmath.besseljzero`.  Note the data
type and ordering of arguments may differ in implementation.

**Spherical basis**

.. autosummary::

    spherical_harmonic
    spherical_besselj
    spherical_besselj_root

|

"""
import numpy as np
from scipy.special import spherical_jn, sph_harm
from mpmath import besseljzero

from harmonia.collections.utils import bisect_roots


def spherical_harmonic(ell, m, theta, phi):
    r"""Spherical harmonic function.

    Parameters
    ----------
    ell : int, array_like
        Degree :math:`\ell \geqslant 0` of the spherical harmonic function.
    m : int, array_like
        Order :math:`-\ell \leqslant m \leqslant \ell` of the spherical
        harmonic function.
    theta : float, array_like
        Polar angle :math:`\theta \in [0, \pi]`.
    phi: float, array_like
        Azimuthal angle :math:`\phi \in [0, 2\pi]`.

    Returns
    -------
    complex, array_like
        Function value :math:`Y_{\ell m}(\theta, \phi)`.

    """
    return sph_harm(m, ell, phi, theta)


def spherical_besselj(ell, x, derivative=False):
    r"""Spherical Bessel function of the first kind or its derivative.

    Parameters
    ----------
    ell : int, array_like
        Degree :math:`\ell \geqslant 0` of the spherical harmonic function.
    x : float, array_like
        Positive argument :math:`x > 0`.
    derivative : bool, optional
        If `True` (default is `False`), evaluate the derivative instead.

    Returns
    -------
    float, array_like
        Function value.

    """
    return spherical_jn(ell, x, derivative=derivative)


def spherical_besselj_root(ell, nmax, only=True, derivative=False):
    r"""Compute positive zeros, up to a maximum number, of spherical Bessel
    functions of the first kind or their derivatives.

    Solving for roots of the spherical Bessel function :math:`j_\nu(x)` relies
    on the identity

    .. math:: j_\nu(x) = \sqrt{\frac{\pi}{2x}} J_{\nu+1/2}(x)

    where :math:`J_\nu(x)` is the Bessel funcion of the first kind.  For roots
    of the derivative function :math:`j'_\nu(x)`, the interval bisection method
    is employed.

    Parameters
    ----------
    ell : int
        Degree :math:`\ell \geqslant 0` of the spherical harmonic function.
    nmax : int
        Maximum number of positive zeros to be found (``nmax >= 1``).
    only : bool, optional
        If `True` (default), return the maximum-number root only.
    derivative : bool, optional
        If `True` (default is `False`), compute the root of the derivative
        function instead.

    Returns
    -------
    u_ell : float, array_like
        Positive zero(s).

    """
    if not derivative:
        if only:
            u_ell = float(besseljzero(ell+1/2, nmax, derivative=0))
        else:
            u_ell = np.array(
                [
                    float(besseljzero(ell+1/2, n, derivative=0))
                    for n in range(1, nmax+1)
                ]
            )
    else:
        # Empirically set root search range [`ell`+1, `nmax`*max{4, `ell`}].
        u_ell = bisect_roots(
            lambda x: spherical_besselj(ell, x, derivative=True),
            ell+1, nmax*max(4, ell),
            maxnum=nmax
        )
        if only:
            u_ell = u_ell[-1]

    return u_ell
