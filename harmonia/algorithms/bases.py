"""
Fourier bases (:mod:`~harmonia.algorithms.bases`)
===============================================================================

Evaluations related to Fourier basis functions.

Function evaluation relies on :func:`scipy.special.sph_harm`,
:func:`scipy.special.spherical_jn` and :func:`mpmath.besseljzero`.  Note the
argument ordering and data type may differ in this implementation.

**Spherical basis**

.. autosummary::

    spherical_harmonic
    spherical_besselj
    spherical_besselj_root

|

"""
import numpy as np
from mpmath import besseljzero
from scipy.special import spherical_jn, sph_harm

from harmonia.collections.utils import binary_search


def spherical_harmonic(ell, m, theta, phi):
    r"""Spherical harmonic function :math:`Y_{\ell m}(\theta, \phi)`.

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
        Function value.

    """
    return sph_harm(m, ell, phi, theta)


def spherical_besselj(ell, x, derivative=False):
    r"""Spherical Bessel function of the first kind :math:`j_{\ell}(x)`, or its
    derivative  :math:`j'_{\ell}(x)`.

    Parameters
    ----------
    ell : int, array_like
        Degree :math:`\ell \geqslant 0` of the spherical harmonic function.
    x : float, array_like
        Non-negative argument :math:`x \geqslant 0`.
    derivative : bool, optional
        If `True` (default is `False`), evaluate the derivative instead.

    Returns
    -------
    float, array_like
        Function value.

    """
    return spherical_jn(ell, x, derivative=derivative)


def spherical_besselj_root(ell, n, only=True, derivative=False):
    r"""Compute positive zeros :math:`u_{\ell n}`, up to a maximum number
    :math:`n`, of spherical Bessel functions of the first kind or their
    derivatives.

    Solving for roots of the spherical Bessel function :math:`j_\ell(x)` relies
    on the identity

    .. math:: j_\ell(x) = \sqrt{\frac{\pi}{2x}} J_{\ell + 1/2}(x)

    where :math:`J_\ell(x)` is the Bessel funcion of the first kind.

    Solving for roots of the derivative function :math:`j'_\ell(x)` employs the
    bisection method, with the initial interval ansatz :math:`\ell + 1
    \leqslant x \leqslant n \cdot \mathrm{max}\{4, \ell\}`.

    Parameters
    ----------
    ell : int
        Degree :math:`\ell \geqslant 0` of the spherical harmonic function.
    n : int
        Maximum number of positive zeros to be found.
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
            u_ell = float(besseljzero(ell+1/2, n, derivative=0))
        else:
            u_ell = np.array(
                [
                    float(besseljzero(ell+1/2, i, derivative=0))
                    for i in range(1, n+1)
                ]
            )
    else:
        u_ell = binary_search(
            lambda x: spherical_besselj(ell, x, derivative=True),
            ell + 1,
            n * max(4, ell),
            maxnum=n,
        )
        if only:
            u_ell = u_ell[-1]

    return u_ell
