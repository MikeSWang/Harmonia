"""
Fourier bases (:mod:`~harmonia.algorithms.bases`)
===========================================================================

Evaluate Fourier basis functions and their derivatives, and determine their
zeros.

**Spherical basis functions**

.. autosummary::

    spherical_harmonic
    spherical_besselj
    spherical_besselj_root

|

"""
from __future__ import division

import numpy as np
from mpmath import besseljzero
from scipy.special import sph_harm, spherical_jn

from harmonia.collections.utils import binary_search


def spherical_harmonic(ell, m, theta, phi):
    r"""Evaluate the spherical harmonic function
    :math:`Y_{\ell m}(\theta, \phi)` of degree :math:`\ell \geqslant 0` and
    order :math:`\vert{m}\vert \leqslant \ell` at the polar angle
    :math:`\theta \in [0, \pi]` and the azimuthal angle :math:`\phi \in
    [0, 2\pi]`.

    Parameters
    ----------
    ell : int, array_like
        Degree of the spherical harmonic function, ``ell >= 0``.
    m : int, array_like
        Order of the spherical harmonic function, ``|m| <= ell``.
    theta : float, array_like
        Polar angle in the interval ``[0, pi]``.
    phi: float, array_like
        Azimuthal angle in the interval ``[0, 2*pi]``.

    Returns
    -------
    complex, array_like
        :math:`Y_{\ell m}` value at `theta` and `phi`.

    """
    return sph_harm(m, ell, phi, theta)


def spherical_besselj(ell, x, derivative=False):
    r"""Evaluate the spherical Bessel function of the first kind
    :math:`j_{\ell}(x)` of order :math:`\ell \geqslant 0` or its derivative
    :math:`j'_{\ell}(x)`, .

    Parameters
    ----------
    ell : int, array_like
        Order of the spherical harmonic function, ``ell >= 0``.
    x : float, array_like
        Function argument.
    derivative : bool, optional
        If `True` (default is `False`), evaluate the derivative function
        instead.

    Returns
    -------
    float, array_like
        :math:`j_{\ell}` or :math:`j'_{\ell}` value at `x`.

    """
    return spherical_jn(ell, x, derivative=derivative)


def spherical_besselj_root(ell, nmax, only=True, derivative=False):
    r"""Compute positive zeros :math:`u_{\ell n}` of the spherical Bessel
    function of the first kind of order :math:`\ell`, :math:`j_{\ell}`, or
    its derivative function, up to a maximal number :math:`n_\textrm{max}`.

    Solving for roots of the spherical Bessel function relies on the
    identity :math:`j_\ell(x) = \sqrt{\pi/(2x)} J_{\ell + 1/2}(x)`, where
    :math:`J_\ell(x)` is the Bessel funcion of the first kind.  Solving for
    roots of the derivative function employs the interval bisection method,
    with the initial interval ansatz :math:`\ell + 1 \leqslant x \leqslant
    n_\textrm{max} \operatorname{max}\{4, \ell\}`.

    Parameters
    ----------
    ell : int
        Order of the spherical Bessel function, ``ell >= 0``.
    nmax : int
        Maximal number of positive zeros to be found, ``nmax >= 1``.
    only : bool, optional
        If `True` (default), return the maximal root only.
    derivative : bool, optional
        If `True` (default is `False`), compute the zero(s) of the
        derivative function instead.

    Returns
    -------
    u_ell : float, array_like
        Positive zero(s) for order `ell` (in ascending order).

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
        u_ell = binary_search(
            lambda x: spherical_besselj(ell, x, derivative=True),
            ell+1, nmax * max(4, ell), maxnum=nmax
        )
        if only:
            u_ell = u_ell[-1]

    return u_ell
