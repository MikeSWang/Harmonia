"""
Fourier bases (:mod:`~harmonia.algorithms.bases`)
===========================================================================

Compute quantities related to Fourier basis functions.

.. autosummary::

    spherical_harmonic
    spherical_besselj
    spherical_besselj_root

|

"""
# pylint: disable=no-name-in-module
import numpy as np
from mpmath import besseljzero
from scipy.special import sph_harm, spherical_jn

from harmonia.utils import binary_search


def spherical_harmonic(ell, m, theta, phi, conj=False):
    r"""Evaluate the spherical harmonic function.

    This returns :math:`Y_{\ell m}(\theta, \phi)` of degree :math:`\ell
    \geqslant 0` and order :math:`\vert {m} \vert \leqslant \ell` at the
    polar angle :math:`\theta \in [0, \pi]` and the azimuthal angle
    :math:`\phi \in [0, 2\pi]`.

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
    conj : bool, optional
        If `True` (default is `False`), return the complex conjugate.

    Returns
    -------
    complex, array_like
        :math:`Y_{\ell m}` value at `theta` and `phi`.

    """
    return np.conj(sph_harm(m, ell, phi, theta)) if conj \
        else sph_harm(m, ell, phi, theta)


def spherical_besselj(ell, x, derivative=False):
    r"""Evaluate the spherical Bessel function of the first kind or
    its derivative.

    This returns :math:`j_{\ell}(x)` or :math:`j'_{\ell}(x)` of order
    :math:`\ell \geqslant 0`.

    Parameters
    ----------
    ell : int, array_like
        Order of the spherical harmonic function, ``ell >= 0``.
    x : float, array_like
        Argument of the spherical Bessel function.
    derivative : bool, optional
        If `True` (default is `False`), evaluate the derivative instead.

    Returns
    -------
    float, array_like
        :math:`j_{\ell}` or :math:`j'_{\ell}` value at `x`.

    """
    return spherical_jn(ell, x, derivative=derivative)


def spherical_besselj_root(ell, nmax, only=True, derivative=False):
    r"""Find positive zero(s) :math:`u_{\ell n}` of the spherical Bessel
    function :math:`j_{\ell}` of the first kind of order :math:`\ell`, or
    its derivative :math:`j'_{\ell}`, up to a maximum number
    :math:`n_\textrm{max}`.

    Solving for roots of the spherical Bessel function relies on the
    identity :math:`j_\ell(x) = \sqrt{\pi/(2x)} J_{\ell + 1/2}(x)`, where
    :math:`J_\ell(x)` is the Bessel funcion of the first kind.  Solving for
    roots of the derivative function employs the interval bisection method,
    with the interval ansatz :math:`\ell + 1 \leqslant x \leqslant
    n_\textrm{max} \operatorname{max}\{4, \ell\}`.

    Parameters
    ----------
    ell : int
        Order of the spherical Bessel function, ``ell >= 0``.
    nmax : int
        Maximum number of positive zeros to be found, ``nmax >= 1``.
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
        # `mpmath` returns a ``mpf`` float which needs conversion.
        if only:
            u_ell = float(besseljzero(ell + 0.5, nmax, derivative=0))
        else:
            u_ell = np.asarray([
                float(besseljzero(ell + 0.5, n, derivative=0))
                for n in range(1, nmax + 1)
            ])
    else:
        u_elln_list = binary_search(
            lambda x: spherical_besselj(ell, x, derivative=True),
            ell + 1, nmax * max(4, ell),
            maxnum=nmax
        )
        u_ell = u_elln_list[-1] if only else u_elln_list

    return u_ell
