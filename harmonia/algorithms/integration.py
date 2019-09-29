"""
Numerical integration (:mod:`~harmonia.algorithms.integration`)
===============================================================================

Numerically integrate against Fourier basis functions.

.. topic:: Caution

    Quadrature integration of spherical Bessel functions may converge slowly.

**Generic spherical integrals**

.. autosummary::

    angular_spherical_integral
    radial_spherical_integral

**Linear kernel spherical integrals**

.. autosummary::

    angular_harmonic_integral
    radial_besselj_integral

|

"""
import numpy as np
from scipy.integrate import dblquad, quad

from .bases import spherical_besselj, spherical_harmonic


def _angular_integrand(phi, theta, afunc, complex_part):
    """Complex angular integrand with Jacobian.

    Note
    ----
    Angular arguments are in reverse order for outward double integration.

    Parameters
    ----------
    phi, theta: float, array_like
        Angular coordinates in radians.
    afunc : callable
        Angular function to be integrated.
    complex_part : {'real', 'imag'}
        Real or imaginary part.

    Returns
    -------
    float, array_like
        Real or imaginary part of the complex angular integrand value.

    Raises
    ------
    ValueError
        If `complex_part` is neither ``'real'`` nor ``'imag'``.

    """
    if complex_part.lower() == 'real':
        return np.abs(np.sin(theta)) * afunc(theta, phi).real
    if complex_part.lower() == 'imag':
        return np.abs(np.sin(theta)) * afunc(theta, phi).imag

    raise ValueError("`complex_part` neither 'real' nor 'imag'. ")


def _radial_integrand(r, rfunc):
    """Radial integrand with Jacobian.

    Parameters
    ----------
    r: float, array_like
        Radial coordinate.
    rfunc : callable
        Radial function to be integrated.

    Returns
    -------
    float, array_like
        Radial integrand value.

    """
    return r**2 * rfunc(r)


def angular_spherical_integral(angular_func):
    r"""Full angular integral.

    Note
    ----
    Arguments of `angular_func` must be in radians in the following order and
    range: :math:`(\theta, \phi) \in [0, \pi] \times [0, 2\pi]`.

    Parameters
    ----------
    angular_func : callable
        Angular function be integrated.

    Returns
    -------
    complex
        Angular integral value.

    """
    theta_range = (0, np.pi)
    phi_range = (0, 2*np.pi)

    integral_real, _ = dblquad(
        _angular_integrand,
        *theta_range,
        *phi_range,
        args=(angular_func, 'real'),
    )
    integral_imag, _ = dblquad(
        _angular_integrand,
        *theta_range,
        *phi_range,
        args=(angular_func, 'imag'),
    )

    return integral_real + 1j*integral_imag


def radial_spherical_integral(radial_func, rmax):
    """Radial integral up to the given maximal radius.

    Parameters
    ----------
    radial_func : callable
        Radial function to be integrated.
    rmax : float
        Upper radial limit.

    Returns
    -------
    integral : float
        Radial integral value.

    """
    integral, _ = quad(_radial_integrand, 0, rmax, args=(radial_func,))

    return integral


def angular_harmonic_integral(angular_func, ell, m, *args, conjugate=True,
                              **kwargs):
    r"""Full angular integral against spherical harmonic functions.

    Note
    ----
    Arguments of `angular_func` must be in radians in the following order and
    range: :math:`(\theta, \phi) \in [0, \pi] \times [0, 2\pi]`.

    Parameters
    ----------
    angular_func : callable
        Angular function be integrated.
    ell : int
        Degree of the spherical harmonic function.
    m : int
        Order of the spherical harmonic function.
    conjugate : bool, optional
        If `True` (default), use conjugate of the spherical harmonic function.
    *args, **kwargs
        Additional positional and keyword arguments to be passed to
        `angular_func`.

    Returns
    -------
    complex
        Integral value.

    """
    def _int_kernel(theta, phi):
        return angular_func(theta, phi, *args, **kwargs) \
            * spherical_harmonic(ell, m, theta, phi)

    if conjugate:
        return np.conj(angular_spherical_integral(_int_kernel))
    return angular_spherical_integral(_int_kernel)


def radial_besselj_integral(radial_func, ell, k, rmax, *args, **kwargs):
    """Radial integral against spherical Bessel functions at the specified wave
    number up to the given maximal radius.

    Parameters
    ----------
    radial_func : callable
        Radial function to be integrated.
    ell : int
        Order of the spherical Bessel function.
    k : float
        Wave number.
    rmax : float
        Upper radial limit.
    *args, **kwargs
        Additional positional and keyword arguments to be passed to
        `radial_func`.

    Returns
    -------
    float
        Integral value.

    """
    return radial_spherical_integral(
        lambda r: radial_func(r, *args, **kwargs) \
            * spherical_besselj(ell, k*r),
        rmax,
    )
