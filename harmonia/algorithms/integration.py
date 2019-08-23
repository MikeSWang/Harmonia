"""
Numerical integration (:mod:`~harmonia.algorithms.integration`)
===============================================================================

Numerical integration against Fourier basis functions using `scipy.integrate`.

**Generic spherical integrals**

.. autosummary::

    radial_spherical_int
    angular_spherical_int

**Linear kernel spherical integrals**

.. autosummary::

    radial_sphint_besselj
    angular_sphint_harmonic

.. todo::

    Cautious use of quadrature integration for spherical Bessel functions.

"""
import numpy as np
from scipy.integrate import dblquad, quad

from .bases import sph_besselj, sph_harmonic


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


def _angular_integrand(phi, theta, afunc, complex_part):
    """Complex angular integrand with Jacobian.

    Parameters
    ----------
    phi, theta: float, array_like
        Angular coordinates in radians.
    afunc : callable
        Angular function to be integrated.
    complex_part : {'real', 'imag'}
        Real or imaginary part of the complex value.

    Returns
    -------
    float, array_like
        Complex part of the angular integrand value.

    Raises
    ------
    ValueError
        If `complex_part` is neither ``'real'`` nor ``'imag'``.

    Notes
    -----
    Angular coordinates are in reverse order in arguments for outward double
    integration.

    """
    jacobian = np.abs(np.sin(theta))

    if complex_part.lower() == 'real':
        return jacobian * afunc(theta, phi).real
    if complex_part.lower() == 'imag':
        return jacobian * afunc(theta, phi).imag
    raise ValueError("`complex_part` neither 'real' nor 'imag'. ")


def radial_spherical_int(rfunc, rmax):
    """Radial integral up to the given maximal radius.

    Parameters
    ----------
    rfunc : callable
        Radial function to be integrated.
    rmax : float
        Upper radial limit.

    Returns
    -------
    integral : float
        Radial integral value.

    """
    integral, _ = quad(_radial_integrand, 0, rmax, args=(rfunc))

    return integral


def angular_spherical_int(afunc):
    r"""Full angular integral.

    Parameters
    ----------
    afunc : callable
        Angular function be integrated.

    Returns
    -------
    complex
        Full angular integral value.

    Notes
    -----
    Arguments of the angular function `afunc` must be in radians and in the
    following order and range: :math:`(\theta, \phi) \in [0, \pi] \times
    [0, 2\pi)`.

    """
    integral_real, _ = dblquad(
        _angular_integrand, 0, np.pi, 0, 2*np.pi, args=(afunc, 'real')
        )
    integral_imag, _ = dblquad(
        _angular_integrand, 0, np.pi, 0, 2*np.pi, args=(afunc, 'imag')
        )

    return integral_real + 1j*integral_imag


def radial_sphint_besselj(rfunc, ell, k, rmax, *args, **kwargs):
    """Radial integral against spherical Bessel functions at the specified wave
    number up to the given maximal radius.

    Parameters
    ----------
    rfunc : callable
        Radial function to be integrated.
    ell : int
        Order of the spherical Bessel function.
    k : float
        Wave number.
    rmax : float
        Upper radial limit.
    *args, **kwargs
        Additional positional and keyword arguments to be passed to `rfunc`.

    Returns
    -------
    float
        Integral value

    """
    def kernel(r):
        return rfunc(r, *args, **kwargs) * sph_besselj(ell, k*r)

    return radial_spherical_int(kernel, rmax)


def angular_sphint_harmonic(afunc, ell, m, *args, conjugate=True, **kwargs):
    r"""Full angular integral against spherical harmonic functions.

    Parameters
    ----------
    afunc : callable
        Angular function be integrated.
    ell : int
        Degree of the spherical harmonic function.
    m : int
        Order of the spherical harmonic function.
    conjugate : bool, optional
        If `True` (default), use conjugate of the spherical harmonic function.
    *args, **kwargs
        Additional positional and keyword arguments to be passed to `afunc`.

    Returns
    -------
    complex
        Integral value.

    Notes
    -----
    The arguments of the angular function `afunc` must be in radians and in
    the following order and range: :math:`(\theta, \phi) \in [0, \pi] \times
    [0, 2\pi)`.

    """
    def kernel(theta, phi):
        return afunc(theta, phi, *args, **kwargs) \
            * sph_harmonic(ell, m, theta, phi)

    if conjugate:
        return np.conj(angular_spherical_int(kernel))
    return angular_spherical_int(kernel)
