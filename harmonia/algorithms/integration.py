"""
Numerical integration (:mod:`~harmonia.algorithms.integration`)
===========================================================================

Integrate numerically in specified coordinate systems.

.. warning::

    Quadrature integration of spherical functions may suffer from poor
    convergence.


**Spherical integrals**

.. autosummary::

    angular_integral
    radial_integral
    pixelated_angular_integral

|

"""
import healpy as hp
import numpy as np
from scipy.integrate import dblquad, quad


# Spherical integrals
# -----------------------------------------------------------------------------

def _angular_integrand(phi, theta, func, complex_part):
    r"""Evaluate the angular integrand with the Jacobian factor
    :math:`\sin\theta`.

    Notes
    -----
    Angular arguments are in reverse order for outward double integration.

    Parameters
    ----------
    phi, theta: float, array_like
        Angular coordinates in radians.
    func : callable
        Angular function to be integrated.
    complex_part : {'real', 'imag'}
        Real or imaginary part.

    Returns
    -------
    float, array_like
        Real or imaginary part of the complex angular integrand value.

    """
    if complex_part.lower() == 'real':
        return np.abs(np.sin(theta)) * np.real(func(theta, phi))

    if complex_part.lower() == 'imag':
        return np.abs(np.sin(theta)) * np.imag(func(theta, phi))

    raise ValueError("`complex_part` is neither 'real' nor 'imag'.")


def _radial_integrand(r, func):
    r"""Evaluate the radial integrand with the Jacobian factor :math:`r^2`.

    Parameters
    ----------
    r: float, array_like
        Radial coordinate.
    func : callable
        Radial function to be integrated.

    Returns
    -------
    float, array_like
        Radial integrand value.

    """
    return r**2 * func(r)


def angular_integral(angular_func):
    r"""Compute the full spherical angular integral.

    Notes
    -----
    Arguments :math:`(\theta, \phi)` of `angular_func` must be in radians
    in the domain :math:`[0, \pi] \times [0, 2\pi]`.

    Parameters
    ----------
    angular_func : callable
        Angular function to be integrated.

    Returns
    -------
    complex
        Angular integral value.

    """
    theta_range = (0., np.pi)
    phi_range = (0., 2*np.pi)

    integral_real, _ = dblquad(
        _angular_integrand, *theta_range, *phi_range,
        args=(angular_func, 'real')
    )
    integral_imag, _ = dblquad(
        _angular_integrand, *theta_range, *phi_range,
        args=(angular_func, 'imag')
    )

    return integral_real + 1j*integral_imag


def radial_integral(radial_func, rmax):
    """Compute the radial integral up to the given maximum radius.

    Parameters
    ----------
    radial_func : callable
        Radial function to be integrated.
    rmax : float
        Upper radial limit ``rmax > 0``.

    Returns
    -------
    integral : float
        Radial integral value.

    """
    integral, _ = quad(_radial_integrand, 0., rmax, args=(radial_func,))

    return integral


def pixelated_angular_integral(angular_func, nside):
    r"""Compute the full spherical angular integral with pixelation.

    Notes
    -----
    Arguments :math:`(\theta, \phi)` of `angular_func` must be in radians
    in the domain :math:`[0, \pi] \times [0, 2\pi]`.

    Parameters
    ----------
    angular_func : callable
        Angular function to be integrated.
    nside : int
        'NSIDE' parameter for `healpy` pixelation.

    Returns
    -------
    complex
        Angular integral value.

    """
    num_pixel = hp.nside2npix(nside)

    theta, phi = hp.pix2ang(nside, ipix=range(num_pixel))

    pixel_area = 4 * np.pi / num_pixel

    pixel_value = angular_func(theta, phi)

    return pixel_area * np.sum(pixel_value)
