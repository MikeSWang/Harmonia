"""
Spherical kernels (:mod:`~harmonia.reader.kernels`)
===========================================================================

Evaluate integration kernels for computing spherical Fourier coupling
coefficients.

"""
import numpy as np

from harmonia.algorithms.bases import spherical_besselj, spherical_harmonic


def angular_kernel(theta, phi, mu, nu, mask=None):
    r"""Evaluate the angular coupling kernel.

    Parameters
    ----------
    theta, phi : float, array_like
        Angular coordinates :math:`\theta`, :math:`\phi`.
    mu, nu : tuple(int, int, int)
        Coefficient triplet index.
    mask : callable or None, optional
        Mask as a function of angular coordinates (default is `None`).

    Returns
    -------
    kernel : complex :class:`numpy.ndarray`
        Angular coupling kernel value.

    """
    kernel = \
        spherical_harmonic(mu[0], mu[1], theta, phi, conj=True) \
        * spherical_harmonic(nu[0], nu[1], theta, phi)

    if callable(mask):
        kernel *= mask(np.column_stack([theta, phi]))

    return kernel


def radial_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None,
                  bias_evolution=None, clustering_evolution=None,
                  z_from_r=None, chi_of_z=None):
    """Evaluate the radial coupling kernel.

    Parameters
    ----------
    r : float, array_like
        Radial coordinate.
    mu, nu : tuple(int, int, int)
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate
        (default is `None`).
    bias_evolution, clustering_evolution : callable or None, optional
        Bias and clustering evolution as a function of redshift
        normalised to unity at the current epoch (default is `None`).
    z_from_r : callable or None, optional
        Comoving distance-to-redshift conversion in a variable
        cosmological model (default is `None`).
    chi_of_z : callable or None, optional
        Comoving redshift-to-distance conversion in a fiducial
        cosmological model (default is `None`).

    Returns
    -------
    kernel : float :class:`numpy.ndarray`
        Radial coupling kernel value.

    Raises
    ------
    TypeError
        If `z_from_r` is not callable when any of `bias_evolution`,
        `clustering_evolution` and `chi_of_z` is.

    """
    if any(map(callable, [chi_of_z, clustering_evolution, bias_evolution])) \
            and not callable(z_from_r):
        raise TypeError(
            "`z_from_r` must be callable when any of "
            "`bias_evolution`, `clustering_evolution` and `chi_of_z` is."
        )

    if not callable(chi_of_z):
        r_tilde = r
    else:
        r_tilde = chi_of_z(z_from_r(r))

    kernel = \
        spherical_besselj(mu[0], k_mu * r_tilde) \
        * spherical_besselj(nu[0], k_nu * r)

    if callable(selection):
        kernel *= selection(r)

    if callable(weight):
        kernel *= weight(r_tilde)

    if callable(clustering_evolution):
        kernel *= clustering_evolution(z_from_r(r))

    if callable(bias_evolution):
        kernel *= bias_evolution(z_from_r(r))

    return kernel


def RSD_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None,
               weight_derivative=None, growth_evolution=None,
               clustering_evolution=None, z_from_r=None, chi_of_z=None,
               differential_AP_distortion=None):
    """Evaluate the RSD coupling kernel.

    Parameters
    ----------
    r : float, array_like
        Radial coordinate.
    mu, nu : tuple(int, int, int)
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight, weight_derivative : callable or None, optional
        Selection, weight or weight derivative as a function of the
        radial coordinate (default is `None`).
    growth_evolution, clustering_evolution : callable or None, optional
        Growth rate evolution or clustering evolution as a function of
        redshift normalised to unity at the current epoch (default is
        `None`).
    z_from_r : callable or None, optional
        Cosmological comoving distance-to-redshift conversion (default
        is `None`).
    chi_of_z : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is
        `None`).
    differential_AP_distortion : callable or None, optional
        Differential AP distortion as a function of redshift (default
        is `None`).

    Returns
    -------
    kernel : float, array_like
        RSD coupling kernel value.

    Raises
    ------
    TypeError
        If `z_from_r` is not callable when any of `growth_evolution`,
        `clustering_evolution` and `chi_of_z` is.
    TypeError
        If `weight` and `weight_derivative` are not both None or callable.
    TypeError
        If `chi_of_z` and `AP_distortion` are not both None or callable.

    """
    if any(map(callable, [chi_of_z, growth_evolution, clustering_evolution])) \
            and not callable(z_from_r):
        raise TypeError(
            "`z_from_r` must be callable when any of "
            "`growth_evolution`, `clustering_evolution` and `chi_of_z` is."
        )

    apply_weight = any(map(callable, [weight, weight_derivative]))
    weight_applicable = all(map(callable, [weight, weight_derivative]))
    if apply_weight and not weight_applicable:
        raise TypeError(
            "`weight` and `weight_derivative` must both be callable "
            "in order to apply weight."
        )

    apply_AP = any(map(callable, [chi_of_z, differential_AP_distortion]))
    AP_applicable = all(map(callable, [chi_of_z, differential_AP_distortion]))
    if apply_AP and not AP_applicable:
        raise TypeError(
            "`chi_of_z` and `differential_AP_distortion` must both be "
            "callable in order to apply AP correction."
        )

    kernel = spherical_besselj(nu[0], k_nu*r, derivative=True)

    if apply_AP:
        r_tilde = chi_of_z(z_from_r(r))
        kernel *= differential_AP_distortion(z_from_r(r))
    else:
        r_tilde = r

    if callable(selection):
        kernel *= selection(r)

    if apply_weight:
        kernel *= (
            weight_derivative(r_tilde)
            * spherical_besselj(mu[0], k_mu * r_tilde) \
            + k_mu * weight(r_tilde) \
            * spherical_besselj(mu[0], k_mu * r_tilde, derivative=True)
        )
    else:
        kernel *= k_mu * spherical_besselj(mu[0], k_mu * r, derivative=True)

    if callable(growth_evolution):
        kernel *= growth_evolution(z_from_r(r))

    if callable(clustering_evolution):
        kernel *= clustering_evolution(z_from_r(r))

    return kernel


def shot_noise_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None):
    """Evalaute the two-point correlator shot noise kernel.

    Parameters
    ----------
    r : float, array_like
        Radial coordinate.
    mu, nu : tuple(int, int, int)
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate
        (default is `None`).

    Returns
    -------
    kernel : float, array_like
        Two-point correlator shot noise kernel value.

    """
    kernel = \
        spherical_besselj(mu[0], k_mu * r) \
        * spherical_besselj(nu[0], k_nu * r)

    if callable(selection):
        kernel *= selection(r)

    if callable(weight):
        kernel *= weight(r) ** 2

    return kernel
