r"""
Scale dependence (:mod:`~harmonia.cosmology.scale_dependence`)
===========================================================================

Compute scale-dependent modifications due to local primordial
non-Gausianity :math:`f_\textrm{NL}`.

.. autosummary::

    scale_dependent_bias
    non_gaussianity_biased_power_spectrum

|

"""
from nbodykit import cosmology

_REDSHIFT_EPOCH = 0.
_SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY = 1.686
_SPEED_OF_LIGHT_IN_HUNDRED_KM_PER_S = 2998.


def scale_dependent_bias(k, f_nl, b_const, cosmo):
    """Scale-dependent bias for non-vanishing local primordial
    non-Gaussianity.

    The transfer function appearing in the Poisson kernel is computed using
    ``CLASS``.

    Parameters
    ----------
    k : float, array_like
        Fourier scale/wave number.
    f_nl : float
        Local primordial non-Gaussnianity.
    b_const : float
        Constant linear bias.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.

    Returns
    -------
    bias : float, array_like
        Scale-dependent bias.

    """
    num_factors = 3*_SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY * cosmo.Omega0_m \
         * (cosmo.h / _SPEED_OF_LIGHT_IN_HUNDRED_KM_PER_S)**2

    bias = b_const + (b_const - 1) * f_nl * num_factors \
        / cosmology.power.transfers.CLASS(cosmo, redshift=_REDSHIFT_EPOCH)(k) \
        / k**2

    return bias


def non_gaussianity_biased_power_spectrum(f_nl, b_const, cosmo):
    """Biased power spectrum with non-Gaussianity scale dependence
    modification.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussnianity.
    b_const : float
        Constant linear bias.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.

    Returns
    -------
    modified_power_spectrum : callable
        Non-Gaussianity modified biased power spectrum.

    """
    power_spectrum = cosmology.LinearPower(cosmo, redshift=_REDSHIFT_EPOCH)
    modified_power_spectrum = lambda k: \
        scale_dependent_bias(k, f_nl, b_const, cosmo) * power_spectrum(k)

    return modified_power_spectrum
