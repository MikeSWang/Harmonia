r"""
Scale dependence (:mod:`~harmonia.cosmology.scale_dependence`)
===========================================================================

Compute scale-dependent modifications to galaxy clustering at the current
epoch, i.e. redshift :math:`z = 0`, due to local primordial non-Gausianity
:math:`f_\textrm{NL}` .

.. autosummary::

    scale_dependent_bias
    scale_modified_power_spectrum

|

"""
from nbodykit import cosmology

_REDSHIFT_EPOCH = 0.
_SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY = 1.686
_SPEED_OF_LIGHT_IN_HUNDRED_KM_PER_S = 2998.


def scale_dependent_bias(f_nl, b_const, cosmo):
    r"""Return scale-dependent bias as a function for non-vanishing local
    primordial non-Gaussianity :math:`f_\textrm{NL}`.

    The constant bias :math:`b_1` is modified as

    .. math::

        b_1 \mapsto b_1 + (b_1 - 1) f_\textrm{NL} \frac{\alpha(k)}{k^2} \,,

    where

    .. math::

        \alpha(k) = 3 \left( \frac{H_0}{\mathrm{c}} \right)^2
            \frac{\Omega_\textrm{m} \delta_\textrm{c}}{T(k)}

    and the transfer function :math:`T(k)` is computed by ``nbodykit``
    with ``CLASS``.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussnianity.
    b_const : float
        Constant linear bias at the current epoch.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.

    Returns
    -------
    bias : callable
        Scale-dependent bias as a function of the Fourier scale (in h/Mpc).

    """
    num_factors = 3*_SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY * cosmo.Omega0_m \
         * (cosmo.h / _SPEED_OF_LIGHT_IN_HUNDRED_KM_PER_S)**2

    def bias(k):

        b_k = b_const + (b_const - 1) * f_nl * (num_factors / k**2) / (
            cosmology.power.transfers.CLASS(
                cosmo,
                redshift=_REDSHIFT_EPOCH
            )(k)
        )

        return b_k

    return bias


def scale_modified_power_spectrum(f_nl, b_const, cosmo):
    """Biased power spectrum with non-Gaussianity scale dependence
    modification.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussnianity.
    b_const : float
        Constant linear bias at the current epoch.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.

    Returns
    -------
    modified_power_spectrum : callable
        Non-Gaussianity modified biased power spectrum as a function of
        the Fourier scale (in h/Mpc).

    """
    _scale_dependent_bias = scale_dependent_bias(f_nl, b_const, cosmo)
    _power_spectrum = cosmology.LinearPower(cosmo, redshift=_REDSHIFT_EPOCH)

    modified_power_spectrum = lambda k: _scale_dependent_bias(k) \
        * _power_spectrum(k)

    return modified_power_spectrum
