r"""
Scale dependence (:mod:`~harmonia.cosmology.scale_dependence`)
===========================================================================

Compute scale-dependent modifications to galaxy clustering from local
primordial non-Gausianity :math:`f_\textrm{NL}`.

The constant linear bias is modified as

    .. math::

        b_1(z) \mapsto b_1(z) \
            + f_\textrm{NL} [b_1(z) - p] \frac{A(k,z)}{k^2} \,,

where :math:`p` is a tracer species--dependent parameter and the
scale-dependence modification kernel

    .. math::

        A(k,z) = 3 \left( \frac{H_0}{\mathrm{c}} \right)^2
            \frac{\Omega_\textrm{m} \delta_\textrm{c}}{T(k,z)}

relates to the transfer function :math:`T(k,z)` computed by ``nbodykit``
with ``CLASS``.

.. autosummary::

    bias_modification
    scale_modified_power_spectrum

|

"""
from nbodykit import cosmology


def _scale_modification_kernel(cosmo, redshift):
    r"""Return scale-dependence modification kernel :math:`A(k,z)` for a
    a given cosmology.

    Parameters
    ----------
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.
    redshift : float
        Redshift at which quantities are evaluated.

    Returns
    -------
    kernel : callable
        Scale-dependence modification kernel as a function of the Fourier
        scale (in h/Mpc).

    """
    SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY = 1.686
    SPEED_OF_LIGHT_IN_HUNDRED_KM_PER_S = 2998.

    num_factors = 3 * (cosmo.h / SPEED_OF_LIGHT_IN_HUNDRED_KM_PER_S)**2 \
        * SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY * cosmo.Omega0_m
    transfer_func = \
        cosmology.power.transfers.CLASS(cosmo, redshift=redshift)

    def kernel(k):

        return num_factors / transfer_func(k)

    return kernel


def bias_modification(bz_const, cosmo, redshift=0., tracer_parameter=1.):
    r"""Return the scale-dependent modification to the constant linear bias
    amplified by the local primordial non-Gaussianity value
    :math:`f_\textrm{NL}`.

    Parameters
    ----------
    bz_const : float
        Constant linear bias.  Must be the value at the specified redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.
    redshift : float, optional
        Redshift at which quantities are evaluated (default is 0.).
    tracer_parameter : float, optional
        Tracer species--dependent parameter :math:`p` (default is 1.).

    Returns
    -------
    delta_bias : callable
        Scale-dependent bias modificationn as a function of the Fourier
        scale (in h/Mpc).

    """
    def delta_bias(k):

        d_bias = (bz_const - tracer_parameter) / k**2 \
            * _scale_modification_kernel(cosmo, redshift)(k)

        return d_bias

    return delta_bias


def scale_modified_power_spectrum(f_nl, bz_const, cosmo, redshift=0.,
                                  tracer_parameter=1.):
    """Biased power spectrum with non-Gaussianity scale dependence
    modification.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussnianity.
    bz_const : float
        Constant linear bias.  Must be the value at the specified redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.
    redshift : float, optional
        Redshift at which quantities are evaluated (default is 0.).
    tracer_parameter : float, optional
        Tracer species--dependent parameter :math:`p` (default is 1.).

    Returns
    -------
    modified_power_spectrum : callable
        Non-Gaussianity modified biased power spectrum as a function of
        the Fourier scale (in h/Mpc).

    """

    def modified_power_spectrum(k):

        bias_k = bz_const + f_nl * bias_modification(
            bz_const,
            cosmo,
            redshift=redshift,
            tracer_parameter=tracer_parameter
        )(k)
        power_k = cosmology.LinearPower(cosmo, redshift=redshift)(k)

        return bias_k**2 * power_k

    return modified_power_spectrum
