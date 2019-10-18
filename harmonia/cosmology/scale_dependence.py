r"""
Scale dependence (:mod:`~harmonia.cosmology.scale_dependence`)
===========================================================================

Compute scale-dependent modifications to galaxy clustering from local
primordial non-Gausianity :math:`f_\textrm{NL}`.

The constant linear bias is modified as

    .. math::

        b_1(z) \mapsto b_1(z) + f_\textrm{NL} [b_1(z) - p] A(k,z) \,,

where :math:`p` is a tracer species--dependent parameter and the
scale-dependence modification kernel

    .. math::

        A(k,z) = 3 \left( \frac{H_0}{\mathrm{c}} \right)^2
            \frac{\Omega_\textrm{m} \delta_\textrm{c}}{k^2 T(k,z)}

relates to the cosmological model and its transfer function :math:`T(k,z)`
computed by ``nbodykit`` with ``CLASS``.

.. autosummary::

    bias_modification
    scale_modified_power_spectrum

|

"""
from nbodykit import cosmology


def scale_modification(cosmo, redshift):
    r"""Return the scale-dependence modification kernel :math:`A(k,z)/k^2`
    for a given cosmological model.

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

        return num_factors / (k**2 * transfer_func(k))

    return kernel


def scale_dependent_bias(bz_const, f_nl, cosmo, redshift=0.,
                         tracer_parameter=1.):
    r"""Return the scale-dependent bias modulated by local primordial
    non-Gaussianity.

    Parameters
    ----------
    bz_const : float
        Constant linear bias.  Must be the value at the specified redshift.
    f_nl
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.
    redshift : float, optional
        Redshift at which quantities are evaluated (default is 0.).
    tracer_parameter : float, optional
        Tracer species--dependent parameter :math:`p` (default is 1.).

    Returns
    -------
    bias_k : callable
        Scale-dependent bias as a function of the Fourier scale (in h/Mpc).

    """
    def bias_k(k):

        bias_of_k = bz_const + (bz_const - tracer_parameter) \
            * scale_modification(cosmo, redshift)(k)

        return bias_of_k

    return bias_k


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
    power_spectrum : callable
        Non-Gaussianity modified biased power spectrum as a function of
        the Fourier scale (in h/Mpc).

    """

    def power_spectrum(k):

        bias = scale_dependent_bias(
            bz_const,
            f_nl,
            cosmo,
            redshift=redshift,
            tracer_parameter=tracer_parameter
        )
        power = cosmology.LinearPower(cosmo, redshift=redshift)

        return bias(k)**2 * power(k)

    return power_spectrum
