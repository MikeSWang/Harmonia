r"""
Scale dependence (:mod:`~harmonia.cosmology.scale_dependence`)
===========================================================================

Compute scale-dependence modifications to galaxy clustering from local
primordial non-Gausianity :math:`f_\textrm{NL}`.

The scale-independent linear bias is modified as

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

    scale_dependence_modification
    scale_dependent_bias
    modified_power_spectrum

|

"""
from nbodykit import cosmology


def scale_dependence_modification(cosmo, redshift):
    r"""Return the scale-dependence modification kernel :math:`A(k,z)` for
    a given cosmological model.

    Parameters
    ----------
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.
    redshift : float
        Redshift at which quantities are evaluated.

    Returns
    -------
    callable
        Scale-dependence modification kernel as a function of the Fourier
        scale (in h/Mpc).

    """
    SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY = 1.686
    SPEED_OF_LIGHT_IN_HUNDRED_KM_PER_S = 2998.
    NORMALISATION = 1.3

    num_factors = 3 * NORMALISATION * cosmo.Omega0_m \
        * SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY \
        * (cosmo.h / SPEED_OF_LIGHT_IN_HUNDRED_KM_PER_S)**2
    transfer_func = cosmology.power.transfers.CLASS(cosmo, redshift=redshift)

    return lambda k: num_factors / (k**2 * transfer_func(k))


def scale_dependent_bias(b_1, f_nl, cosmo, redshift=0., tracer_parameter=1.):
    r"""Return the scale-dependent bias modulated by local primordial
    non-Gaussianity.

    Parameters
    ----------
    b_1 : float
        Scale-independent linear bias.  Must be the value at the specified
        redshift.
    f_nl : float
        Local primordial non-Gaussnianity.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.
    redshift : float, optional
        Redshift at which quantities are evaluated (default is 0.).
    tracer_parameter : float, optional
        Tracer species--dependent parameter :math:`p` (default is 1.).

    Returns
    -------
    b_k : callable
        Scale-dependent bias as a function of the Fourier scale (in h/Mpc).

    """
    def b_k(k):

        b_of_k = b_1 + f_nl * (b_1 - tracer_parameter) \
            * scale_dependence_modification(cosmo, redshift)(k)

        return b_of_k

    return b_k


def modified_power_spectrum(f_nl, b_1, cosmo, redshift=0., tracer_parameter=1.,
                            power_spectrum=None):
    """Return the biased power spectrum with non-Gaussianity
    scale-dependence modification.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussnianity.
    b_1 : float
        Scale-independent linear bias.  Must be the value at the specified
        redshift.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model for density parameters and the transfer
        function.
    redshift : float, optional
        Redshift at which quantities are evaluated (default is 0.).
    tracer_parameter : float, optional
        Tracer species--dependent parameter :math:`p` (default is 1.).

    Other Parameters
    ----------------
    power_spectrum : callable or None, optional
        Power spectrum model without scale-dependence modification.  If not
        `None` (default), this overrides that given by `cosmo`.

    Returns
    -------
    callable
        Non-Gaussianity modified biased power spectrum as a function of
        the Fourier scale (in h/Mpc).

    """
    if power_spectrum is None:
        power_spectrum = cosmology.LinearPower(cosmo, redshift=redshift)

    bias = scale_dependent_bias(
        b_1, f_nl, cosmo, redshift=redshift, tracer_parameter=tracer_parameter
    )

    return lambda k: bias(k)**2 * power_spectrum(k)
