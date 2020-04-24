r"""
Scale dependence (:mod:`~harmonia.cosmology.scale_dependence`)
===========================================================================

Compute scale-dependence modifications to galaxy clustering from local
primordial non-Gausianity :math:`f_\textrm{NL}`.

The scale-independent linear bias is modified as

    .. math::

        b_1(z) \mapsto b_1(z) + f_\textrm{NL} [b_1(z) - p] A(k,z) \,,

where :math:`p` is a tracer-dependent parameter and the scale-dependence
modification kernel

    .. math::

        A(k,z) = 1.27 \left( \frac{H_0}{\mathrm{c}} \right)^2
            \frac{3\Omega_\textrm{m,0} \delta_\textrm{c}}{k^2 D(z) T(k)}

relates to the cosmological model and its growth factor :math:`D(z)`
(normalised to unity at the current epoch; hence the numerical factor
1.27) and transfer function :math:`T(k)` (normalised to unity as
:math:`k \to 0`).

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
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology`
        Cosmological model.
    redshift : float
        Redshift at which quantities are evaluated.

    Returns
    -------
    callable
        Scale-dependence modification kernel as a function of wavenumber
        (in :math:`h`/Mpc).

    """
    SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY = 1.686
    SPEED_OF_LIGHT_IN_KM_PER_S = 299792.
    GROWTH_FACTOR_NORMALISATION = 1.27

    num_factors = GROWTH_FACTOR_NORMALISATION \
        * 3 * cosmo.Omega0_m * SPHERICAL_COLLAPSE_CRITICAL_OVERDENSITY \
        * (cosmo.H0 / SPEED_OF_LIGHT_IN_KM_PER_S) ** 2

    transfer_func = cosmology.power.transfers.CLASS(cosmo, redshift=redshift)

    return lambda k: num_factors / (k**2 * transfer_func(k))


def scale_dependent_bias(b_1, f_nl, cosmo, redshift=0., tracer_p=1.):
    r"""Return the scale-dependent bias modulated by local primordial
    non-Gaussianity.

    Parameters
    ----------
    b_1 : float
        Scale-independent linear bias at input redshift.
    f_nl : float
        Local primordial on-Gaussianity.
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology`
        Cosmological model.
    redshift : float, optional
        Redshift at which quantities are evaluated (default is 0.).
    tracer_p : float, optional
        Tracer-dependent parameter :math:`p` (default is 1.).

    Returns
    -------
    callable
        Scale-dependent bias as a function of wavenumber (in
        :math:`h`/Mpc).

    """
    scale_dependence = scale_dependence_modification(cosmo, redshift)

    def b_k(k):
        return b_1 + f_nl * (b_1 - tracer_p) * scale_dependence(k)

    return b_k


def modified_power_spectrum(b_1, f_nl, cosmo, redshift=0., tracer_p=1.,
                            nbar=None, contrast=None):
    r"""Return the tracer power spectrum modified by primordial
    non-Gaussianity.

    Parameters
    ----------
    b_1 : float
        Scale-independent linear bias at input redshift.
    f_nl : float
        Local primordial non-Gaussianity.
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology`
        Cosmological model.
    redshift : float, optional
        Redshift at which quantities are evaluated (default is 0.).
    tracer_p : float, optional
        Tracer-dependent parameter :math:`p` (default is 1.).
    nbar : float or None, optional
        If not `None` (default), add ``1/nbar`` as shot noise to the
        resulting power spectrum.
    contrast : float or None, optional
        If not `None` (default), add additional ``1/(contrast*nbar)``
        as shot noise to the resulting power spectrum.

    Returns
    -------
    callable
        Tracer power spectrum modified by primordial non-Gaussianity as a
        function of wavenumber (in :math:`h`/Mpc).

    """
    bias = scale_dependent_bias(
        b_1, f_nl, cosmo, redshift=redshift, tracer_p=tracer_p
    )

    power_spectrum = cosmology.LinearPower(cosmo, redshift=redshift)

    alpha = 0. if contrast is None else 1 / contrast
    shot_noise = 0. if nbar is None else (1 + alpha) / nbar

    return lambda k: bias(k) ** 2 * power_spectrum(k) + shot_noise
