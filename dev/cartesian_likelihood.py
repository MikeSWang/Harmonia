"""Cartesian likelihood constructed from data realisation and covariance
modelling.

"""
import numpy as np
from nbodykit.lab import cosmology

from harmonia.cosmology import modified_power_spectrum

_OVERFLOW_DOWNSCALE = 10**4
_CURRENT_REDSHIFT = 0.


def parametrised_moments(wavenumbers, b_10, nbar=None, f_nl=None, cosmo=None,
                         mode_count=None, contrast=None, power_spectrum=None):
    """Compute the parametrised moment(s) of power spectrum realisations.

    Parameters
    ----------
    wavenumbers : float, array_like
        Wavenumbers at which the power spectrum moments are evaluated.
    b_10 : float
        Scale-independent linear bias at the current epoch.
    nbar : float or None, optional
        Mean particle number density (in cubic h/Mpc).  If `None`
        (default), shot noise is not included.
    f_nl : float or None, optional
        Local primordial non-Gaussianity (default is `None`).
    cosmo : :class:`nbodykit.cosmology.Cosmology` or None, optional
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating
        scale-dependent bias (default is `None`).
    mode_count : int, array_like or None, optional
        Number of grid modes for each wavenumber bin (default is `None`).

    Other Parameters
    ----------------
    contrast : float or None, optional
        Contrast of a random catalogue number density giving rise to
        additional shot noise.
    power_spectrum : callable or None, optional
        Power spectrum model.  If not `None` (default), this overrides that
        given by `cosmo`.

    Returns
    -------
    expected_power : float :class:`numpy.ndarray`
        Power spectrum expectation at the specified wavenumbers.
    power_variance : float :class:`numpy.ndarray`
        Power spectrum variance at the specified wavenumbers.

    """
    if f_nl is None:
        if power_spectrum is None:
            power_spectrum = cosmology.LinearPower(
                cosmo, redshift=_CURRENT_REDSHIFT
            )
        expected_power = b_10**2 * power_spectrum(wavenumbers)
    else:
        if power_spectrum is None:
            power_spectrum = modified_power_spectrum(
                f_nl, b_10, cosmo, redshift=_CURRENT_REDSHIFT
            )
        else:
            power_spectrum = modified_power_spectrum(
                f_nl, b_10, cosmo,
                redshift=_CURRENT_REDSHIFT,
                power_spectrum=power_spectrum
            )
        expected_power = power_spectrum(wavenumbers)

    if nbar is not None:
        shot_noise = 1 / nbar
        if contrast is not None:
            shot_noise *= 1 + 1 / contrast
    else:
        shot_noise = 0.

    expected_power += shot_noise

    if mode_count is None:
        power_variance = None
    else:
        power_variance = 2 / mode_count * expected_power**2

    return expected_power, power_variance


def cartesian_map_likelihood(param_points, param_name, cartesian_data,
                             nbar, cosmo=None, bias=None, f_nl=None,
                             contrast=None, power_spectrum=None):
    """Evaluate the Cartesian map logarithmic likelihood.

    Parameters
    ----------
    param_points : float, array_like
        Parameter sampling values.
    param_name : {'f_nl', 'bias'}
        Name of the parameter to be sampled, either ``'f_nl'`` for the
        local primordial non-Gaussianity or ``'bias'`` for the
        scale-independent linear bias.
    cartesian_data : dict
        Compressed Cartesian map data including summary statistics.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    cosmo : :class:`nbodykit.cosmology.Cosmology` or None, optional
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating
        scale-dependent bias (default is `None`).
    bias : float or None, optional
        Scale-independent linear bias at the current epoch (default is
        `None`).
    f_nl : float or None, optional
        Local primordial non-Gaussianity (default is `None`).

    Other Parameters
    ----------------
    contrast : float or None, optional
        Contrast of a random catalogue number density giving rise to
        additional shot noise.
    power_spectrum : callable or None, optional
        Power spectrum model.  If not `None` (default), this overrides that
        given by `cosmo`.

    Returns
    -------
    log_likelihood : float :class:`numpy.ndarray`
        Logarithmic likelihood evaluated at the parameter sampling points.

    Raises
    ------
    ValueError
        If `bias` is `None` when `param_name` is ``'f_nl'`` or `f_nl` is
        `None` when `param_name` is ``'bias'``, or if `param_name` neither.
    ValueError
        If the data array shapes in `cartesian_data` do not match.

    """
    data_vector = cartesian_data['Pk']

    wavenumbers, mode_count = cartesian_data['k'], cartesian_data['Nk']
    if np.squeeze(mode_count).shape != np.squeeze(wavenumbers).shape:
        raise ValueError(
            "`mode_count` and `wavenumbers` shapes do not match. "
        )

    log_likelihood = np.zeros_like(param_points)
    if param_name == 'f_nl':
        if bias is None:
            raise ValueError(
                "`bias` value must be provided "
                "for sampling non-Gaussnaity. "
            )
        for idx, param in enumerate(param_points):
            mean_vector, variance_vector = parametrised_moments(
                wavenumbers,
                bias,
                nbar=nbar,
                f_nl=param,
                cosmo=cosmo,
                mode_count=mode_count,
                contrast=contrast,
                power_spectrum=power_spectrum
            )

            det_divider = - 1/2 * np.sum(np.log(variance_vector))
            exponent = - 1/2 * np.sum(
                (data_vector - mean_vector)**2 / variance_vector
            )

            log_likelihood[idx] = det_divider + exponent
    elif param_name == 'bias':
        for idx, param in enumerate(param_points):
            mean_vector, variance_vector = parametrised_moments(
                wavenumbers,
                param,
                nbar=nbar,
                f_nl=None,
                cosmo=cosmo,
                mode_count=mode_count,
                contrast=contrast,
                power_spectrum=power_spectrum
            )

            det_divider = - 1/2 * np.sum(np.log(variance_vector))
            exponent = - 1/2 * np.sum(
                (data_vector - mean_vector)**2 / variance_vector
            )

            log_likelihood[idx] = det_divider + exponent

    return log_likelihood
