"""Hybrid likelihood constructed from data realisation and modelling.

"""
import numpy as np
from nbodykit.lab import cosmology

from harmonia.cosmology import modified_power_spectrum
from harmonia.reader import complex_normal_pdf, multivariate_normal_pdf

_OVERFLOW_DOWNSCALE = 10**4


# Spherical likelihood
# ------------------------------------------------------------------------------

def _spherical_parametrised_covariance(two_point_model, pivot, nbar, b_10,
                                       f_nl, independence=False, diag=False):
    r"""Compute the parametrised covariance matrix given the 2-point
    function base model and the pivot axis for data vectorisation of
    transformed fields.

    Parameters
    ----------
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function base model.
    pivot : {'natural', 'transposed', 'spectral'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    f_nl : float
        Local primordial non-Gaussianity.
    b_10 : float
        Scale-indepepdent linear bias at the current epoch.
    independence : bool, optional
        If `True` (default is `False`), independence amongst Fourier modes
        is assumed and the diagonal covariance matrix is computed without
        coupling coefficients.
    diag : bool, optional
        If `True` (default is `False`), only the diagonal elements of the
        covariance matrix are computed by summation against of coupling
        coefficients.

    Returns
    -------
    covariance : complex or float :class:`numpy.ndarray`
        Parametrised covariance matrix.

    See Also
    --------
    :class:`~harmonia.reader.spherical_model.TwoPointFunction`

    """
    if independence:
        variance = two_point_model.mode_variance(
            pivot,
            nbar=nbar,
            b_10=b_10,
            f_nl=f_nl
        )
        covariance = np.diag(variance)
    else:
        covariance = two_point_model.two_point_covariance(
            pivot,
            diag=diag,
            nbar=nbar,
            b_10=b_10,
            f_nl=f_nl
        )

    return covariance


def spherical_map_likelihood(param_points, param_name, spherical_data,
                             two_point_model, pivot, nbar, bias=None,
                             f_nl=None, breakdown=False, remove_degrees=(),
                             mode_indices=None, independence=False,
                             diag=False):
    """Evaluate the spherical map logarithmic likelihood.

    Parameters
    ----------
    param_points : float, array_like
        Parameter sampling values.
    param_name : {'f_nl', 'bias'}
        Name of the parameter to be sampled, either ``'f_nl'`` for the
        local primordial non-Gaussianity or ``'bias'`` for the
        scale-independent linear bias.
    spherical_data : :class:`~harmonia.algorithms.morph.SphericalArray`
        Spherical data array of the transformed field.
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function base model.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    bias : float or None, optional
        Scale-independent linear bias of the tracer particles at the
        current epoch.  Cannot be `None` (default) if `param_name` is
        ``'f_nl'``.
    f_nl : float or None, optional
        Local primordial non-Gaussianity.  Cannot be `None` (default) if
        `param_name` is ``'bias'``.

    Other parameters
    ----------------
    remove_degrees : int, array_like, optional
        If not an empty tuple (default), modes whose spherical degree is an
        element are removed from the data vector and parametrised
        covariance.
    mode_indices : list of (int, int, int) or None, optional
        Mode indices of the spherical data.  Cannot be `None` (default) if
        `remove_degrees` is not `None`.
    breakdown : bool, optional
        If `True` (default is `False`), the contribution from each data
        vector is broken down but the covariance matrix must be diagonal.
    independence : bool, optional
        If `True` (default is `False`), independence amongst Fourier modes
        is assumed and the diagonal covariance matrix is computed without
        coupling coefficients.
    diag : bool, optional
        If `True` (default is `False`), only the diagonal elements of the
        covariance matrix are computed by summation against of coupling
        coefficients.

    Returns
    -------
    log_likelihood : float :class:`numpy.ndarray`
        Logarithmic likelihood evaluated at the parameter sampling points.

    Raises
    ------
    ValueError
        If `bias` is `None` when `param_name` is ``'f_nl'`` or `f_nl` is
        `None` when `param_name` is ``'bias'``, or if `param_name` neither.

    """
    data_vector = spherical_data.unfold(pivot, return_only='data')
    if remove_degrees:
        if mode_indices is None:
            raise ValueError(
                "`mode_indices` must be provided "
                "if `remove_degrees` is non-empty. "
            )
        excluded_deg = np.fromiter(
            map(lambda index: index[0] in remove_degrees, mode_indices),
            dtype=bool
        )
        data_vector = data_vector[~excluded_deg]

    param_covar_args = (two_point_model, pivot, nbar)

    if breakdown:
        log_likelihood = np.zeros((len(param_points), len(data_vector)))
    else:
        log_likelihood = np.zeros_like(param_points)
    for idx, param in enumerate(param_points):
        if param_name == 'f_nl':
            if bias is None:
                raise ValueError(
                    "`bias` value must be provided "
                    "for sampling non-Gaussnaity. "
                )
            sample_covar = _spherical_parametrised_covariance(
                *param_covar_args,
                bias,
                param,
                independence=independence,
                diag=diag
            )
        elif param_name == 'bias':
            sample_covar = _spherical_parametrised_covariance(
                *param_covar_args,
                param,
                f_nl,
                independence=independence,
                diag=diag
            )
        else:
            raise ValueError(
                f"Invalid `param_name` for sampling: {param_name}. "
            )

        if remove_degrees:
            sample_covar = sample_covar[:, ~excluded_deg][~excluded_deg, :]

        log_likelihood[idx] = complex_normal_pdf(
            data_vector,
            sample_covar,
            downscale=_OVERFLOW_DOWNSCALE,
            elementwise=breakdown
        )

    return log_likelihood


# Cartesian likelihood
# ------------------------------------------------------------------------------

def _cartesian_parametrised_moments(wavenumbers, b_10, nbar=None, f_nl=None,
                                    cosmo=None, reshift=0., mode_count=None,
                                    contrast=None, power_spectrum=None):
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
    reshift : float, optional
        Reshift at which models are evaluated (default is 0.).
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
            power_spectrum = cosmology.LinearPower(cosmo, reshift)
        expected_power = b_10**2 * power_spectrum(wavenumbers)
    else:
        if power_spectrum is None:
            power_spectrum = modified_power_spectrum(
                f_nl, b_10, cosmo, redshift=reshift
            )
        else:
            power_spectrum = modified_power_spectrum(
                f_nl, b_10, cosmo,
                redshift=reshift,
                power_spectrum=power_spectrum
            )
        expected_power = power_spectrum(wavenumbers)

    if nbar is not None:
        shot_noise = 1 / nbar
        if contrast is not None:
            shot_noise *= 1 + 1 / contrast
        expected_power += shot_noise

    if mode_count is None:
        power_variance = None
    else:
        power_variance = 2 / mode_count * expected_power**2

    return expected_power, power_variance


def cartesian_map_likelihood(param_points, param_name, cartesian_data, nbar,
                             cosmo=None, redshift=0., bias=None, f_nl=None,
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
    reshift : float, optional
        Reshift at which models are evaluated (default is 0.).
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
            mean_vector, variance_vector = _cartesian_parametrised_moments(
                wavenumbers,
                bias,
                nbar=nbar,
                f_nl=param,
                cosmo=cosmo,
                redshift=redshift,
                mode_count=mode_count,
                contrast=contrast,
                power_spectrum=power_spectrum
            )
            log_likelihood[idx] = multivariate_normal_pdf(
                data_vector, mean_vector, variance_vector
            )
    elif param_name == 'bias':
        for idx, param in enumerate(param_points):
            mean_vector, variance_vector = _cartesian_parametrised_moments(
                wavenumbers,
                param,
                nbar=nbar,
                f_nl=f_nl,
                cosmo=cosmo,
                redshift=redshift,
                mode_count=mode_count,
                contrast=contrast,
                power_spectrum=power_spectrum
            )
            log_likelihood[idx] = multivariate_normal_pdf(
                data_vector, mean_vector, variance_vector
            )

    return log_likelihood
