"""Spherical likelihood constructed from data realisation and covariance
modelling.

"""
import numpy as np

_OVERFLOW_DOWNSCALE = 10**4


def _chi_square(dat_vector, cov_matrix, by_element=False):
    """Compute chi-square for the specified data vector and covariance.

    The data vector is assumed to be zero-centred.

    Parameters
    ----------
    dat_vector : float or complex, array_like
        1-d data vector centred at zero.
    cov_matrix : float or complex, array_like
        2-d covariance matrix.
    by_element : bool, optional
        If `True` (default is `False`), return chi-square from each element
        of `dat_vector`.

    Returns
    -------
    chi_square : float, array_like
        Chi-square value(s).

    Raises
    ------
    ValueError
        If `data` is not equivalent to a 1-d vector.

    """
    if len(set(np.shape(dat_vector)).difference({1})) > 1:
        raise ValueError("`data` is not equivalent to a 1-d vector. ")

    if not by_element:
        chi_square = np.transpose(np.conj(dat_vector)) \
            @ np.linalg.inv(cov_matrix) \
            @ dat_vector
    else:
        chi_square = np.abs(dat_vector)**2 / np.diag(cov_matrix)

    return np.real(chi_square)


def _log_complex_normal_pdf(dat_vector, cov_matrix):
    """Compute natural log of the complex normal probability density
    function at specified data vector and covariance.

    The data vector is assumed to be zero-centred.

    Parameters
    ----------
    dat_vector : float or complex, array_like
        1-d data vector centred at zero.
    cov_matrix : float or complex, array_like
        2-d covariance matrix.

    Returns
    -------
    log_prob_density : float
        Log PDF value.

    Raises
    ------
    ValueError
        If `data` is not equivalent to a 1-d vector.

    """
    if len(set(np.shape(dat_vector)).difference({1})) > 1:
        raise ValueError("`data` is not equivalent to a 1-d vector. ")

    det_divider = np.log(np.abs(np.linalg.det(cov_matrix)))
    chisq_exponent = np.transpose(np.conj(dat_vector)) \
        @ np.linalg.inv(cov_matrix) \
        @ dat_vector

    log_prob_density = - det_divider - np.real(chisq_exponent)

    return log_prob_density


def _f_nl_parametrised_covariance(f_nl, b_const, nbar, two_point_model, pivot):
    r"""Parametrised covariance matrix by local primordial non-Gaussianity.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussianity parameter.
    b_const : float
        Constant linear bias of the tracer particles at the current epoch.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function model without scale modification.
    pivot : {'natural', 'transposed', 'spectral'}
        Pivot axis for unpacking indexed data into a 1-d vector.

    Returns
    -------
    covariance : complex or float :class:`numpy.ndarray`
        Covariance matrix value.

    See Also
    --------
    :class:`~harmonia.reader.spherical_model.TwoPointFunction`

    """
    covariance = two_point_model.two_point_covariance(
        pivot,
        diag=True,  # HACK: temporary speed-up
        nbar=nbar,
        b_const=b_const,
        f_nl=f_nl
    )

    return covariance


def _f_nl_parametrised_variance(f_nl, b_const, nbar, two_point_model, pivot):
    r"""Parametrised diagonal covariance matrix by local primordial
    non-Gaussianity.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussianity parameter.
    b_const : float
        Constant linear bias of the tracer particles at the current epoch.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function model without scale modification.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.

    Returns
    -------
    variance : float :class:`numpy.ndarray`
        Diagonal covariance matrix.

    See Also
    --------
    :class:`~harmonia.reader.spherical_model.TwoPointFunction`

    """
    variance = two_point_model.mode_variance(
        pivot,
        nbar=nbar,
        b_const=b_const,
        f_nl=f_nl
    )

    return np.diag(variance)


def spherical_map_f_nl_chi_square(sample_parameters, data_vector, pivot,
                                  two_point_model, nbar, bias,
                                  by_mode=False):
    """Evaluate the spherical map chi-square of the local non-Gaussianity
    parameter.

    The data vector is assumed to be zero-centred.

    Parameters
    ----------
    sample_parameters : float, array_like
        Sampling values of the local non-Gaussnaity parameter.
    data_vector : float or complex, array_like
        1-d data vector centred at zero.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function model without scale modification.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    bias : float
        Constant linear bias of the tracer particles at the current
        epoch.

    Returns
    -------
    sampled_chisq : float :class:`numpy.ndarray`
        Chi-square evaluated at the sample parameters.

    Raises
    ------
    ValueError
        If `sample_parameters` is not equivalent to a 1-d vector.

    """
    if len(set(np.shape(sample_parameters)).difference({1})) > 1:
        raise ValueError(
            "`sample_parameters` is not equivalent to a flat array. "
        )

    data_vector = data_vector / _OVERFLOW_DOWNSCALE

    if by_mode:
        sampled_chisq = np.zeros((len(sample_parameters), len(data_vector)))
    else:
        sampled_chisq = np.zeros(len(sample_parameters))
    for idx, parameter in enumerate(sample_parameters):
        sample_covar = _f_nl_parametrised_variance(
            parameter,
            bias,
            nbar,
            two_point_model,
            pivot
        )

        sample_covar = sample_covar / _OVERFLOW_DOWNSCALE**2

        sampled_chisq[idx] = _chi_square(
            data_vector, sample_covar, by_element=by_mode
        )

    return sampled_chisq


def spherical_map_f_nl_likelihood(sample_parameters, data_vector, pivot,
                                  two_point_model, nbar, bias):
    """Evaluate the spherical map likelihood of the local non-Gaussianity
    parameter.

    The data vector is assumed to be zero-centred.

    Parameters
    ----------
    sample_parameters : float, array_like
        Sampling values of the local non-Gaussnaity parameter.
    data_vector : float or complex, array_like
        1-d data vector.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function model without scale modification.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    bias : float
        Constant linear bias of the tracer particles at the current
        epoch.

    Returns
    -------
    sampled_likelihood : float :class:`numpy.ndarray`
        Likelihood evaluated at the sample parameters.

    Raises
    ------
    ValueError
        If `sample_parameters` is not equivalent to a flat array.

    """
    if len(set(np.shape(sample_parameters)).difference({1})) > 1:
        raise ValueError(
            "`sample_parameters` is not equivalent to a flat array. "
        )

    data_vector = data_vector / _OVERFLOW_DOWNSCALE

    sampled_likelihood = np.zeros(len(sample_parameters))
    for idx, parameter in enumerate(sample_parameters):
        sample_covar = _f_nl_parametrised_variance(  # not ..._variance
            parameter,
            bias,
            nbar,
            two_point_model,
            pivot
        )
        sample_covar = sample_covar / _OVERFLOW_DOWNSCALE**2
        sampled_likelihood[idx] = \
            _log_complex_normal_pdf(data_vector, sample_covar)

    return sampled_likelihood
