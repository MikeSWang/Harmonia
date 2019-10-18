"""Spherical likelihood constructed from data realisation and covariance
modelling.

"""
import numpy as np


def _f_nl_parametrised_covariance(f_nl, pivot, two_point_model):
    r"""Parametrised covariance matrix by local primordial non-Gaussianity.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussianity parameter.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function model without scale modification yet.

    Returns
    -------
    covariance : complex or float :class:`numpy.ndarray`
        Covariance matrix value.

    See Also
    --------
    :class:`~harmonia.reader.spherical_model.TwoPointFunction`

    """
    covariance = \
        two_point_model.two_point_covariance(pivot, diag=True, f_nl=f_nl)

    return covariance


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
    data_dim = len(dat_vector)

    det_divider = data_dim * np.log(np.pi) \
        + np.log(np.abs(np.linalg.det(cov_matrix)))

    chisq_exponent = np.transpose(np.conj(dat_vector)) \
        @ np.linalg.inv(cov_matrix) \
        @ dat_vector

    log_prob_density = - det_divider - np.real(chisq_exponent)

    return log_prob_density


def _f_nl_parametrised_chi_square(sample_parameters, dat_vector, pivot,
                                  two_point_model):
    """Parametrised chi-square value by local primordial non-Gaussianity.

    The data vector is assumed to be zero-centred.

    Parameters
    ----------
    sample_parameters : float, array_like
        Sampling values of the local non-Gaussnaity parameter.
    dat_vector : float or complex, array_like
        1-d data vector centred at zero.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function model without scale modification yet.

    Returns
    -------
    sampled_chisq : float
        Chi-square evaluated at the sample parameters.

    Raises
    ------
    ValueError
        If `data` is not equivalent to a 1-d vector.

    """
    if len(set(np.shape(dat_vector)).difference({1})) > 1:
        raise ValueError("`data` is not equivalent to a 1-d vector. ")

    sampled_chisq = np.zeros(len(sample_parameters))
    for idx, parameter in enumerate(sample_parameters):
        _sample_covar = _f_nl_parametrised_covariance(
            parameter,
            pivot,
            two_point_model
        )
        sampled_chisq[idx] = np.real(
            np.transpose(np.conj(dat_vector))
            @ np.linalg.inv(_sample_covar)
            @ dat_vector
        )

    return sampled_chisq


def spherical_map_f_nl_likelihood(sample_parameters, data_vector, pivot,
                                  two_point_model):
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
        2-point function model without scale modification yet.

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

    sampled_likelihood = np.zeros(len(sample_parameters))
    for idx, parameter in enumerate(sample_parameters):
        _sample_covar = _f_nl_parametrised_covariance(
            parameter,
            pivot,
            two_point_model
        )
        sampled_likelihood[idx] = \
            _log_complex_normal_pdf(data_vector, _sample_covar)

    return sampled_likelihood
