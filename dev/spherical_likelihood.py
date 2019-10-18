"""Spherical likelihood constructed from data realisation and covariance
modelling.

"""
import numpy as np

from harmonia.reader import TwoPointFunction


def _f_nl_parametrised_covariance(f_nl, pivot, *twopt_args, **twopt_kwargs):
    r"""Parametrised covariance matrix by local primordial non-Gaussianity.

    Parameters
    ----------
    f_nl : float
        Local primordial non-Gaussianity parameter.
    pivot : {‘natural’, ‘scale’, ‘lmn’, ‘lnm’, ‘nlm’, ‘ln’, ‘k’}
        Pivot axis for unpacking index data into a 1-d vector.
    *twopt_args
        Positional arguments to be passed to :class:`~.TwoPointFunction`
        such as `disc`, `nbar`, `b_1`.
    **twopt_kwargs
        Keyword arguments to be passed to :class:`~.TwoPointFunction` such
        as either `f_0` plus `power_spectrum` or `cosmo`, pre-computed
        `couplings`, specifications such as `survey_specs` or `cosmo_specs`
        and MPI communicator `comm`.

    Returns
    -------
    covariance : complex or float, array_like
        Covariance matrix evaluated at input parameter(s).

    See Also
    --------
    :class:`~harmonia.reader.spherical_model.TwoPointFunction`


    .. todo::

        `f_nl` is the only varying cosmological parameter, so currently
        `couplings` are provided and MPI `comm` is disabled, with redundant
        `survey_specs` and `cosmo_specs`.

    """
    two_point_model = TwoPointFunction(*twopt_args, f_nl=f_nl, **twopt_kwargs)

    covariance = two_point_model.two_point_covariance(pivot, diag=True)

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

    dat_vector /= 1e4
    cov_matrix /= 1e4 ** 2

    det_divider = data_dim * np.log(np.pi) \
        + np.log(np.abs(np.linalg.det(cov_matrix)))

    chisq_exponent = np.real(
        np.transpose(np.conj(dat_vector)) @ np.linalg.inv(cov_matrix) \
            @ dat_vector
    )

    log_prob_density = - chisq_exponent - det_divider

    return log_prob_density


def spherical_map_likelihood_f_nl(sample_parameters, data_vector, pivot,
                                  *twopt_args, **twopt_kwargs):
    """Evaluate the spherical map likelihood of the local non-Gaussianity
    parameter.

    Parameters
    ----------
    sample_parameters : float, array_like
        Sampling values of the local non-Gaussnaity parameter.
    data_vector : float or complex, array_like
        1-d data vector.

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
            *twopt_args,
            **twopt_kwargs
        )
        sampled_likelihood[idx] = \
            _log_complex_normal_pdf(data_vector, _sample_covar)

    return sampled_likelihood
