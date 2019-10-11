"""Spherical likelihood constructed from data realisation and covariance
modelling.

"""
import numpy as np

from harmonia.reader import TwoPointFunction


def _f_nl_parametrised_covariance(f_nl, pivot, *twopt_args, **twopt_kwargs):
    r"""Parametrised covariance matrix by local primordial non-Gaussianity.

    Parameters
    ----------
    f_nl : float, array_like
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

    covariance = two_point_model.two_point_covariance(pivot)

    return covariance


def _complex_normal_pdf(dat_vector, cov_matrix):
    """Compute the value of the complex normal probability density function
    at specified data vector and covariance.

    The data vector is assumed to be zero-centred.

    Parameters
    ----------
    dat_vector : float or complex, array_like
        1-d data vector centred at zero.
    cov_matrix : float or complex, array_like
        2-d covariance matrix.

    Returns
    -------
    prob_density : float
        PDF value.

    Raises
    ------
    ValueError
        If `data` is not equivalent to a 1-d vector.

    """
    if len(set(np.shape(dat_vector)).difference({1})) > 1:
        raise ValueError("`data` is not equivalent to a 1-d vector. ")
    data_dim = len(dat_vector)

    det_divider = np.pi**data_dim * np.linalg.det(cov_matrix)

    chisq_exponent = np.transpose(np.conj(dat_vector)) \
        @ np.linalg.inv(cov_matrix) @ dat_vector

    prob_density = np.exp(- chisq_exponent) / det_divider

    return prob_density


def spherical_map_likelihood_f_nl(sample_parameter, data_vector, pivot,
                                  *twopt_args, **twopt_kwargs):
    """Evaluate the spherical map likelihood of the local non-Gaussianity
    parameter.

    Parameters
    ----------
    sample_parameter : float, array_like
        Sampling values of the local non-Gaussnaity parameter.
    data_vector : float or complex, array_like
        1-d data vector.

    Returns
    -------
    sampled_likelihood : float, array_like
        Likelihood evaluated at the sample parameters.

    """
    _sample_covar = _f_nl_parametrised_covariance(
        sample_parameter,
        pivot,
        *twopt_args,
        **twopt_kwargs
    )

    sampled_likelihood = _complex_normal_pdf(data_vector, _sample_covar)

    return sampled_likelihood
