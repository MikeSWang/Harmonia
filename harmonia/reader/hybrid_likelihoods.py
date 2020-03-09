"""
Hybrid likelihoods (:mod:`~harmonia.reader.hybrid_likelihoods`)
===========================================================================

Build hybrid likelihoods for cosmological parameter inference.


Probability distributions
---------------------------------------------------------------------------

.. autosummary::

    complex_normal_pdf
    multivariate_normal_pdf
    modified_student_pdf


Spherical likelihood
---------------------------------------------------------------------------

.. autosummary::

    spherical_parametrised_covariance
    spherical_map_log_likelihood


Cartesian likelihood
---------------------------------------------------------------------------

.. autosummary::

    cartesian_parametrised_moments
    cartesian_map_log_likelihood

"""
import collections
import warnings
from itertools import product

import numpy as np
from scipy.special import loggamma

from harmonia.algorithms.arrays import CartesianArray, SphericalArray
from harmonia.collections.utils import PositiveDefinitenessWarning
from harmonia.collections.utils import (
    check_positive_definiteness,
    ensure_positive_definiteness,
)
from harmonia.collections.utils import mat_logdet, mpi_compute, progress_status


class LikelihoodWarning(UserWarning):
    """Likelihood evaluation warning.

    """
    pass


# Probability distributions
# -----------------------------------------------------------------------------

def _chi_square(dat_vector, cov_matrix, elementwise=False):
    """Calculate chi-square from zero-centred data vector and covariance
    matrix.

    Parameters
    ----------
    dat_vector : complex, array_like
        Zero-centred data vector.
    cov_matrix : complex, array_like
        Covariance matrix.
    elementwise : bool, optional
        If `True` (default is `False`), return elementwise contribution to
        chi-square from data vector component.  This implicitly neglects
        the off-diagonal part of the covariance matrix.

    Returns
    -------
    chi_sq : float :class:`numpy.ndarray`
        Chi-square value(s).

    Raises
    ------
    ValueError
        If `dat_vector` is not equivalent to a 1-d vector or `cov_matrix`
        is not equivalent to a 2-d matrix.

    """
    if np.squeeze(dat_vector).ndim != 1:
        raise ValueError("`dat_vector` is not equivalent to a 1-d vector. ")
    if np.squeeze(cov_matrix).ndim != 2:
        raise ValueError("`cov_matrix` is not equivalent to a 2-d matrix. ")

    if elementwise:
        chi_sq = np.abs(dat_vector)**2 / np.real(np.diag(cov_matrix))
    else:
        chi_sq = np.real(
            np.conj(dat_vector).T @ np.linalg.inv(cov_matrix) @ dat_vector
        )

    return chi_sq


def complex_normal_pdf(dat_vector, cov_matrix, return_log=True, downscale=None,
                       elementwise=False):
    """Compute the complex normal probability density function or its
    natural logarithm given the zero-centred data vector and its
    covariance matrix.

    Parameters
    ----------
    dat_vector : complex, array_like
        Zero-centred data vector.
    cov_matrix : complex, array_like
        Covariance matrix.
    return_log : bool, optional
        If `True` (default), return logarithmic probability density.
    downscale : float or None, optional
        If not `None` (default), the data vector and covariance matrix are
        simultaneous downscaled in computing the exponent and matrix
        determinant normalisation.
    elementwise : bool, optional
        If `True` (default is `False`), elementwise contributions from each
        data vector component to the (logarithmic) probability density are
        returned.  This implicitly neglects the off-diagonal part of the
        covariance matrix.

    Returns
    -------
    density : float, array_like
        Logarithmic probability density value.

    """
    dat_vector, cov_matrix = np.squeeze(dat_vector), np.squeeze(cov_matrix)
    if cov_matrix.ndim == 1:
        cov_matrix = np.diag(cov_matrix)
    if np.size(dat_vector)**2 != np.size(cov_matrix):
        raise ValueError(
            "`data_vector` and `cov_matrix` dimensions are not consistent: "
            "{} and {}. ".format(np.size(dat_vector), np.shape(cov_matrix))
        )

    dat_dim = np.size(dat_vector)

    log_normalisation_const = dat_dim * np.log(np.pi)
    if downscale is not None:
        dat_vector = dat_vector / downscale
        cov_matrix = cov_matrix / downscale**2
        log_normalisation_const -= 2 * dat_dim * np.log(downscale)

    if not check_positive_definiteness(cov_matrix):
        warnings.warn(
            "`cov_matrix` fails positive definiteness. ",
            PositiveDefinitenessWarning
        )
        cov_matrix, ensured = ensure_positive_definiteness(
            cov_matrix, tweak_param=1e-4, maxiter=5
        )
        warnings.warn(
            "`cov_matrix` is modified to ensure positive definiteness. ",
            PositiveDefinitenessWarning
        )
        if not ensured:
            warnings.warn(
                "Modified `cov_matrix` still fails positive definiteness. ",
                PositiveDefinitenessWarning
            )

    if elementwise:
        log_det_cov_mat = np.log(np.abs(np.diag(cov_matrix)))
    else:
        log_det_cov_mat = mat_logdet(cov_matrix)

    chi_sq = _chi_square(dat_vector, cov_matrix, elementwise=elementwise)

    density = - log_normalisation_const - log_det_cov_mat - chi_sq

    if not return_log:
        density = np.exp(density)

    return density


def multivariate_normal_pdf(data_vector, mean_vector, cov_matrix,
                            return_log=True):
    """Compute the multivariate normal probability density function or its
    natural logarithm given the data vector, its mean vector and covariance
    matrix.

    Parameters
    ----------
    data_vector : float, array_like
        Data vector.
    mean_vector : float, array_like
        Mean vector.
    cov_matrix : float, array_like
        Covariance matrix.
    return_log : bool, optional
        If `True` (default), return logarithmic probability density.

    Returns
    -------
    density : float, array_like
        Logarithmic probability density value.

    Raises
    ------
    ValueError
        If the dimensions of `data_vector`, `mean_vector` and `cov_matrix`
        are not consistent.

    """
    data_vector, mean_vector, cov_matrix = np.squeeze(data_vector), \
        np.squeeze(mean_vector), np.squeeze(cov_matrix)
    if cov_matrix.ndim == 1:
        cov_matrix = np.diag(cov_matrix)
    if np.size(data_vector) != np.size(data_vector) \
            or np.size(data_vector)**2 != np.size(cov_matrix):
        raise ValueError(
            "Dimensions of `data_vector`, `mean_vector` and `cov_matrix` "
            "are not consistent: {}, {} and {}. ".format(
                np.size(data_vector),
                np.size(mean_vector),
                np.shape(cov_matrix),
            )
        )

    dat_dim = np.size(data_vector)

    log_normalisation_const = dat_dim * np.log(2*np.pi)

    log_det_cov_mat = mat_logdet(cov_matrix)

    exponent = _chi_square(
        data_vector - mean_vector, cov_matrix, elementwise=False
    )

    density = 1/2 * (- log_normalisation_const - log_det_cov_mat - exponent)

    if not return_log:
        density = np.exp(density)

    return density


def modified_student_pdf(data_vector, mean_vector, cov_matrix, num_sample,
                         return_log=True):
    """Compute the multivariate modified Student probability density
    function or its natural logarithm given the data vector, its mean
    vector and covariance matrix.

    Parameters
    ----------
    data_vector : float, array_like
        Data vector.
    mean_vector : float, array_like
        Mean vector.
    cov_matrix : float, array_like
        Covariance matrix.
    num_sample : int
        Number of samples used in estimating the covariance matrix.
    return_log : bool, optional
        If `True` (default), return logarithmic probability density.

    Returns
    -------
    density : float, array_like
        (Logarithmic) probability density value.

    Raises
    ------
    ValueError
        If the dimensions of `data_vector`, `mean_vector` and `cov_matrix`
        are not consistent.

    """
    data_vector, mean_vector, cov_matrix = np.squeeze(data_vector), \
        np.squeeze(mean_vector), np.squeeze(cov_matrix)
    if cov_matrix.ndim == 1:
        cov_matrix = np.diag(cov_matrix)
    if np.size(data_vector) != np.size(data_vector) \
            or np.size(data_vector)**2 != np.size(cov_matrix):
        raise ValueError(
            "Dimensions of `data_vector`, `mean_vector` and `cov_matrix` "
            "are not consistent: {}, {} and {}. ".format(
                np.size(data_vector),
                np.size(mean_vector),
                np.shape(cov_matrix),
            )
        )

    dat_dim = np.size(data_vector)

    log_normalisation_const = - dat_dim/2 * np.log((num_sample - 1)*np.pi) \
        + loggamma(num_sample/2) - loggamma((num_sample -dat_dim)/2)

    log_det_cov_mat = mat_logdet(cov_matrix)

    core = np.log(
        1 + _chi_square(
            data_vector - mean_vector, cov_matrix, elementwise=False
        ) / (num_sample - 1)
    )

    density = log_normalisation_const - (log_det_cov_mat + num_sample*core)/2

    if not return_log:
        density = np.exp(density)

    return density


# Spherical likelihood
# -----------------------------------------------------------------------------

def spherical_parametrised_covariance(b_1, f_nl, two_point_model, pivot,
                                      independence=False, diag=False,
                                      **model_kwargs):
    r"""Compute the parametrised covariance matrix given the spherical
    2-point function base model and the pivot axis for data vectorisation
    of transformed fields.

    Parameters
    ----------
    b_1 : float
        Scale-independent linear bias of the tracer particles.
    f_nl : float or None
        Local primordial non-Gaussianity.
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function base model.
    pivot : {'natural', 'transposed', 'spectral'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    independence : bool, optional
        If `True` (default is `False`), independence amongst Fourier modes
        is assumed and the diagonal covariance matrix is computed without
        coupling coefficients.
    diag : bool, optional
        If `True` (default is `False`), only the diagonal elements of the
        covariance matrix are computed by summation against of coupling
        coefficients.
    **model_kwargs
        Keyword arguments to be passed to `two_point_model` methods
        :meth:`~.spherical_model.TwoPointFunction.two_point_covariance`
        or :meth:`~.spherical_model.TwoPointFunction.mode_variance` (if
        `independence` is `True`).

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
            pivot, b_1=b_1, f_nl=f_nl, **model_kwargs
        )
        covariance = np.diag(variance)
    else:
        covariance = two_point_model.two_point_covariance(
            pivot, b_1=b_1, f_nl=f_nl, diag=diag, **model_kwargs
        )

    return covariance


def spherical_map_log_likelihood(bias, non_gaussianity, mean_number_density,
                                 two_point_model, spherical_data, pivot,
                                 breakdown=False, exclude_degrees=(),
                                 logger=None, comm=None, **covariance_kwargs):
    """Evaluate the spherical map logarithmic likelihood.

    Parameters
    ----------
    bias : float, array_like
        Scale-independent linear bias of the tracer particles.
    non_gaussianity : float, array_like or None
        Local primordial non-Gaussianity.
    mean_number_density : float
        Mean particle number density (in cubic :math:`h`/Mpc).
    two_point_model : |two_point_model| , array_like
        2-point function base model.
    spherical_data : :class:`~harmonia.algorithms.arrays.SphericalArray`
        Spherical data array of the transformed field.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    logger : :class:`logging.Logger` or None, optional
        Logger (default is `None`).
    comm : :class:`mpi4py.MPI.Comm` or None, optional
        MPI communicator (default is `None`).

    Other parameters
    ----------------
    breakdown : bool, optional
        If `True` (default is `False`), the contribution from each data
        vector component is broken down assuming a diagonal covariance
        matrix.
    exclude_degrees : tuple of int, optional
        If not ``()`` (default), modes whose spherical degree is an
        element are removed from the likelihood.
    **covariance_kwargs
        Keyword arguments to be passed to
         :func:`~.hybrid_likelihoods.spherical_parametrised_covariance`.

    Returns
    -------
    log_likelihood : float :class:`numpy.ndarray`
        Logarithmic likelihood evaluated at the parameter sampling points.


    .. |two_point_model| replace::

        :class:`~.spherical_model.TwoPointFunction`

    """
    _OVERFLOW_DOWNSCALE = 10**4

    data_vector = spherical_data.unfold(pivot, return_only='data')

    mode_indices = SphericalArray\
        .build(disc=two_point_model.disc)\
        .unfold(pivot, return_only='index')

    excluded_deg = np.fromiter(
        map(lambda index: index[0] in exclude_degrees, mode_indices),
        dtype=bool
    )

    data_vector = data_vector[~excluded_deg]

    axis_to_squeeze = ()
    if not isinstance(bias, collections.Iterable):
        bias = (bias,)
        axis_to_squeeze += (0,)
    if not isinstance(non_gaussianity, collections.Iterable):
        non_gaussianity = (non_gaussianity,)
        axis_to_squeeze += (1,)
    if not isinstance(two_point_model, collections.Iterable):
        two_point_model = (two_point_model,)
        axis_to_squeeze += (2,)

    out_shape = (len(bias), len(non_gaussianity), len(two_point_model))
    if breakdown:
        out_shape += (len(data_vector),)

    sampled_points = list(product(bias, non_gaussianity, two_point_model))

    def _likelihood_eval(sample_point):

        b_1, f_nl, tpm = sample_point

        sample_covar = spherical_parametrised_covariance(
            b_1, f_nl, tpm, pivot,
            nbar=mean_number_density,
            **covariance_kwargs
        )[:, ~excluded_deg][~excluded_deg, :]

        sample_likelihood = complex_normal_pdf(
            data_vector, sample_covar,
            downscale=_OVERFLOW_DOWNSCALE,
            elementwise=breakdown
        )

        return sample_likelihood

    with warnings.catch_warnings(record=True) as captured_warnings:
        if comm is None:
            log_likelihood = []
            for idx, sample_point in enumerate(sampled_points):
                log_likelihood.append(_likelihood_eval(sample_point))
                progress_status(
                    idx, len(sampled_points), logger,
                    process_name=f"spherical likelihood evaluation"
                )
        else:
            log_likelihood = mpi_compute(
                sampled_points, _likelihood_eval, comm,
                logger=logger, process_name="spherical likelihood evaluation"
            )

    unique_warning_msgs = set(map(
        lambda warning_obj: warning_obj.message, captured_warnings
    ))
    for msg in unique_warning_msgs:
        if comm is None or comm.rank == 0:
            warnings.warn(msg, LikelihoodWarning)

    log_likelihood = np.reshape(log_likelihood, out_shape)
    if axis_to_squeeze:
        log_likelihood = np.squeeze(log_likelihood, axis=axis_to_squeeze)

    return log_likelihood


# Cartesian likelihood
# -----------------------------------------------------------------------------

def cartesian_parametrised_moments(b_1, f_nl, windowed_power_model, pivot,
                                   orders, correlation_modeller,
                                   **model_kwargs):
    """Compute the parametrised moment(s) of power spectrum multipoles.

    Parameters
    ----------
    b_1 : float
        Scale-independent linear bias of the tracer particles.
    f_nl : float or None
        Local primordial non-Gaussianity.
    nbar : float
        Mean particle number density (in cubic :math:`h`/Mpc).
    windowed_power_model : :class:`~.cartesian_model.WindowedPowerSpectrum`
        Windowed power spectrum base model.
    pivot : {'order', 'scale'}
        Order in which the data is unpacked as a 1-d vector.
    orders : list of int
        Order(s) of the power spectrum multipoles.
    correlation_modeller : :class:`~.cartesian_model.WindowedCorrelation`
        Window-induced correlation modeller.  Must be pivoted at `pivot`,
        i.e. its :attr:`pivot` attribute must agree with input `pivot`.
    **model_kwargs
        Keyword arguments to be passed to `windowed_power_model`.

    Returns
    -------
    expectation : float :class:`numpy.ndarray`
        Power spectrum expectation at the specified wavenumbers.
    covariance : float :class:`numpy.ndarray`
        Power spectrum variance at the specified wavenumbers.

    Raises
    ------
    AssertionError
        If the :attr:`wavenumbers` attribute of `windowed_power_model`
        does not match the wavenumbers of `correlation_modeller`.

    """
    fiducial_wavenumbers = correlation_modeller.fiducial_multipoles.coord_array
    fiducial_expectation = correlation_modeller.fiducial_vector(pivot)
    fiducial_covariance = correlation_modeller.windowed_correlation

    assert np.shape(np.squeeze(fiducial_wavenumbers)) \
        == np.shape(np.squeeze(windowed_power_model.wavenumbers)), (
            "The wavenumbers at which the power spectrum model is evaluated "
            "do not match the wavenumbers at which "
            "the fiducial covariance matrix is estimated. "
        )
    assert len(orders) == len(correlation_modeller.orders), (
        "The multipoles for which the fiducial covariance matrix is "
        "estimated do not match the multipoles for which "
        "the power spectrum model is evaluated. "
    )

    expectation_filling = windowed_power_model.convolved_multipoles(
        orders, b_1, f_nl=f_nl, **model_kwargs
    )

    expectation_array = CartesianArray(
        filling=expectation_filling,
        coord_key='k',
        variable_key_root='power_'
    )

    if pivot == 'order':
        expectation = expectation_array.unfold('variable', return_only='data')
    if pivot == 'scale':
        expectation = expectation_array.unfold('coord', return_only='data')

    covariance = np.diag(expectation/fiducial_expectation) \
        @ fiducial_covariance \
        @ np.diag(expectation/fiducial_expectation)

    return expectation, covariance


def cartesian_map_log_likelihood(bias, non_gaussianity, mean_number_density,
                                 windowed_power_model, cartesian_data,
                                 correlation_modeller, pivot, orders,
                                 num_covar_sample=None, logger=None, comm=None,
                                 **covariance_kwargs):
    """Evaluate the Cartesian map logarithmic likelihood.

    Parameters
    ----------
    bias : float, array_like
        Scale-independent linear bias of the tracer particles.
    non_gaussianity : float, array_like or None
        Local primordial non-Gaussianity.
    mean_number_density : float
        Mean particle number density (in cubic :math:`h`/Mpc).
    windowed_power_model : :class:`~.cartesian_model.WindowedPowerSpectrum`
        Windowed power spectrum base model.
    cartesian_data : :class:`~harmonia.algorithms.arrays.CartesianArray`
        Cartesian data array of the compressed field.
    correlation_modeller : :class:`~.cartesian_model.WindowedCorrelation`
        Window-induced correlation modeller.  Must be pivoted at `pivot`,
        i.e. its :attr:`pivot` attribute must agree with input `pivot`.
    pivot : {'order', 'scale'}
        Order in which the data is unpacked as a 1-d vector.
    orders : list of int
        Order(s) of the power spectrum multipoles to include.
    num_covar_sample : int or None, optional
        If not `None` (default), this is the number of sampled used in
        estimating the covariance and the modified Student distribution is
        used.
    logger : :class:`logging.Logger` or None, optional
        Logger (default is `None`).
    comm : :class:`mpi4py.MPI.Comm` or None, optional
        MPI communicator (default is `None`).
    **covariance_kwargs
        Keyword arguments to be passed to
        :func:`~.reader.hybrid_likelihoods.cartesian_parametrised_moments`.

    Returns
    -------
    log_likelihood : float :class:`numpy.ndarray`
        Logarithmic likelihood evaluated at the parameter sampling points.

    """
    if pivot == 'order':
        data_vector = cartesian_data.unfold('variable', return_only='data')
    if pivot == 'scale':
        data_vector = cartesian_data.unfold('coord', return_only='data')

    axis_to_squeeze = ()
    if not isinstance(bias, collections.Iterable):
        bias = (bias,)
        axis_to_squeeze += (0,)
    if not isinstance(non_gaussianity, collections.Iterable):
        non_gaussianity = (non_gaussianity,)
        axis_to_squeeze += (1,)
    if not isinstance(windowed_power_model, collections.Iterable):
        windowed_power_model = (windowed_power_model,)
        axis_to_squeeze += (2,)

    out_shape = (len(bias), len(non_gaussianity), len(windowed_power_model))
    sampled_points = list(product(bias, non_gaussianity, windowed_power_model))

    def _likelihood_eval(sample_point):

        b_1, f_nl, wpm = sample_point

        sample_mean, sample_covar = cartesian_parametrised_moments(
            b_1, f_nl, wpm, pivot, orders, correlation_modeller,
            nbar=mean_number_density, **covariance_kwargs
        )

        if num_covar_sample:
            sample_likelihood = modified_student_pdf(
                data_vector, sample_mean, sample_covar,
                num_sample=num_covar_sample
            )
        else:
            sample_likelihood = multivariate_normal_pdf(
                data_vector, sample_mean, sample_covar
            )

        return sample_likelihood

    if comm is None:
        log_likelihood = []
        for idx, sample_point in enumerate(sampled_points):
            log_likelihood.append(_likelihood_eval(sample_point))
            progress_status(
                idx, len(sampled_points), logger,
                process_name=f"cartesian likelihood evaluation"
            )
    else:
        log_likelihood = mpi_compute(
            sampled_points, _likelihood_eval, comm,
            logger=logger, process_name="cartesian likelihood evaluation"
        )

    log_likelihood = np.reshape(log_likelihood, out_shape)
    if axis_to_squeeze:
        log_likelihood = np.squeeze(log_likelihood, axis=axis_to_squeeze)

    return log_likelihood
