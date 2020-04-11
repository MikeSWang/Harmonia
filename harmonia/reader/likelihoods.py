"""
Cosmological likelihoods (:mod:`~harmonia.reader.likelihoods`)
===========================================================================

Construct likelihoods from Spherical Fourier coefficients and Cartesian
power spectrum multipoles of random fields for cosmological parameter
inference.


Probability distributions
---------------------------------------------------------------------------

.. autosummary::

    complex_normal_pdf
    multivariate_normal_pdf
    modified_student_pdf


Likelihood inference
---------------------------------------------------------------------------

.. autosummary::

    spherical_covariance
    cartesian_moments
    LogLikelihood

|

.. |correlator_matrix| replace::

    :meth:`~harmonia.reader.models.SphericalCorrelator.correlator_matrix`

.. |convolved_power_multipoles| replace::

    :meth:`~.reader.models.CartesianMultipoles.convolved_power_multipoles`

"""
import logging
import warnings

# pylint: disable=no-name-in-module
import numpy as np
from scipy.special import loggamma

from harmonia.utils import (
    PositiveDefinitenessWarning,
    is_positive_definite,
    mat_logdet,
)


# Probability distributions
# -----------------------------------------------------------------------------

def _are_valid_moments(first_moment, second_moment):

    criterion1 = (np.squeeze(first_moment).ndim == 1)
    criterion2 = (np.squeeze(second_moment).ndim == 2)
    criterion3 = (np.shape(first_moment) * 2 == np.shape(second_moment))

    return all([criterion1, criterion2, criterion3])


def chi_square(data_vector, covariance_matrix):
    """Calculate chi-square from zero-centred data vector and its
    covariance matrix.

    Parameters
    ----------
    data_vector : complex, array_like
        Zero-centred data vector.
    covariance_matrix : complex, array_like
        Covariance matrix.

    Returns
    -------
    chi_sq : float :class:`numpy.ndarray`
        Chi-square value.

    """
    data_vector = np.squeeze(data_vector)
    covariance_matrix = np.squeeze(covariance_matrix)
    if not _are_valid_moments(data_vector, covariance_matrix):
        raise ValueError("Check input dimensions.")

    chi_sq = np.real(
        np.conj(data_vector).T @ np.linalg.inv(covariance_matrix) @ data_vector
    )

    return chi_sq


def complex_normal_pdf(data_vector, covariance_matrix, ret_log=True,
                       downscale=None):
    """Compute the complex normal probability density function or its
    natural logarithm given the zero-centred data vector and its
    covariance matrix.

    Parameters
    ----------
    data_vector : complex, array_like
        Zero-centred data vector.
    covariance_matrix : complex, array_like
        Covariance matrix.
    ret_log : bool, optional
        If `True` (default), return logarithmic probability density.
    downscale : float or None, optional
        If not `None` (default), the data vector and covariance matrix are
        simultaneous downscaled to avoid numerical issue.

    Returns
    -------
    float, array_like
        (Logarithmic) probability density.

    """
    data_vector = np.squeeze(data_vector)
    covariance_matrix = np.squeeze(covariance_matrix)
    if not _are_valid_moments(data_vector, covariance_matrix):
        raise ValueError("Check input dimensions.")

    dim = np.size(data_vector)

    log_normalisation = dim * np.log(np.pi)
    if downscale is not None:
        data_vector = data_vector / downscale
        covariance_matrix = covariance_matrix / downscale ** 2
        log_normalisation -= 2 * dim * np.log(downscale)

    if not is_positive_definite(covariance_matrix):
        warnings.warn(
            "`covariance_matrix` is not positive definite.",
            PositiveDefinitenessWarning
        )

    density = \
        - log_normalisation \
        - mat_logdet(covariance_matrix) \
        - chi_square(data_vector, covariance_matrix)

    return density if ret_log else np.exp(density)


def multivariate_normal_pdf(data_vector, expectation_vector, covariance_matrix,
                            ret_log=True):
    """Compute the multivariate normal probability density function or its
    natural logarithm given the data vector, its mean vector and covariance
    matrix.

    Parameters
    ----------
    data_vector : float, array_like
        Data vector.
    expectation_vector : float, array_like
        Mean vector.
    covariance_matrix : float, array_like
        Covariance matrix.
    ret_log : bool, optional
        If `True` (default), return logarithmic probability density.

    Returns
    -------
    density : float, array_like
        Logarithmic probability density value.

    """
    data_vector = np.squeeze(data_vector)
    expectation_vector = np.squeeze(expectation_vector)
    covariance_matrix = np.squeeze(covariance_matrix)
    if not _are_valid_moments(data_vector, covariance_matrix) or \
            not _are_valid_moments(expectation_vector, covariance_matrix):
        raise ValueError("Check input dimensions.")

    dim = np.size(data_vector)

    log_normalisation = dim * np.log(2*np.pi)

    log_determinant = mat_logdet(covariance_matrix)

    chi_sq = chi_square(data_vector - expectation_vector, covariance_matrix)

    density = - (log_normalisation + log_determinant + chi_sq) / 2

    return density if ret_log else np.exp(density)


def modified_student_pdf(data_vector, expectation_vector, covariance_matrix,
                         degree, ret_log=True):
    """Compute the multivariate modified Student probability density
    function or its natural logarithm given the data vector, its mean
    vector and covariance matrix.

    Parameters
    ----------
    data_vector : float, array_like
        Data vector.
    expectation_vector : float, array_like
        Mean vector.
    covariance_matrix : float, array_like
        Covariance matrix.
    degree : int
        The degree number.  This could be the number of empirical
        covariance matrices used to obtain the estimated
        `covariance_matrix`.
    ret_log : bool, optional
        If `True` (default), return logarithmic probability density.

    Returns
    -------
    float, array_like
        (Logarithmic) probability density value.

    """
    data_vector = np.squeeze(data_vector)
    expectation_vector = np.squeeze(expectation_vector)
    covariance_matrix = np.squeeze(covariance_matrix)
    if not _are_valid_moments(data_vector, covariance_matrix) \
            or not _are_valid_moments(expectation_vector, covariance_matrix):
        raise ValueError("Check input dimensions.")

    dim = np.size(data_vector)

    log_normalisation = \
        + dim / 2. * np.log((degree - 1) * np.pi) \
        + loggamma((degree - dim) / 2.) \
        - loggamma(degree / 2.)

    log_determinant = mat_logdet(covariance_matrix)

    log_pseudo_chisq = degree / 2. * np.log(
        1 + chi_square(
            data_vector - expectation_vector, covariance_matrix
        ) / (degree - 1)
    )

    density = - (log_normalisation + log_determinant / 2 + log_pseudo_chisq)

    return density if ret_log else np.exp(density)


# Likelihoods
# -----------------------------------------------------------------------------

class LikelihoodWarning(UserWarning):
    """Likelihood evaluation warning.

    """


def spherical_covariance(b_1, f_nl, spherical_model, pivot, **kwargs):
    r"""Compute the parametrised covariance matrix of spherical Fourier
    coefficients.

    Parameters
    ----------
    b_1 : float
        Scale-independent linear bias.
    f_nl : float or None
        Local primordial non-Gaussianity.
    spherical_model : :class:`~harmonia.reader.models.SphericalCorrelator`
        Spherical correlator base model.
    pivot : {'natural', 'spectral'}
        Pivot order for vectorisation.
    **kwargs
        Parameters (other than `b_1`, `f_nl` and `pivot`) to be passed to
        |correlator_matrix| of `spherical_correlator`.

    Returns
    -------
    covariance_matrix : complex :class:`numpy.ndarray`
        Covariance matrix.

    See Also
    --------
    :class:`~harmonia.reader.models.SphericalCorrelator`

    """
    covariance_matrix = spherical_model.correlator_matrix(
        pivot, b_1=b_1, f_nl=f_nl, **kwargs
    )

    return covariance_matrix


def cartesian_moments(b_1, f_nl, cartesian_model, covariance_estimator,
                      orders, pivot, **kwargs):
    """Compute the parametrised mean and covariance of Cartesian
    power spectrum multipoles.

    Parameters
    ----------
    b_1 : float
        Scale-independent linear bias of the tracer particles.
    f_nl : float or None
        Local primordial non-Gaussianity.
    cartesian_model : :class:`~.models.CartesianMultipoles`
        Cartesian power multipoles base model.
    covariance_estimator : :class:`~.synthesis.CovarianceEstimator`
        Cartesian power multipole covariance estimator.  Its
        :attr:`wavenumbers` must match wavenumbers associated
        with `cartesian_model`.
    orders : list of int
        Orders of the power spectrum multipoles.
    pivot : {'order', 'wavenumber'}
        Pivot order for vectorisation.
    **kwargs
        Parameters (other than `b_1`, `f_nl` and `orders`) to be passed to
        |convolved_power_multipoles| of `cartesian_model`.

    Returns
    -------
    expectation : float :class:`numpy.ndarray`
        Power spectrum expectation at the specified wavenumbers.
    covariance : float :class:`numpy.ndarray`
        Power spectrum variance at the specified wavenumbers.

    """
    # Check model and estimator wavenumbers agree.
    assert np.allclose(
        cartesian_model.attrs['wavenumbers'], covariance_estimator.wavenumbers
    ), (
        "The wavenumbers at which the Cartesian power multipole model "
        "is evaluated must match the wavenumbers at which "
        "the fiducial covariance matrix is estimated."
    )

    fiducial_expectation = covariance_estimator.get_fiducial_vector(pivot)
    fiducial_covariance = covariance_estimator.get_fiducial_covariance(pivot)

    expectation = cartesian_model.convolved_power_multipoles(
        orders, b_1, f_nl=f_nl, **kwargs
    ).vectorise(pivot)

    covariance = np.linalg.multi_dot([
        np.diag(expectation / fiducial_expectation),
        fiducial_covariance,
        np.diag(expectation / fiducial_expectation)
    ])

    return expectation, covariance


class LogLikelihood:
    """Construct the logarithmic likelihood function from
    cosmological data.

    Parameters
    ----------
    spherical_data : :class:`~.arrays.SphericalArray` *or None, optional*
        Spherical Fourier coefficient data (default is `None`).
    cartesian_data : :class:`~.arrays.CartesianArray` *or None, optional*
        Spherical Fourier coefficient data (default is `None`).
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator (default is `None`).

    Attributes
    ----------
    spherical_data : :class:`~.algorithms.arrays.SphericalArray` or None
        Spherical Fourier coefficient data.
    cartesian_data : :class:`~.algorithms.arrays.CartesianArray` or None
        Spherical Fourier coefficient data.

    """

    def __init__(self, spherical_data=None, cartesian_data=None, comm=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.comm = comm

        self.spherical_data = spherical_data
        self.cartesian_data = cartesian_data

    def spherical_map_likelihood(self, spherical_model, b_1, f_nl,
                                 pivot='natural', exclude_degrees=(),
                                 compression_matrix=None, **kwargs):
        """Evaluate the spherical map logarithmic likelihood.

        Parameters
        ----------
        spherical_model : :class:`~.reader.models.SphericalCorrelator`
            Spherical correlator base model.
        b_1 : float
            Scale-independent linear bias.
        f_nl : float or None
            Local primordial non-Gaussianity.
        pivot : {'natural', 'spectral'}, optional
            Pivot order for vectorisation (default is 'natural').
        exclude_degrees : tuple of int, optional
            If not empty (default), modes whose spherical degree
            match one of its elements are removed from the likelihood.
        compression_matrix : :class:`numpy.ndarray` *or None*, optional
            If not `None` (default), both the data vector and the model
            covariance matrix are processed for data compression.  This
            must be compatible with `exclude_degrees`, i.e. it accounts
            for elements removed from the data vector and covariance
            matrix by `exclude_degrees`.
        **kwargs
            Additional parameters to be passed to
            :func:`spherical_covariance`.

        Returns
        -------
        log_likelihood : float
            Logarithmic likelihood.

        See Also
        --------
        :class:`~harmonia.surveyor.synthesis.generate_compression_matrix`

        """
        _OVERFLOW_DOWNSCALE = 10**4

        data_vector = self.spherical_data.vectorise(pivot)

        covariance_matrix = spherical_covariance(
            b_1, f_nl, spherical_model, pivot, **kwargs
        )

        # pylint: disable=no-member
        if exclude_degrees:
            deselector = np.logical_and.reduce([
                self.spherical_data.array['index'][:, 0] == deg
                for deg in exclude_degrees
            ])

            data_vector = data_vector[~deselector]
            covariance_matrix = \
                covariance_matrix[~deselector, :][:, ~deselector]

        if compression_matrix is not None:
            data_vector = np.linalg.multi_dot([
                compression_matrix, data_vector
            ])
            covariance_matrix = np.linalg.multi_dot([
                compression_matrix, data_vector, np.conj(compression_matrix.T)
            ])

        log_likelihood = complex_normal_pdf(
            data_vector, covariance_matrix, downscale=_OVERFLOW_DOWNSCALE,
        )

        return log_likelihood

    def cartesian_map_likelihood(self, cartesian_model, covariance_estimator,
                                 b_1, f_nl, orders, pivot, num_samples=None,
                                 **kwargs):
        """Evaluate the Cartesian map logarithmic likelihood.

        Parameters
        ----------
        cartesian_model : :class:`~.models.CartesianMultipoles`
            Cartesian power multipoles base model.
        covariance_estimator : :class:`~.synthesis.CovarianceEstimator`
            Cartesian power multipole covariance estimator.  Its
            :attr:`wavenumbers` must match wavenumbers associated
            with `cartesian_model`.
        b_1 : float
            Scale-independent linear bias of the tracer particles.
        f_nl : float or None
            Local primordial non-Gaussianity.
        orders : list of int
            Orders of the power spectrum multipoles.
        pivot : {'order', 'wavenumber'}
            Pivot order for vectorisation.
        num_samples : int or None, optional
            If `None` (default), the normal distribution is used without
            correction for covariance estimation uncertainty; otherwise
            it is passed as `degree` to :func:`modified_student_pdf` for
            covariance estimation uncertainty correction (see [1]_).
        **kwargs
            Additional parameters to be passed to
            :func:`cartesian_moments`.

        Returns
        -------
        log_likelihood : float
            Logarithmic likelihood.


        .. [1] Sellentin E. & Heavens A. F., 2016. MNRAS 456(1), L132–L136.
           [arXiv: `1511.05969 <https://arxiv.org/abs/1511.05969>`_]

        """
        data_vector = self.cartesian_data.vectorise(pivot)

        expectation_vector, covariance_matrix = cartesian_moments(
            b_1, f_nl, cartesian_model, covariance_estimator, orders, pivot,
            **kwargs
        )

        if num_samples is None:
            log_likelihood = multivariate_normal_pdf(
                data_vector, expectation_vector, covariance_matrix
            )
        else:
            log_likelihood = modified_student_pdf(
                data_vector, expectation_vector, covariance_matrix,
                degree=num_samples
            )

        return log_likelihood
