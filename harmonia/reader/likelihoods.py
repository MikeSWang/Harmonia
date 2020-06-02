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

    # Check dimensions of the expectation and variance are consistent.
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

    # pylint: disable=unexpected-keyword-arg
    chi_sq = np.dot(
        np.conj(data_vector), np.linalg.solve(covariance_matrix, data_vector)
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
        - np.real_if_close(
            chi_square(data_vector, covariance_matrix), tol=10**10
        )

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


def spherical_covariance(pivot, spherical_model, **kwargs):
    r"""Compute the parametrised covariance matrix of spherical Fourier
    coefficients.

    Parameters
    ----------
    pivot : {'natural', 'spectral'}
        Pivot order for vectorisation.
    spherical_model : :class:`~harmonia.reader.models.SphericalCorrelator`
        Spherical correlator base model.
    **kwargs
        Parameters (other than `pivot`) to be passed to |correlator_matrix|
        of `spherical_correlator`.

    Returns
    -------
    covariance_matrix : complex :class:`numpy.ndarray`
        Covariance matrix.

    See Also
    --------
    :class:`~harmonia.reader.models.SphericalCorrelator`

    """
    covariance_matrix = spherical_model.correlator_matrix(pivot, **kwargs)

    return covariance_matrix


def cartesian_moments(pivot, orders, cartesian_model,
                      covariance_estimator=None, mode_counts=None, **kwargs):
    """Compute the parametrised mean and covariance of Cartesian
    power spectrum multipoles.

    Parameters
    ----------
    pivot : {'order', 'wavenumber'}
        Pivot order for vectorisation.
    orders : list of int
        Orders of the power spectrum multipoles.
    cartesian_model : :class:`~.models.CartesianMultipoles`
        Cartesian power multipoles base model.
    covariance_estimator : :class:`~.CovarianceEstimator` *or None, optional*
        Cartesian power multipole covariance estimator.  Its
        :attr:`wavenumbers` must match wavenumbers associated
        with `cartesian_model`.  If `None`, no correlation between power
        spectrum multipoles is assumed but `mode_counts` must be provided
        for calculating the power spectrum variance.
    mode_counts : int, array_like or None, optional
        Number of independent modes for each power spectrum measurement
        (default is `None`) used to calculate the power spectrum variance.
        Ignored if `covariance_estimator` is provided.
    **kwargs
        Parameters (other than `orders`) to be passed to
        |convolved_power_multipoles| of `cartesian_model`.

    Returns
    -------
    expectation : float :class:`numpy.ndarray`
        Power spectrum expectation at the specified wavenumbers.
    covariance : float :class:`numpy.ndarray`
        Power spectrum variance at the specified wavenumbers.

    """
    expectation = cartesian_model.convolved_power_multipoles(
        orders, **kwargs
    ).vectorise(pivot)

    # Check model and estimator wavenumbers agree.
    if covariance_estimator is None:
        covariance = expectation ** 2 / np.asarray(mode_counts)
    else:
        assert np.allclose(
            cartesian_model.attrs['wavenumbers'],
            covariance_estimator.wavenumbers,
            atol=0.001
        ), (
            "The wavenumbers at which the Cartesian power multipole model "
            "is evaluated must match the wavenumbers at which "
            "the fiducial covariance matrix is estimated."
        )

        fiducial_expectation = \
            covariance_estimator.get_fiducial_vector(pivot)
        fiducial_covariance = \
            covariance_estimator.get_fiducial_covariance(pivot)

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
    covariance_estimator : :class:`~.CovarianceEstimator` *or None, optional*
        Cartesian multipole covariance estimator (default is `None`).
    mode_counts : int, array_like or None, optional
        Number of independent modes for each Cartesian data point (default
        is `None`) as an alternative to `covariance_estimator`.  Ignored
        if `covariance_estimator` is provided.
    base_spherical_model : :class:`~.SphericalCorrelator` *or None, optional*
        Baseline spherical correlator model (default is `None`).
    base_cartesian_model : :class:`~.CartesianMultipoles` *or None, optional*
        Baseline Cartesian multipole model (default is `None`).
    spherical_pivot : {'natural', 'spectral'}, optional
        Pivot order for spherical map data vectorisation (default is
        'natural').
    cartesian_pivot : {'order', 'wavenumber'}, optional
        Pivot order for Cartesian map data vectorisation (default is
        'order').
    nbar : float or None, optional
        Mean particle number density (in cubic :math:`h`/Mpc).  If
        `None` (default), shot noise is neglected.
    contrast : float or None, optional
        If not `None` (default), this adds additional shot noise level
        ``1 / (contrast * nbar)`` due to a FKP-style random catalogue.
    tracer_p : float, optional
        Tracer-dependent parameter for bias modulation by `f_nl`
        (default is 1.).
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator (default is `None`).

    Attributes
    ----------
    attrs : dict
        Directory holding input parameters not corresponding to any of
        the following attributes.
    spherical_data : :class:`~.algorithms.arrays.SphericalArray` or None
        Spherical Fourier coefficient data.
    cartesian_data : :class:`~.algorithms.arrays.CartesianArray` or None
        Spherical Fourier coefficient data.
    covariance_estimator : :class:`~.CovarianceEstimator` or None
        Cartesian multipole covariance estimator.
    base_spherical_model : :class:`~.SphericalCorrelator` or None
        Baseline spherical correlator model.
    base_cartesian_model : :class:`~.CartesianMultipoles` or None
        Baseline Cartesian multipole model.

    """

    def __init__(self, spherical_data=None, cartesian_data=None,
                 covariance_estimator=None, mode_counts=None,
                 base_spherical_model=None, base_cartesian_model=None,
                 spherical_pivot='natural', cartesian_pivot='order',
                 nbar=None, contrast=None, tracer_p=1., comm=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.comm = comm

        self.attrs = {
            'spherical_pivot': spherical_pivot,
            'cartesian_pivot': cartesian_pivot,
            'mode_counts': mode_counts,
            'nbar': nbar,
            'contrast': contrast,
            'tracer_p': tracer_p,
        }

        self.spherical_data = spherical_data
        self.cartesian_data = cartesian_data
        self.covariance_estimator = covariance_estimator
        self.base_spherical_model = base_spherical_model
        self.base_cartesian_model = base_cartesian_model

    def spherical_map_likelihood(self, b_1, f_nl, exclude_degrees=(),
                                 compression_matrix=None, **kwargs):
        """Evaluate the spherical map logarithmic likelihood.

        Parameters
        ----------
        b_1 : float
            Scale-independent linear bias.
        f_nl : float or None
            Local primordial non-Gaussianity.
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
            Additional parameters to be passed to |correlator_matrix| of
            :attr:`base_spherical_model`.

        Returns
        -------
        log_likelihood : float
            Logarithmic likelihood.

        See Also
        --------
        :class:`~harmonia.surveyor.synthesis.generate_compression_matrix`

        :class:`~harmonia.reader.likelihoods.spherical_covariance`

        |correlator_matrix|

        """
        _OVERFLOW_DOWNSCALE = 10**4

        data_vector = \
            self.spherical_data.vectorise(self.attrs['spherical_pivot'])

        covariance_matrix = spherical_covariance(
            self.attrs['spherical_pivot'], self.base_spherical_model,
            b_1=b_1, f_nl=f_nl,
            nbar=self.attrs['nbar'],
            contrast=self.attrs['contrast'],
            tracer_p=self.attrs['tracer_p'],
            **kwargs
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
                compression_matrix,
                covariance_matrix,
                np.conj(compression_matrix.T)
            ])

        log_likelihood = complex_normal_pdf(
            data_vector, covariance_matrix, downscale=_OVERFLOW_DOWNSCALE,
        )

        return log_likelihood

    def cartesian_map_likelihood(self, b_1, f_nl, orders=None,
                                 num_samples=None, **kwargs):
        """Evaluate the Cartesian map logarithmic likelihood.

        Parameters
        ----------
        b_1 : float
            Scale-independent linear bias of the tracer particles.
        f_nl : float or None
            Local primordial non-Gaussianity.
        orders : list of int or None, optional
            Orders of the power spectrum multipoles.  If `None` (default),
            only the monopole is used.
        num_samples : int or None, optional
            If `None` (default), the normal distribution is used without
            correction for covariance estimation uncertainty; otherwise
            it is passed as `degree` to :func:`modified_student_pdf` for
            covariance estimation uncertainty correction [1]_.
        **kwargs
            Additional parameters to be passed to
            |convolved_power_multipoles| of :attr:`base_cartesian_model`.

        Returns
        -------
        log_likelihood : float
            Logarithmic likelihood.

        See Also
        --------
        :class:`~harmonia.reader.likelihoods.cartesian_moments`

        |convolved_power_multipoles|


        .. [1] Sellentin E. & Heavens A. F., 2016. MNRAS 456(1), L132–L136.
           [arXiv: `1511.05969 <https://arxiv.org/abs/1511.05969>`_]

        """
        orders = orders or [0]

        data_vector = \
            self.cartesian_data.vectorise(self.attrs['cartesian_pivot'])

        expectation_vector, covariance_matrix = cartesian_moments(
            self.attrs['cartesian_pivot'], orders, self.base_cartesian_model,
            covariance_estimator=self.covariance_estimator,
            mode_counts=self.attrs['mode_counts'],
            b_1=b_1, f_nl=f_nl,
            nbar=self.attrs['nbar'],
            contrast=self.attrs['contrast'],
            tracer_p=self.attrs['tracer_p'],
            **kwargs
        )

        if self.covariance_estimator is not None and num_samples is not None:
            log_likelihood = modified_student_pdf(
                data_vector, expectation_vector, covariance_matrix,
                degree=num_samples
            )
        else:
            log_likelihood = multivariate_normal_pdf(
                data_vector, expectation_vector, covariance_matrix
            )

        return log_likelihood
