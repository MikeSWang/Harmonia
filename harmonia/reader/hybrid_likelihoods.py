"""
Hybrid likelihoods (:mod:`~harmonia.reader.hybrid_likelihoods`)
===========================================================================

Build hybrid likelihoods for cosmological parameter inference.

Probability distributions
---------------------------------------------------------------------------

.. autosummary::

    complex_normal_pdf
    multivariate_normal_pdf


Spherical likelihood
---------------------------------------------------------------------------

Cartesian likelihood
---------------------------------------------------------------------------

Hybrid likelihood
---------------------------------------------------------------------------

"""
import numpy as np

from harmonia.collections import matrix_log_det


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
        (Logarithmic) probability density value.

    """
    dat_vector, cov_matrix = np.squeeze(dat_vector), np.squeeze(cov_matrix)
    if cov_matrix.ndim == 1:
        cov_matrix = np.diag(cov_matrix)
    if np.size(dat_vector)**2 != np.size(cov_matrix):
        raise ValueError(
            "`data_vector` and `cov_matrix` dimensions are not consistent: "
            "{} and ({}, {}). ".format(
                np.size(dat_vector),
                np.size(cov_matrix, axis=0),
                np.size(cov_matrix, axis=1),
            )
        )

    dat_dim = np.size(dat_vector)

    log_normalisation_const = dat_dim * np.log(np.pi)

    if downscale is not None:
        dat_vector = dat_vector / downscale
        cov_matrix = cov_matrix / downscale**2
        log_normalisation_const -= 2 * dat_dim * np.log(downscale)

    if elementwise:
        var_vector = np.diag(cov_matrix)
        sign_product = np.prod(np.sign(var_vector))
        if sign_product != 1.:
            raise ValueError("`cov_matrix` is not positive definite. ")
        log_det_cov_mat = np.log(np.abs(var_vector))
    else:
        log_det_cov_mat = matrix_log_det(cov_matrix)

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
        (Logarithmic) probability density value.

    Raises
    ------
    ValueError
        If the dimensions of `data_vector`, `mean_vector` and `cov_matrix`
        are not consistent.

    """
    data_vector, mean_vector, cov_matrix = np.squeeze(data_vector),\
        np.squeeze(mean_vector), np.squeeze(cov_matrix)
    if cov_matrix.ndim == 1:
        cov_matrix = np.diag(cov_matrix)
    if np.size(data_vector) != np.size(data_vector)\
            or np.size(data_vector)**2 != np.size(cov_matrix):
        raise ValueError(
            "Dimensions of `data_vector`, `mean_vector` and `cov_matrix` "
            "are not consistent: {}, {} and ({}, {}). ".format(
                np.size(data_vector),
                np.size(mean_vector),
                np.size(cov_matrix, axis=0),
                np.size(cov_matrix, axis=1),
            )
        )

    dat_dim = np.size(data_vector)

    log_normalisation_const = dat_dim * np.log(2*np.pi)

    log_det_cov_mat = matrix_log_det(cov_matrix)

    exponent = _chi_square(
        data_vector - mean_vector,
        cov_matrix,
        elementwise=False
    )

    density = 1/2 * (- log_normalisation_const - log_det_cov_mat - exponent)

    if not return_log:
        density = np.exp(density)

    return density


# Spherical likelihood
# -----------------------------------------------------------------------------


# Cartesian likelihood
# -----------------------------------------------------------------------------


# Hybrid likelihood
# -----------------------------------------------------------------------------
