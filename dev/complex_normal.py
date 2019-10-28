"""Complex normal probability distribution.

"""
import numpy as np

from harmonia.collections import matrix_log_det


def chi_square(dat_vector, cov_matrix, by_diag=False):
    """Calculate chi-square for zero-centred data vector and its covariance 
    matrix.

    Parameters
    ----------
    dat_vector : float or complex, array_like
        1-d data vector centred at zero.
    cov_matrix : float or complex, array_like
        2-d covariance matrix.
    by_diag : bool, optional
        If `True` (default is `False`), return chi-square from each
        component of the data vector assuming a diagonal covariance matrix.

    Returns
    -------
    chi_sq : float :class:`numpy.ndarray`
        Chi-square value(s).

    Raises
    ------
    ValueError
        If `dat_vector` is not in an equivalent shape for a 1-d vector.

    """
    if np.squeeze(dat_vector).ndim > 1:
        raise ValueError("`data` is not equivalent to a 1-d vector. ")

    if by_diag:
        chi_sq = np.abs(dat_vector)**2 / np.diag(cov_matrix).real
    else:
        chi_sq = np.real(
            np.conj(dat_vector).T @ np.linalg.inv(cov_matrix) @ dat_vector
        )

    return chi_sq


def complex_normal_log_pdf(dat_vector, cov_matrix, overflow_rescale=None):
    """Compute natural logarithm of the complex normal probability density
    function given the zero-centred data vector and covariance matrix.

    Parameters
    ----------
    dat_vector : float or complex, array_like
        1-d data vector centred at zero.
    cov_matrix : float or complex, array_like
        2-d covariance matrix.
    overflow_rescale : float or None, optional
        If not `None` (default), the data vector and covariance matrix are
        simultaneous rescaled by division in computing the exponent and
        matrix determinant normalisation.

    Returns
    -------
    log_pdf: float
        Log PDF value.

    """
    dat_dim = np.size(dat_vector)
    normalisation_const = - dat_dim * np.log(np.pi)

    if overflow_rescale is not None:
        dat_vector = dat_vector / overflow_rescale
        cov_matrix = cov_matrix / overflow_rescale**2
        normalisation_const += 2 * dat_dim * np.log(overflow_rescale)

    det_divider = matrix_log_det(cov_matrix)

    chisq_exponent = chi_square(dat_vector, cov_matrix)

    log_pdf = normalisation_const - det_divider - chisq_exponent

    return log_pdf
