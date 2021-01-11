import numpy as np
import pytest

from harmonia.reader.likelihoods import (
    chi_square,
    complex_normal_pdf,
    modified_student_pdf,
    multivariate_normal_pdf,
)


@pytest.mark.parametrize(
    "ndim,nsample",
    [
        (3, 10000),
    ]
)
def test_chi_square(ndim, nsample):

    samples = np.random.randn(nsample, ndim)
    covariance_matrix = np.eye(ndim)

    assert np.allclose(
        np.mean([
            chi_square(sample_vector, covariance_matrix)
            for sample_vector in samples
        ]),
        ndim,
        atol=10*np.sqrt(2 * ndim / nsample)  # allow 10-sigma deviation
    ), "Incorrect chi square evaluation."


@pytest.mark.parametrize(
    "data_vector,covariance_matrix,value",
    [
        ([1. + 1.j, - 1. - 1.j], np.eye(2), np.exp(-4) / np.pi ** 2),
    ]
)
def test_complex_normal_pdf(data_vector, covariance_matrix, value):
    print(value)
    assert np.isclose(
        complex_normal_pdf(data_vector, covariance_matrix, ret_log=False),
        value
    ), "Incorrect complex normal PDF evaluation."


@pytest.mark.parametrize(
    "data_vector,expectation_vector,covariance_matrix,degree,value",
    [
        ([1., -1.], [0., 0.], np.eye(2), 100, 0.0579565),
    ]
)
def test_modified_student_pdf(data_vector, expectation_vector,
                              covariance_matrix, degree, value):

    assert np.isclose(
        modified_student_pdf(
            data_vector, expectation_vector, covariance_matrix,
            degree, ret_log=False
        ), value
    ), "Incorrect modified Student PDF evaluation."


@pytest.mark.parametrize(
    "data_vector,expectation_vector,covariance_matrix,degree",
    [
        ([1., -1.], [0., 0.], np.eye(2), 1000),
    ]
)
def test_multivariate_normal_pdf(data_vector, expectation_vector,
                                 covariance_matrix, degree):

    assert np.isclose(
        multivariate_normal_pdf(
            data_vector, expectation_vector, covariance_matrix, ret_log=False
        ),
        modified_student_pdf(
            data_vector, expectation_vector, covariance_matrix,
            degree, ret_log=False
        ),
        rtol=1/np.sqrt(degree)
    ), (
        "Multivariat normal and modified Student PDF "
        "evaluations do not match in the large number limit."
    )
