import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD

import harmonia.utils as u


@pytest.mark.parametrize(
    "data_array,mapping,comm,values",
    [
        ([1, 2, 3, 4], np.square, COMM_WORLD, [1, 4, 9, 16]),
    ]
)
def test_mpi_compute(data_array, mapping, comm, values):
    assert u.mpi_compute(data_array, mapping, comm=comm) \
        == pytest.approx(values), "MPI computation failed."


@pytest.mark.parametrize(
    "const,args",
    [
        (1j, (5, 'hello', None, False)),
    ]
)
def test_const_function(const, args):
    assert u.const_function(const)(*args) == pytest.approx(const), \
        "Constant function fails to return the specified constant."


@pytest.mark.parametrize(
    "vector,result",
    [
        (
            [[1, 0, 0], [1, 1, 0]],
            [[1, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0]]
        ),
    ]
)
def test_normalise_vector(vector, result):
    assert np.allclose(u.normalise_vector(vector), result), \
        "Vector normalisation failed."


@pytest.mark.slow
@pytest.mark.parametrize(
    "func,a,b,maxnum,values",
    [
        (np.sin, -0.01, 3.15, 1, np.array([0])),
        (np.cos, -0.01, 3.15, 10, np.array([np.pi/2])),
    ]
)
def test_binary_search(func, a, b, maxnum, values):
    assert u.binary_search(func, a, b, maxnum=maxnum) \
        == pytest.approx(values), \
        "Binary search fails to find all roots in the specified interval."


@pytest.mark.parametrize(
    "covar,value",
    [
        ([[2, 1], [1, 2]], [[1, 0.5], [0.5, 1]])
    ]
)
def test_covar_to_corr(covar, value):
    assert np.allclose(u.covar_to_corr(covar), value), \
        "Covariance conversion to correlation matrix failed."


@pytest.mark.parametrize(
    "matrix,value",
    [
        (np.array([[1, 2], [3, 4]]), 0.693147)
    ]
)
def test_mat_logdet(matrix, value):
    assert u.mat_logdet(matrix) == pytest.approx(value), \
        "Incorrect logarithmic determinant of matrix."


@pytest.mark.parametrize(
    "matrix,value",
    [
        (np.array([[1, 2], [3, 4]]), False)
    ]
)
def test_is_positive_definite(matrix, value):
    assert (u.is_positive_definite(matrix) is value), \
        "Positive definiteness check failed."
