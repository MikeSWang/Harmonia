import pytest
import numpy as np

from concordia.algorithms.discretise import DiscreteSpectrum

TEST_PARAMS = dict(
    radius = 100.,
    condition = 'Dirichlet',
    cutoff = 0.1,
    degmax = None,
    cuton = 0.,
    degmin = 0
)

TEST_CASE_ELL_INDEX = 2


@pytest.fixture
def TestInstance():
    return DiscreteSpectrum(**TEST_PARAMS)


def test_discretisation(TestInstance):
    assert TestInstance.degrees == [0, 1, 2, 3, 4, 5]
    assert TestInstance.depths == [3, 2, 2, 1, 1, 1]
    assert TestInstance.nmodes == sum(
        [(2*ell + 1) * nmax
         for ell, nmax in zip(TestInstance.degree, TestInstance.depths)]
        )
    assert np.allclose(
        TestInstance.roots[TEST_CASE_ELL_INDEX],
        [5.76345919689455, 9.09501133047636,]
        )


def test_wavenumbers(TestInstance):
    assert np.allclose(
        TestInstance.wavenumbers[TEST_CASE_ELL_INDEX],
        [0.0576345919689455, 0.0909501133047636,]
        )


def test_waveindices(TestInstance):
    assert TestInstance.waveindices[TEST_CASE_ELL_INDEX] == [
        (TEST_CASE_ELL_INDEX, n)
        for n in range(1, TestInstance.depths[TEST_CASE_ELL_INDEX]+1)
        ]


def test_normcoeff(TestInstance):
    assert np.allclose(
        TestInstance.normcoeff[TEST_CASE_ELL_INDEX],
        [0.00007297680749341, 0.00017165606577,]
        )
