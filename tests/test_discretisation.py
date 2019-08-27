import logging

import numpy as np
import pytest

import testrc
from harmonia.algorithms.discretisation import DiscreteSpectrum

TEST_PARAMS = dict(
    radius = 100.,
    condition = 'Dirichlet',
    cutoff = 0.1,
    cuton = 0.,
    degmax = None,
    degmin = 0,
)

TEST_CASE_ELL = 2


@pytest.fixture(scope='module')
def discrete_spectrum():
    return DiscreteSpectrum(**TEST_PARAMS)


def test_init(discrete_spectrum, caplog):
    with caplog.at_level(logging.INFO):
        DiscreteSpectrum(**TEST_PARAMS)
        assert (
            caplog.records[-1].message.startswith("Spectrum(") and
            caplog.records[-1].message.endswith("modes in total. ")
            )

    assert discrete_spectrum.degrees == [0, 1, 2, 3, 4, 5]
    assert discrete_spectrum.depths == [3, 2, 2, 1, 1, 1]
    assert discrete_spectrum.nmodes == sum(
        [(2*ell + 1) * nmax
         for ell, nmax in zip(
             discrete_spectrum.degrees, discrete_spectrum.depths
             )]
        )
    assert np.allclose(
        discrete_spectrum.roots[TEST_CASE_ELL],
        [5.76345919689455, 9.09501133047636,]
        )

    assert hasattr(discrete_spectrum, 'attrs')
    assert discrete_spectrum._wavenumbers is None
    assert discrete_spectrum._waveindices is None
    assert discrete_spectrum._normcoeff is None


def test_discretise(caplog):
    with caplog.at_level(logging.DEBUG):
        DiscreteSpectrum.discretise(
            TEST_PARAMS['radius'], TEST_PARAMS['condition'],
            TEST_PARAMS['cuton'], TEST_PARAMS['cutoff'],
            TEST_PARAMS['degmin'], TEST_PARAMS['degmax']
            )
        assert caplog.records[0].message.startswith(
            "Results for degree {ell} appended.".format(ell=0)
            )
        assert caplog.records[-1].message.startswith(
            "No more roots found. Last degree"
            )
    with pytest.raises(ValueError):
        DiscreteSpectrum.discretise(
            TEST_PARAMS['radius'], 'invalid_condition',
            TEST_PARAMS['cuton'], TEST_PARAMS['cutoff'],
            TEST_PARAMS['degmin'], TEST_PARAMS['degmax']
            )


def test_wavenumbers(discrete_spectrum):
    assert np.allclose(
        discrete_spectrum.wavenumbers[TEST_CASE_ELL],
        [0.0576345919689455, 0.0909501133047636,]
        )
    assert discrete_spectrum._wavenumbers is not None


def test_normcoeff(discrete_spectrum):
    assert np.allclose(
        discrete_spectrum.normcoeff[TEST_CASE_ELL],
        [0.00007297680749341, 0.00017165606577,]
        )
    assert discrete_spectrum._normcoeff is not None


def test_waveindices(discrete_spectrum):
    assert discrete_spectrum.waveindices[TEST_CASE_ELL] == [
        (TEST_CASE_ELL, n)
        for n in range(1, discrete_spectrum.depths[TEST_CASE_ELL]+1)
        ]
    assert discrete_spectrum._waveindices is not None
