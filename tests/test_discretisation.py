import logging

import pytest

from harmonia.algorithms.discretisation import DiscreteSpectrum

TEST_PARAMS = dict(
    spectrum=dict(
        radius=100.,
        condition='Dirichlet',
        cutoff=0.1,
        cuton=0.,
        maxdeg=None,
        mindeg=0,
    ),
    discretisation=dict(
        radius=100.,
        condition='Dirichlet',
        kmax=0.1,
        kmin=0.,
        ellmax=None,
        ellmin=0,
    ),
    ell=2,
)


@pytest.fixture(scope='module')
def discrete_spectrum():
    return DiscreteSpectrum(**TEST_PARAMS['spectrum'])


def test_init(discrete_spectrum, caplog):

    with caplog.at_level(logging.INFO):
        DiscreteSpectrum(**TEST_PARAMS['spectrum'])
        assert caplog.records[-1].message.startswith("Spectrum") \
            and caplog.records[-1].message.endswith("modes in total. ")

    assert discrete_spectrum.degrees == [0, 1, 2, 3, 4, 5]
    assert discrete_spectrum.depths == [3, 2, 2, 1, 1, 1]
    assert discrete_spectrum.mode_count == sum(
        [
            (2*ell + 1) * nmax
            for ell, nmax in zip(
                discrete_spectrum.degrees,
                discrete_spectrum.depths,
            )
        ],
    )
    assert discrete_spectrum.roots[TEST_PARAMS['ell']] \
        == pytest.approx([5.7634591969, 9.0950113305])

    assert hasattr(discrete_spectrum, 'attrs')
    assert discrete_spectrum._wavenumbers is None \
        and discrete_spectrum._root_indices is None \
        and discrete_spectrum._normalisations is None


def test_discretise(caplog):
    with caplog.at_level(logging.DEBUG):
        DiscreteSpectrum.discretise(**TEST_PARAMS['discretisation'])
        assert caplog.records[0].message.startswith(
            "Results for degree 0 appended."
        )
        assert caplog.records[-1].message.startswith(
            "No more roots found. Last degree"
        )
    with pytest.raises(ValueError):
        DiscreteSpectrum.discretise(100, 'invalid_condition', 0., 1., 0, 5)


def test_wavenumbers(discrete_spectrum):
    assert discrete_spectrum.wavenumbers[TEST_PARAMS['ell']] \
        == pytest.approx([0.0576345920, 0.0909501133])
    assert discrete_spectrum._wavenumbers is not None


def test_root_indices(discrete_spectrum):
    assert discrete_spectrum.root_indices[TEST_PARAMS['ell']] == [
        (TEST_PARAMS['ell'], n)
        for n in range(1, discrete_spectrum.depths[TEST_PARAMS['ell']]+1)
    ]
    assert discrete_spectrum._root_indices is not None


def test_normalisations(discrete_spectrum):
    assert discrete_spectrum.normalisations[TEST_PARAMS['ell']] \
        == pytest.approx([0.0000729768, 0.0001716561])
    assert discrete_spectrum._normalisations is not None
