import logging

import pytest

from harmonia.algorithms.discretisation import DiscreteSpectrum

TEST_ARGS = dict(
    radius=100.,
    condition='Dirichlet',
    highcut=0.1,
    lowcut=0.,
    maxdeg=None,
    mindeg=0,
)

TEST_ELL = 2


@pytest.fixture(scope='module')
def discrete_spectrum():
    return DiscreteSpectrum(**TEST_ARGS)


def test_DiscreteSpectrum(discrete_spectrum, caplog):

    with pytest.raises(ValueError):
        DiscreteSpectrum(100, 'invalid_condition', 0.1)

    with caplog.at_level(logging.DEBUG):
        DiscreteSpectrum(**TEST_ARGS)
        assert caplog.records[0].message.startswith("Roots for degree")
        assert caplog.records[-2].message.startswith(
            "No more modes. Last degree"
        )
        assert caplog.records[-1].message.startswith("Spectrum") \
            and caplog.records[-1].message.endswith("modes in total. ")

    assert discrete_spectrum.degrees == [0, 1, 2, 3, 4, 5]
    assert discrete_spectrum.depths == [3, 2, 2, 1, 1, 1]
    assert discrete_spectrum.roots[TEST_ELL] \
        == pytest.approx([5.7634591969, 9.0950113305])
    assert discrete_spectrum.mode_count == sum(
        [
            (2*ell + 1) * nmax
            for ell, nmax in zip(
                discrete_spectrum.degrees, discrete_spectrum.depths
            )
        ]
    )

    assert hasattr(discrete_spectrum, 'attrs')
    assert discrete_spectrum._wavenumbers is None \
        and discrete_spectrum._root_indices is None \
        and discrete_spectrum._normalisations is None

    assert discrete_spectrum.wavenumbers[TEST_ELL] \
        == pytest.approx([0.0576345920, 0.0909501133])
    assert discrete_spectrum._wavenumbers is not None

    assert discrete_spectrum.root_indices[TEST_ELL] == [
        (TEST_ELL, n)
        for n in range(1, discrete_spectrum.depths[TEST_ELL]+1)
    ]
    assert discrete_spectrum._root_indices is not None

    assert discrete_spectrum.normalisations[TEST_ELL] \
        == pytest.approx([0.0000729768, 0.0001716561])
    assert discrete_spectrum._normalisations is not None
