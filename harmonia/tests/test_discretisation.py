import pytest

from harmonia.algorithms.discretisation import DiscreteSpectrum

FIXTURE_ARGS = dict(
    radius=100.,
    condition='dirichlet',
    highcut=0.1,
    lowcut=0.,
    maxdeg=None,
    mindeg=0,
)


@pytest.fixture(scope='module')
def discrete_spectrum():
    return DiscreteSpectrum(**FIXTURE_ARGS)


@pytest.mark.parametrize(
    "ell,n,k,kappa",
    [
        (2, 1, 0.0576346, 0.0000729768),
    ]
)
def test_DiscreteSpectrum(discrete_spectrum, ell, n, k, kappa):

    assert discrete_spectrum.degrees == [0, 1, 2, 3, 4, 5], \
        "Incorrect spherical degrees for DiscreteSpectrum test instance."
    assert discrete_spectrum.depths == [3, 2, 2, 1, 1, 1], \
        "Incorrect spherical depths for DiscreteSpectrum test instance."
    assert discrete_spectrum.mode_counts == [
        (2*ell + 1) * nmax
        for ell, nmax in zip(
            discrete_spectrum.degrees, discrete_spectrum.depths
        )
    ], "Incorrect mode counts for DiscreteSpectrum test instance."

    assert discrete_spectrum.wavenumbers[(ell, n)] == pytest.approx(k), \
        "Incorrect wavenumbers for DiscreteSpectrum test instance."
    assert discrete_spectrum.normalisations[(ell, n)] \
        == pytest.approx(kappa), \
        "Incorrect normalisations for DiscreteSpectrum test instance."
