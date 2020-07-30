import numpy as np
import pytest

from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.algorithms.arrays import CartesianArray, SphericalArray

DISC_PARAMS = dict(
    radius=50.,
    condition='dirichlet',
    highcut=0.25,
    lowcut=0.,
    maxdeg=None,
    mindeg=0,
)

FIXTURE_PARAMS = dict(
    orders=[0, 2],
    wavenumbers=[0.001, 0.04, 0.1]
)


@pytest.fixture(scope='module')
def spherical_array():

    discrete_spectrum = DiscreteSpectrum(**DISC_PARAMS)

    array = SphericalArray(discrete_spectrum)

    # Arbitraty coefficient data entries for testing only.
    array[:] = np.array([
        index[0] + index[-1] * 1j
        for index in array.array['index']
    ])

    return array


class TestSphericalArray:

    @pytest.mark.parametrize(
        "pivot,collapse,value",
        [
            ('spectral', None, [0. + 1.j, 1. + 1.j, 1. + 1.j,])
        ]
    )
    def test_vectorise(self, spherical_array, pivot, collapse, value):
        assert spherical_array.vectorise(pivot, collapse)[:3] \
            == pytest.approx(value), \
            "Vectorisation of SphericalArray test instance failed."

    @pytest.mark.parametrize(
        "key,value",
        [
            (-1, 7. + 1.j),
            ((0, 0, 1), 0. + 1.j),
            (slice(-2, -1), 7. + 1.j)
        ]
    )
    def test___getitem__(self, spherical_array, key, value):
        assert spherical_array[key] == pytest.approx(value), \
            "Container emulation failed for SphericalArray test instance."

    @pytest.mark.parametrize(
        "key,value",
        [
            (-1, 100. - 100.j),
            ((0, 0, 1), 0. - 1.j),
            (slice(-2, -1), 0.5 + 1.5j)
        ]
    )
    def test___setitem__(self, spherical_array, key, value):
        spherical_array[key] = value
        assert spherical_array[key] == pytest.approx(value), \
            "Container emulation failed for SphericalArray test instance."


@pytest.fixture(scope='module')
def cartesian_array():

    array = CartesianArray(**FIXTURE_PARAMS)

    # Arbitraty power data entries for testing only.
    array[:] = np.concatenate([
        array.attrs['wavenumbers'] ** order
        for order in array.attrs['orders']
    ])

    return array


class TestCartesianArray:

    @pytest.mark.parametrize(
        "pivot,value",
        [
            ('order', [1., 1., 1., 1.e-6, 1.6e-3, 1.e-2]),
            ('wavenumber', [1., 1.e-6, 1., 1.6e-3, 1., 1.e-2])
        ]
    )
    def test_vectorise(self, cartesian_array, pivot, value):
        assert cartesian_array.vectorise(pivot) == pytest.approx(value), \
            "Vectorisation of SphericalArray test instance failed."

    @pytest.mark.parametrize(
        "key,value",
        [
            (-1, 0.01),
            ((0, 0.04), 1.),
            ('power_2', [1.e-6, 1.6e-3, 1.e-2])
        ]
    )
    def test___getitem__(self, cartesian_array, key, value):
        assert cartesian_array[key] == pytest.approx(value), \
            "Container emulation failed for CartesianArray test instance."

    @pytest.mark.parametrize(
        "key,value",
        [
            (-1, 0.02),
            ((0, 0.04), 2.),
            ('power_2', [2.e-6, 2.6e-3, 2.e-2])
        ]
    )
    def test___setitem__(self, cartesian_array, key, value):
        cartesian_array[key] = value
        assert cartesian_array[key] == pytest.approx(value), \
            "Container emulation failed for CartesianArray test instance."
