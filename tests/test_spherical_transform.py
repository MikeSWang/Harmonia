import numpy as np
import pytest

from harmonia.algorithms import DiscreteSpectrum
from harmonia.mapper import RandomCatalogue
from harmonia.mapper.spherical_transform import SphericalMap
from harmonia.collections import sort_dict_to_list


TEST_PARAMS = dict(
    mean_density=1e-4,
    rmax=100.,
    kmax=0.1,
    pivot='scale',
    order_collapse=True,
)


@pytest.fixture(scope='module')
def spherical_map():
    return SphericalMap(
        DiscreteSpectrum(TEST_PARAMS['rmax'], 'd', TEST_PARAMS['kmax']),
        RandomCatalogue(TEST_PARAMS['mean_density'], 2*TEST_PARAMS['rmax']),
        mean_density_data=TEST_PARAMS['mean_density']
    )


@pytest.mark.slow
def test_SphericalMap(spherical_map):

    disc = spherical_map.disc
    flat_order = np.concatenate(sort_dict_to_list(disc.wavenumbers)).argsort()

    normalisations = np.concatenate(
        sort_dict_to_list(disc.normalisations)
    )[flat_order]

    spherical_power = np.concatenate(
        spherical_map.spherical_power()
    )[flat_order]

    variance = spherical_power / normalisations

    covariance = spherical_map.two_points_pivoted(
        TEST_PARAMS['pivot'],
        order_collapse=TEST_PARAMS['order_collapse']
    )

    assert variance == pytest.approx(np.diag(covariance))
