import logging

import pytest
from nbodykit.lab import cosmology as cosmo

import testrc
from harmonia.mapper.catalogue_maker import (
    spherical_indicator, RandomCatalogue, LognormalCatalogue
    )

TEST_PARAMS = dict(
    nmean=5e-4,
    boxsize=100.,
    nmesh=128,
    bias=2.,
    rsd=True,
    seed=42,
    Plin=cosmo.LinearPower(cosmo.Planck15, redshift=0., transfer='CLASS')
    )


@pytest.mark.parametrize(
    'pos,rad,ind',
    [([-1, 0, 0], 1, 1),
     ([0.78, -3.23, 5.32], 2., 0)]
    )
def test_spherical_indicator(pos, rad, ind):
    assert int(spherical_indicator(pos, rad)) == ind


@pytest.fixture(scope='module')
def random_catalogue():
    return RandomCatalogue(
        nmean=TEST_PARAMS['nmean'], boxsize=TEST_PARAMS['boxsize']
        )


@pytest.fixture(scope='module')
def lognormal_catalogue():
    return LognormalCatalogue(
        powerlin=TEST_PARAMS['Plin'], nmean=TEST_PARAMS['nmean'],
        boxsize=TEST_PARAMS['boxsize'], nmesh=TEST_PARAMS['nmesh'],
        bias=TEST_PARAMS['bias'], add_RSD=TEST_PARAMS['rsd']
        )


def test_RandomCatalogue_init(random_catalogue, caplog):
    assert random_catalogue.attrs['nbar'] == pytest.approx(
        TEST_PARAMS['nmean']
        )
    with caplog.at_level(logging.DEBUG):
        RandomCatalogue(
            nmean=TEST_PARAMS['nmean'], boxsize=TEST_PARAMS['boxsize'],
            seed=TEST_PARAMS['seed']
            )
        assert caplog.records[-1].message.startswith(
            f"RandomCatalogue(nmean={TEST_PARAMS['nmean']},"
            )


def test_LognormalCatalogue_init(lognormal_catalogue, caplog):
    assert lognormal_catalogue.attrs['nbar'] == pytest.approx(
        TEST_PARAMS['nmean']
        )
    assert lognormal_catalogue.attrs['RSD'] == TEST_PARAMS['rsd']
    with caplog.at_level(logging.DEBUG):
        LognormalCatalogue(
            powerlin=TEST_PARAMS['Plin'], nmean=TEST_PARAMS['nmean'],
            boxsize=TEST_PARAMS['boxsize'], nmesh=TEST_PARAMS['nmesh'],
            bias=TEST_PARAMS['bias'], add_RSD=True
            )
        assert (caplog.records[-1].message ==
            "RSDs added to radial particle velocities. "
            )
