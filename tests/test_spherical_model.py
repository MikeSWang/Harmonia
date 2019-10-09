import numpy as np
import pytest
from nbodykit.lab import cosmology

from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.collections import unit_const
from harmonia.reader import Couplings  # TODO: test `TwoPointFunction`

DISC_PARAMS = dict(
    radius=100.,
    condition='Dirichlet',
    cutoff=0.1,
    cuton=0.,
    maxdeg=None,
    mindeg=0,
)

COSMO_PARAMS = {
    'nbar': 1e-3,
    'bias': 2.,
    'Plin': cosmology.LinearPower(cosmology.Planck15, redshift=0.),
    'beta': 0.35,
}

TEST_PARAMS = {
    'ell': 2,
    'mu': (1, -1, 1),
}


@pytest.fixture(scope='module')
def discrete_spectrum():
    return DiscreteSpectrum(**DISC_PARAMS)


SURVEY_SPECS = {
    'mask': unit_const,
    'selection': unit_const,
    'weight': None,
    'weight_derivative': None,
}

@pytest.mark.parametrize('coupling_type', ['angular', 'radial', 'RSD'])
def test_Couplings(discrete_spectrum, coupling_type):
    couplings_partial = Couplings(discrete_spectrum)
    couplings_full = Couplings(discrete_spectrum, survey_specs=SURVEY_SPECS)
    partial_couplings_fixed_index = couplings_partial.couplings_fixed_index(
        mu=TEST_PARAMS['mu'],
        coupling_type=coupling_type
    )
    full_couplings_fixed_index = couplings_full.couplings_fixed_index(
        mu=TEST_PARAMS['mu'],
        coupling_type=coupling_type
    )
    assert np.allclose(
        partial_couplings_fixed_index[TEST_PARAMS['ell']],
        full_couplings_fixed_index[TEST_PARAMS['ell']],
    )
