import numpy as np
import pytest

from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.collections import unit_const
from harmonia.cosmology import fiducial_cosmology
from harmonia.reader import Couplings, TwoPointFunction

DISC_PARAMS = dict(
    radius=100.,
    condition='Dirichlet',
    cutoff=0.1,
    cuton=0.,
    maxdeg=None,
    mindeg=0,
)


@pytest.fixture(scope='module')
def discrete_spectrum():
    return DiscreteSpectrum(**DISC_PARAMS)


COUPLINGS_SURVEY_SPECS = {
    'mask': unit_const,
    'selection': unit_const,
    'weight': None,
    'weight_derivative': None,
}

COUPLINGS_TEST_PARAMS = {
    'ell': 2,
    'mu': (1, -1, 1),
}

@pytest.mark.parametrize("coupling_type", ['angular', 'radial', 'RSD'])
def test_Couplings(discrete_spectrum, coupling_type):

    couplings_partial = Couplings(discrete_spectrum)
    couplings_full = \
        Couplings(discrete_spectrum, survey_specs=COUPLINGS_SURVEY_SPECS)

    partial_couplings_fixed_index = couplings_partial.couplings_fixed_index(
        mu=COUPLINGS_TEST_PARAMS['mu'],
        coupling_type=coupling_type
    )
    full_couplings_fixed_index = couplings_full.couplings_fixed_index(
        mu=COUPLINGS_TEST_PARAMS['mu'],
        coupling_type=coupling_type
    )

    assert np.allclose(
        partial_couplings_fixed_index[COUPLINGS_TEST_PARAMS['ell']],
        full_couplings_fixed_index[COUPLINGS_TEST_PARAMS['ell']]
    )


TWO_POINT_SURVEY_SPECS = {
    'mask': unit_const,
    'selection': None,
    'weight': None,
    'weight_derivative': None,
}
TWO_POINT_TEST_PARAMS = dict(
    nbar=1e-3,
    b_1=2.,
    f_nl=0.,
)


@pytest.mark.slow
@pytest.mark.parametrize("pivot,part", [('spectral', 'signal')])
def test_TwoPointFunction(discrete_spectrum, pivot, part):

    two_point_model = TwoPointFunction(
        discrete_spectrum,
        growth_rate=None,
        cosmo=fiducial_cosmology,
        survey_specs=TWO_POINT_SURVEY_SPECS
    )
    covariance_matrix = two_point_model.two_point_covariance(
        pivot, part=part, diag=True, **TWO_POINT_TEST_PARAMS
    )

    variance_model = TwoPointFunction(
        discrete_spectrum, growth_rate=None, cosmo=fiducial_cosmology,
    )
    variance_matrix = variance_model.mode_variance(
        pivot, part=part, **TWO_POINT_TEST_PARAMS
    )

    two_point_model_angular_reduction = TwoPointFunction(
        discrete_spectrum, growth_rate=None, cosmo=fiducial_cosmology
    )
    covariance_matrix_angular_reduction = \
        two_point_model_angular_reduction.two_point_covariance(
            pivot, part=part, diag=True, **TWO_POINT_TEST_PARAMS
        )

    assert np.diag(covariance_matrix) == pytest.approx(variance_matrix)
    assert np.diag(covariance_matrix) \
        == pytest.approx(np.diag(covariance_matrix_angular_reduction))
