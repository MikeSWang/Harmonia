import sys
sys.path.insert(0, "../")

import numpy as np

from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.algorithms.morph import SphericalArray

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
    disc = DiscreteSpectrum()
    return SphericalArray(**TEST_PARAMS)

def test_radial_spherical_int():
    assert 0
