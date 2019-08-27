import numpy as np
import pytest

import testrc
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


@pytest.fixture(scope='module')
def spherical_array():
    disc = DiscreteSpectrum(**TEST_PARAMS)
    fill = None
    return SphericalArray(disc.degrees, disc.depths, filling=fill)



