from harmonia.cosmology import fiducial_cosmology
from harmonia.cosmology.scale_dependence import (
    scale_dependent_bias,
    scale_modified_power_spectrum,
)

TEST_PARAMETERS = dict(
    f_nl=-5.,
    b_const=2.,
    cosmo=fiducial_cosmology,
)


def test_scale_dependent_bias():
    assert callable(scale_dependent_bias(**TEST_PARAMETERS))


def test_scale_modified_power_spectrum():
    assert callable(scale_modified_power_spectrum(**TEST_PARAMETERS))
