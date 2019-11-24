import pytest

from harmonia.cosmology import fiducial_cosmology
from harmonia.cosmology.scale_dependence import (
    modified_power_spectrum,
    scale_dependence_modification,
    scale_dependent_bias,
)


@pytest.mark.parametrize("cosmo,redshift", [(fiducial_cosmology, 0.)])
def test_scale_dependence_modification(cosmo, redshift):
    assert callable(scale_dependence_modification(cosmo, redshift))


@pytest.mark.parametrize(
    "b_1,f_nl,cosmo",
    [(2., -10., fiducial_cosmology)]
)
def test_scale_dependent_bias(b_1, f_nl, cosmo):
    assert callable(scale_dependent_bias(b_1, f_nl, cosmo))


@pytest.mark.parametrize(
    "f_nl,b_1,cosmo",
    [(-10., 2., fiducial_cosmology)]
)
def test_modified_power_spectrum(f_nl, b_1, cosmo):
    assert callable(modified_power_spectrum(f_nl, b_1, cosmo))
