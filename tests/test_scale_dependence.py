import pytest

from harmonia.cosmology import fiducial_cosmology
from harmonia.cosmology.scale_dependence import (
    scale_modification,
    scale_dependent_bias,
    scale_modified_power_spectrum,
)


@pytest.mark.parametrize("cosmo,redshift", [(fiducial_cosmology, 0.)])
def test_scale_modification(cosmo, redshift):
    assert callable(scale_modification(cosmo, redshift))


@pytest.mark.parametrize(
    "bz_const,f_nl,cosmo",
    [(2., -10., fiducial_cosmology)]
)
def test_scale_dependent_bias(bz_const, f_nl, cosmo):
    assert callable(scale_dependent_bias(bz_const, f_nl, cosmo))


@pytest.mark.parametrize(
    "f_nl,bz_const,cosmo",
    [(-10., 2., fiducial_cosmology)]
)
def test_scale_modified_power_spectrum(f_nl,bz_const,cosmo):
    assert callable(scale_modified_power_spectrum(f_nl,bz_const,cosmo))
