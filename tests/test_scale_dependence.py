import pytest

from harmonia.cosmology import FIDUCIAL_COSMOLOGY
from harmonia.cosmology.scale_dependence import (
    modified_power_spectrum,
    scale_dependence_modification,
    scale_dependent_bias,
)


@pytest.mark.parametrize("cosmo,redshift", [(FIDUCIAL_COSMOLOGY, 0.)])
def test_scale_dependence_modification(cosmo, redshift):
    assert callable(scale_dependence_modification(cosmo, redshift))


@pytest.mark.parametrize(
    "b_1,f_nl,cosmo",
    [(2., -10., FIDUCIAL_COSMOLOGY)]
)
def test_scale_dependent_bias(b_1, f_nl, cosmo):
    assert callable(scale_dependent_bias(b_1, f_nl, cosmo))


@pytest.mark.parametrize(
    "b_1,f_nl,cosmo",
    [(2., -10., FIDUCIAL_COSMOLOGY)]
)
def test_modified_power_spectrum(b_1, f_nl, cosmo):
    assert callable(modified_power_spectrum(b_1, f_nl, cosmo))
