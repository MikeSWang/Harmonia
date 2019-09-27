import pytest

from harmonia.cosmology.geometry import (
    fiducial_cosmology,
    fiducial_distance,
    redshift_to_distance,
)


def test_fiducial_distance():
    assert fiducial_distance(z=0.) == pytest.approx(0.)


@pytest.mark.parametrize("redshift", [0., 0.55, 1.5])
def test_redshift_to_distance(redshift):
    assert fiducial_distance(redshift) == pytest.approx(
        redshift_to_distance(
            redshift,
            fiducial_cosmology.h,
            fiducial_cosmology.Omega0_b,
            fiducial_cosmology.Omega0_cdm,
        ),
    )
