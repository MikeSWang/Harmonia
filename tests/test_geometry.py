import pytest

from harmonia.cosmology import fiducial_cosmology
from harmonia.cosmology.geometry import fiducial_distance


def test_fiducial_distance():
    assert fiducial_distance(z=0.) == pytest.approx(0.)
