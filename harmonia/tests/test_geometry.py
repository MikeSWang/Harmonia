import numpy as np
import pytest
from nbodykit.cosmology import Planck13, Planck15

from harmonia.cosmology.geometry import (
    differential_AP_distortion,
    redshift_from_distance,
)


@pytest.mark.parametrize(
    "z",
    [
        (1.0,),
        (2.0,),
        (3.0,),
    ]
)
def test_differential_AP_distortion(z):

    assert np.isclose(
        differential_AP_distortion(
            Planck13.comoving_distance, Planck15.comoving_distance
        )(z), 1.,
        rtol=1.e-3
    ), "Incorrect computation of differential AP distortion."


@pytest.mark.skip(reason="Unimplemented.")
def test_redshift_from_distance():
    print("Unimplemented test.")
    raise NotImplementedError
