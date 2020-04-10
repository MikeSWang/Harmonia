import numpy as np
import pytest
from nbodykit.cosmology import Planck15

from harmonia.surveyor.coordinates import (
    cartesian_to_spherical,
    sky_to_spherical,
    spherical_to_cartesian,
    spherical_to_sky,
    to_box_coords,
)


@pytest.mark.parametrize(
    "vec,value",
    [
        (
            [[1, 3, -5], [0.2, -0.88, -10]],
            [
                [5.916079783, 2.577650012, 1.249045772],
                [10.04063743, 3.051592333, 4.935866174],
            ]
        ),
    ]
)
def test_cartesian_to_spherical(vec, value):
    assert np.allclose(cartesian_to_spherical(vec), value), \
        "Incorrect coordinate conversion from Cartesian to spherical."


@pytest.mark.parametrize(
    "vec,value",
    [
        (
            [
                [5.916079783, 2.577650012, 1.249045772],
                [10.04063743, 3.051592333, 4.935866174],
            ],
            [[1, 3, -5], [0.2, -0.88, -10]]
        ),
    ]
)
def test_spherical_to_cartesian(vec, value):
    assert np.allclose(spherical_to_cartesian(vec), value), \
        "Incorrect coordinate conversion from Cartesian to Cartesian."


@pytest.mark.parametrize(
    "sky_coords,z_to_r,values",
    [
        ([0., 90.,], None, [np.pi/2, np.pi/2]),
        ([0., 0., 180.], Planck15.comoving_distance, [0., np.pi/2, np.pi]),
    ]
)
def test_sky_to_spherical(sky_coords, z_to_r, values):
    assert np.allclose(sky_to_spherical(sky_coords, z_to_r), values), \
        "Incorrect coordinate transform from sky to spherical."


@pytest.mark.parametrize(
    "spherical_coords,z_from_r,values",
    [
        ([1., np.pi/2, np.pi/2], None, [0., 90.,]),
    ]
)
def test_spherical_to_sky(spherical_coords, z_from_r, values):
    assert np.allclose(spherical_to_sky(spherical_coords, z_from_r), values), \
        "Incorrect coordinate transform from spherical to sky."


@pytest.mark.parametrize(
    "native_coord_system,box_centre,conversion_kwargs,coords,values",
    [
        ('null', 1., None, [1., 1., 1.], 0),
        ('sky', None, dict(z_from_r=lambda r: r), [1., 1., 1.],
         np.sqrt(3) \
            + (90 - np.rad2deg(np.arcsin(np.sqrt(2/3)))) + np.rad2deg(np.pi/4)
         ),
    ]
)
def test_to_box_coords(native_coord_system, box_centre, conversion_kwargs,
                       coords, values):

    @to_box_coords(
        native_coord_system=native_coord_system,
        box_centre=box_centre,
        conversion_kwargs=conversion_kwargs
    )
    def test_func(coords):
        return np.sum(coords, axis=-1)

    assert np.allclose(test_func(coords), values), \
        "Incorrect coordinate transform to box coordinates."
