import numpy as np
import pytest
from nbodykit.lab import cosmology as cosmo

from testrc import approx
from harmonia.algorithms.fields import (
    generate_regular_grid,
    generate_gaussian_random_field,
    generate_lognormal_random_field,
    threshold_clip,
    lognormal_transform,
    poisson_sample,
    populate_particles
)

power_spectrum = cosmo.LinearPower(cosmo.Planck15, redshift=0.)


@pytest.mark.parametrize(
    "cell_size,num_mesh",
    [
        (1., 32),
        (0.1, 64),
    ],
)
def test_generate_regular_grid(cell_size, num_mesh):
    grid_norm = generate_regular_grid(cell_size, num_mesh, variable='norm')
    assert np.min(grid_norm) == approx(
        np.sqrt(3) * cell_size / 2,
    )
    assert np.max(grid_norm) == approx(
        np.sqrt(3) * (num_mesh - 1) * cell_size / 2,
    )


@pytest.mark.parametrize(
    "boxsize,num_mesh",
    [
        (100, 32),
        (200, 64),
    ],
)
def test_generate_gaussian_random_field(boxsize, num_mesh):
    field, vec_field = generate_gaussian_random_field(
        boxsize,
        num_mesh,
        power_spectrum,
        clip=False,
        return_disp=True,
    )
    assert np.shape(field) == (num_mesh,) * 3
    assert len(vec_field) == 3
    assert all([np.shape(vfield) == (num_mesh,) * 3 for vfield in vec_field])


@pytest.mark.parametrize(
    "boxsize,num_mesh",
    [
        (100, 32),
        (200, 128),
    ],
)
def test_generate_lognormal_random_field(boxsize, num_mesh):
    field, vec_field = generate_lognormal_random_field(
        boxsize,
        num_mesh,
        power_spectrum,
        return_disp=True,
    )
    assert np.shape(field) == (num_mesh,) * 3
    assert len(vec_field) == 3
    assert all([np.shape(vfield) == (num_mesh,) * 3 for vfield in vec_field])


@pytest.mark.parametrize(
    "num_mesh",
    [128, 64],
)
def test_threshold_clip(num_mesh):
    density_contrast = 10*np.random.randn(num_mesh, num_mesh)
    with pytest.warns(RuntimeWarning, match=".* field values are clipped."):
        clipped_field = threshold_clip(density_contrast, threshold=-1.)
        assert np.min(clipped_field) == approx(-1.)


def test_lognormal_transform():
    with pytest.raises(ValueError):
        lognormal_transform(power_spectrum, 'invalid_obj_type')


@pytest.mark.parametrize(
    "shape,mean_density,boxsize",
    [
        ((5, 4, 3), 1e-3, 50),
    ],
)
def test_poisson_sample(shape, mean_density, boxsize):
    with pytest.raises(ValueError):
        poisson_sample(np.ones(shape), mean_density, boxsize)


@pytest.mark.parametrize(
    "num_mesh,mean_density,boxsize",
    [
        (128, 1e-3, 50),
    ],
)
def test_populate_particles(num_mesh, mean_density, boxsize):
    fake_sampled_field = np.ones((num_mesh,)*3)
    fake_vel_offset_fields = [fake_sampled_field,]*3
    pos, disp = populate_particles(
        fake_sampled_field,
        mean_density,
        boxsize,
        vel_offset_fields=fake_vel_offset_fields,
    )
    assert np.size(pos, axis=1) == 3
    assert np.shape(pos) == np.shape(disp)
