import pytest
from nbodykit.lab import cosmology as cosmo

import testrc
from harmonia.algorithms.fields import (
    generate_regular_grid,
    generate_gaussian_random_field,
    generate_lognormal_random_field,
    threshold_clip,
    lognormal_transform,
    poisson_sample,
    populate_particles
    )
