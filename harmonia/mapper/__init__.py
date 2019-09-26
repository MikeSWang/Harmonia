"""
*******************************************************************************
Fourier-space mapper (:mod:`~harmonia.mapper`)
*******************************************************************************

Build and process discrete catalogues into Fourier-space maps.

"""
from .cartesian_reduction import CartesianMap
from .catalogue_maker import (
    spherical_indicator,
    RandomCatalogue,
    NBKCatalogue,
    LogNormalCatalogue,
    GaussianCatalogue,
)
from .spherical_transform import SphericalMap
