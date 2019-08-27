"""
*******************************************************************************
Fourier-space mapper (:mod:`~harmonia.mapper`)
*******************************************************************************

Build and process discrete catalogues into Fourier-space maps.

"""
from .catalogue_maker import (
    spherical_indicator, RandomCatalogue, LognormalCatalogue
    )
from .cartesian_reduction import CartesianMap
from .spherical_transform import SphericalMap