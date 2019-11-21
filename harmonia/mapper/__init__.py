"""
***************************************************************************
Fourier-space mapper (:mod:`~harmonia.mapper`)
***************************************************************************

Build and process discrete catalogues into Fourier-space maps, and probe
the survey window.

"""
from .cartesian_reduction import CartesianMap
from .catalogue_maker import (
    GaussianCatalogue,
    LogNormalCatalogue,
    NBKCatalogue,
    RandomCatalogue,
)
from .spherical_transform import SphericalMap
from .window_synthesis import SurveyWindow
