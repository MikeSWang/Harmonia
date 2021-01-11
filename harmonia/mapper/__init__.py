"""
***************************************************************************
Catalogue mapper (:mod:`~harmonia.mapper`)
***************************************************************************

Build and process discrete catalogues into Fourier-space maps.

.. note::

    Unless otherwise specified, the length dimension in the module is
    in units of Mpc/:math:`h`.

"""
from .catalogue_maker import (
    RandomCatalogue,
    SourceCatalogue,
    SphericalFKPCatalogue,
    spherical_indicator,
)
from .map_transform import CartesianMap, SphericalMap
