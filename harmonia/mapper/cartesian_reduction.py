"""
Cartesian Fourier reduction (:mod:`~harmonia.mapper.cartesian_reduction`)
===========================================================================

Reduce discrete catalogues to compressed Fourier-space maps in Cartesian
coordinates.

.. autosummary::

    CartesianMap

|

"""
import logging


class CartesianMap:
    """Compressed Cartesian Fourier-space map.

    """

    _logger = logging.getLogger('CartesianMap')
