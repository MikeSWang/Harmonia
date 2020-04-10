"""
***************************************************************************
Survey factory (:mod:`~harmonia.surveyor`)
***************************************************************************

Produce from survey definition and specifications the quantities needed
for map analysis, e.g. mask map, selection function, mask/window function
multipoles, covariance estimates and data compressor.

.. note::

    Unless otherwise specified, the length dimension in the module is
    in units of Mpc/:math:`h`.

"""
from .definition import (
    generate_compression_matrix,
    generate_mask_by_sky_fraction,
    generate_mask_from_map,
    generate_selection_by_cut,
    generate_selection_by_distribution,
    generate_selection_from_samples,
    generate_selection_samples,
)
from .coordinates import (
    cartesian_to_spherical,
    sky_to_spherical,
    spherical_to_cartesian,
    spherical_to_sky,
    to_box_coords,
)
# NOTE: No import from ``.synthesis`` to avoid circular import.
