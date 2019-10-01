"""
*******************************************************************************
Miscellaneous collections (:mod:`~harmonia.collections`)
*******************************************************************************

Collect utility functions and settings.

"""
from .style import harmony
from .utils import (
    allocate_tasks,
    allocate_segments,
    bin_edges_from_centres,
    binary_search,
    cartesian_to_spherical,
    clean_warnings,
    collate,
    confirm_directory_path,
    covar_to_corr,
    format_float,
    get_filename,
    mpi_compute,
    normalise_vector,
    overwrite_protection,
    smooth_by_bin_average,
    spherical_to_cartesian,
    spherical_indicator,
    zero_const,
    unit_const,
)
