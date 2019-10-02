"""
***************************************************************************
Cosmological models (:mod:`~harmonia.cosmology`)
***************************************************************************

Compute cosmological quantities for a specified model with
:class:`nbodykit.cosmology`.  The default cosmology is set by *Planck* 2015
results [`arXiv:1502.01589 <https://arxiv.org/abs/1502.01589>`_].

"""
from .geometry import (
    fiducial_cosmology,
    fiducial_distance,
    redshift_to_distance,
)
