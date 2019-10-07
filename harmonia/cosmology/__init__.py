"""
***************************************************************************
Cosmological models (:mod:`~harmonia.cosmology`)
***************************************************************************

Compute cosmological quantities for a specified model with
:class:`nbodykit.cosmology`.  The default cosmology is set by *Planck* 2015
results [`arXiv:1502.01589 <https://arxiv.org/abs/1502.01589>`_].

"""
from nbodykit import cosmology

from .geometry import (
    fiducial_distance,
    redshift_to_distance,
)
from .scale_dependence import (
    scale_dependent_bias,
    non_gaussianity_biased_power_spectrum,
)

fiducial_cosmology = cosmology.Planck15
""":class:`nbodykit.cosmology.cosmology.Cosmology`: Default Planck15
cosmology.

"""
