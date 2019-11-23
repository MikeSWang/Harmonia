"""
***************************************************************************
Cosmological models (:mod:`~harmonia.cosmology`)
***************************************************************************

Compute cosmological quantities for a specified model with
:class:`nbodykit.cosmology`.  The default cosmology is set by *Planck* 2015
results [`arXiv:1502.01589 <https://arxiv.org/abs/1502.01589>`_].

"""
from nbodykit import cosmology

from .geometry import differential_AP_distortion, fiducial_distance
from .scale_dependence import (
    modified_power_spectrum,
    scale_dependence_modification,
    scale_dependent_bias,
)

fiducial_cosmology = cosmology.Planck15
""":class:`nbodykit.cosmology.cosmology.Cosmology`: Default *Planck15*
cosmology.

"""
