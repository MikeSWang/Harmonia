"""
***************************************************************************
Cosmological models (:mod:`~harmonia.cosmology`)
***************************************************************************

Compute quantities related to a specified cosmological model with
:class:`nbodykit.cosmology`.

"""
from nbodykit import cosmology

from .geometry import differential_AP_distortion, fiducial_distance
from .scale_dependence import (
    modified_power_spectrum,
    scale_dependence_modification,
    scale_dependent_bias,
)

FIDUCIAL_COSMOLOGY = cosmology.Planck15
""":class:`nbodykit.cosmology.cosmology.Cosmology`: Default *Planck15*
cosmology [`arXiv:1502.01589 <https://arxiv.org/abs/1502.01589>`_].

"""
