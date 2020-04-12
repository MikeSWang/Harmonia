"""
***************************************************************************
Cosmological modelling (:mod:`~harmonia.cosmology`)
***************************************************************************

Provide general and fiducial cosmological models and compute derived model
quantities.

Cosmological models are implemented by `nbodykit` (see `documentation
<https://nbodykit.readthedocs.io>`_).

.. autosummary::

    BaseModel

|

"""
import logging
from pprint import pformat

from nbodykit.cosmology import Cosmology

from .geometry import differential_AP_distortion
from .scale_dependence import (
    modified_power_spectrum,
    scale_dependence_modification,
    scale_dependent_bias,
)


class BaseModel(Cosmology):
    """Fixed (fiducial) cosmological model built with a parameter
    dictionary read from an external file.

    Parameters
    ----------
    source_file : *str or* :class:`pathlib.Path`
        Cosmological parameter file (as a Python dictionary).

    """

    def __init__(self, source_file):

        logger = logging.getLogger(self.__class__.__name__)

        with open(source_file) as cosmo_source:
            source_params = {
                k: round(v, ndigits=8)
                for k, v in eval(cosmo_source.read()).items()
            }

        cosmo_params = {
            k: v for k, v in source_params.items() if k != 'sigma8'
        }

        super().__init__(**cosmo_params)

        self.match(sigma8=source_params['sigma8'])

        logger.info(
            "Created cosmological model with parameters:\n%s.",
            pformat(source_params)
        )
