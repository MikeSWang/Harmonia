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

from .geometry import differential_AP_distortion, redshift_from_distance
from .scale_dependence import (
    modified_power_spectrum,
    scale_dependence_modification,
    scale_dependent_bias,
)


class BaseModel(Cosmology):
    """Fixed (fiducial) cosmological model built with a parameter
    dictionary read from an external file, e.g. with file contents

    .. code-block::

        {
            'h': 0.70,
            'Omega0_b': 0.044,
            'Omega0_cdm': 0.226,
            'sigma8': 0.8,
        }

    Parameters
    ----------
    source_file : *str or* :class:`pathlib.Path`
        Cosmological parameter file (as a Python dictionary).
    comm : :class:`mpi4py.Comm` *or None*, optional
        MPI communicator (default is `None`).

    """

    def __init__(self, source_file, comm=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.comm = comm

        with open(source_file) as cosmo_source:
            source_params = {
                k: round(v, ndigits=8)
                for k, v in eval(cosmo_source.read()).items()
            }

        cosmo_params = {
            param_name: param_value
            for param_name, param_value in source_params.items()
            if param_name != 'sigma8'
        }

        super().__init__(**cosmo_params)

        self.match(sigma8=source_params['sigma8'])

        if self.comm is None or self.comm.rank == 0:
            self.logger.info(
                "Created cosmological model with parameters:\n%s.",
                pformat(source_params)
            )
