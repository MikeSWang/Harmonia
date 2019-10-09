"""Construct spherical likelihood from data realisation and modelling.

"""
import numpy as np

from harmonia.mapper import LogNormalCatalogue, NBKCatalogue, SphericalMap
from harmonia.reader import TwoPointFunction


def fnl_parametrised_covariance(f_nl, bias, power_spectrum, disc=None):
    r"""Parametrised covariance matrix by local primordial non-Gaussianity.

    Parameters
    ----------
    f_nl : float, array_like
        Local primordial non-Gaussianity parameter.

    Returns
    -------
    covariance : complex or float, array_like
        Covariance matrix evaluated at input parameter(s).

    """



def complex_normal_pdf(param_point, data_vector, model_params):
    """

    """

