"""Predict 2-point function values from a set of cosmological parameters.

"""
import numpy as np
from mpi4py import MPI
from nbodykit.lab import cosmology

from agreement_rc import PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    unit_const,
    zero_const,
)
from harmonia.reader import TwoPointFunction


def initialise():
    """Initialise from input parameters, set up cosmology and return
    runtime information.

    Returns
    -------
    runtime_info : str
        Runtime information.

    Raises
    ------
    AttributeError
        If a required input arameter is missing.

    """
    global pivots, rsd_flag, nbar, bias, redshift, zmax, kmax

    try:
        pivots = params.structure.split(",")
        rsd_flag = params.rsd

        nbar = params.nbar
        bias = params.bias
        redshift = params.redshift
        zmax = params.zmax
        kmax = params.kmax
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global Plin, rmax, beta, cosmo

    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')
    rmax = cosmo.comoving_distance(zmax)
    beta = rsd_flag * cosmo.scale_independent_growth_rate(redshift) / bias

    if len(pivots) > 1:
        pivot_tag = "{}".format(pivots).replace("'", "")
    else:
        pivot_tag = "{}".format(params.structure).replace("'", "")
    pivot_tag = pivot_tag.replace(" ", "")

    if rsd_flag:
        rsd_tag = "{:.2f}".format(beta)
    else:
        rsd_tag = 'none'

    param_tag = "pivots={},nbar={},bias={},beta={},rmax={},kmax={}".format(
        pivot_tag,
        format_float(nbar, 'sci'),
        format_float(bias, 'decdot'),
        rsd_tag,
        format_float(rmax, 'intdot'),
        format_float(kmax, 'sci'),
    )
    runtime_info = "-({})".format(param_tag)
    return runtime_info


def process(runtime_info):
    """Program process.

    Parameters
    ----------
    runtime_info : str
        Program runtime information.

    Returns
    -------
    output_data : dict
        Program output.

    """
    print(runtime_info.strip("-"))

    global couplings

    disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)

    two_point_cosmo = nbar, bias, cosmo
    two_points = TwoPointFunction(
        *two_point_cosmo,
        disc,
        f_0=cosmo.scale_independent_growth_rate(redshift),
        survey_specs=SURVEY_SPECS,
        cosmo_specs=None,
        comm=COMM
    )

    couplings = two_points.couplings

    output_data = {
        pivot: {
            part: two_points.two_point_covariance(pivot, part=part)
            for part in ['signal', 'shotnoise', 'both']
        }
        for pivot in pivots
    }

    return output_data


def finalise(output_data, save=True):
    """Program finalisation with optional data and figure saving.

    Parameters
    ----------
    output_data : dict
        Program output.
    save : bool, optional
        If `True`, aggregate data over all iterations is saved as a
        dictionary.

    Raises
    ------
    AssertionError
        If the output path does not exist.

    """
    base_path = f"{PATHOUT}{script_name}"
    assert confirm_dir(base_path)

    filename = f"{script_name}{program_tag}"
    if save:
        if COMM is not None:
            if COMM.rank == 0:
                np.save(
                    "".join([base_path, "/", filename, "-couplings.npy"]),
                    couplings
                )
                np.save(
                    "".join([base_path, "/", filename, ".npy"]),
                    output_data
                )
        else:
            np.save(
                "".join([base_path, "/", filename, "-couplings.npy"]),
                couplings
            )
            np.save(
                "".join([base_path, "/", filename, ".npy"]),
                output_data
            )


if __name__ == '__main__':

    COMM = MPI.COMM_WORLD
    SURVEY_SPECS = {
        'mask': unit_const,
        'selection': unit_const,
        'weight': unit_const,
        'weight_derivative': zero_const
    }
    SURVEY_SPECS = None

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
