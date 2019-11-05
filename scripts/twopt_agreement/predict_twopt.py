"""Predict 2-point function values from a set of cosmological parameters.

"""
import numpy as np
from mpi4py import MPI

from agreement_rc import PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    unit_const,
    zero_const,
)
from harmonia.cosmology import fiducial_cosmology
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
        If a required input parameter is missing.

    """
    global pivots, rsd_flag, nbar, bias, redshift, kmax, rmax, growth_rate

    try:
        pivots = params.pivots.split(",")
        rsd_flag = params.rsd
        nbar = params.nbar
        bias = params.bias
        redshift = params.redshift
        kmax = params.kmax
        rmax = fiducial_cosmology.comoving_distance(params.zmax)
        growth_rate = \
            fiducial_cosmology.scale_independent_growth_rate(redshift)
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    if len(pivots) > 1:
        pivot_tag = "{}".format(pivots)
    else:
        pivot_tag = "{}".format(pivots).lstrip("[").rstrip("]")

    if rsd_flag:
        rsd_tag = "{:.2f}".format(growth_rate)
    else:
        growth_rate = None
        rsd_tag = 'none'

    runtime_info = "-(pivots={},nbar={},b1={},f0={},rmax={},kmax={})".format(
        pivot_tag.replace("'", "").replace(" ", ""),
        format_float(nbar, 'sci'),
        format_float(bias, 'decdot'),
        rsd_tag,
        format_float(rmax, 'intdot'),
        format_float(kmax, 'sci'),
    )
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

    kwargs = dict(
        f_0=growth_rate,
        cosmo=fiducial_cosmology,
        comm=COMM,
    )

    two_points = TwoPointFunction(disc, **kwargs)

    couplings = two_points.couplings

    output_data = {
        pivot: {
            part: two_points.two_point_covariance(
                pivot,
                nbar=nbar,
                b_10=bias,
                part=part
            )
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
    SURVEY_SPECS_FUTURE = {
        'mask': unit_const,
        'selection': unit_const,
        'weight': unit_const,
        'weight_derivative': zero_const
    }

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
