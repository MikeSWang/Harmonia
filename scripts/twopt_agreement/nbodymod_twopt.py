"""Predict 2-point function values for fixed-epoch N-body simulations.

"""
import numpy as np
from mpi4py import MPI
from scipy.interpolate import interp1d

from agreement_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    unit_const,
    zero_const,
)
from harmonia.reader import TwoPointFunction

PK_FILE_ROOT = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"


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
    global pivots, nbar, bias, rmax, kmax

    try:
        pivots = params.pivots.split(",")
        nbar = params.nbar
        bias = params.bias
        rmax = params.boxside / 2
        kmax = params.kmax
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global Plin, growth_rate

    k_points, Pk_points = np.loadtxt(
        "".join([PATHIN, script_name, "/", PK_FILE_ROOT, ".txt"])
    ).T

    Plin = interp1d(k_points, Pk_points, assume_sorted=True)
    growth_rate = None

    if len(pivots) > 1:
        pivot_tag = "{}".format(pivots)
    else:
        pivot_tag = "{}".format(pivots).lstrip("[").rstrip("]")

    if isinstance(growth_rate, float):
        growth_status = format_float(growth_rate, 'decdot')
    else:
        growth_status = 'none'

    runtime_info = \
        "-(pivots={},nbar={:.2e},b1={:.2f},f0={},rmax={},kmax={})".format(
            pivot_tag.replace("'", "").replace(" ", ""),
            nbar,
            bias,
            growth_status,
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
        power_spectrum=Plin,
        comm=COMM,
        survey_specs=None,  # SURVEY_SPECS
    )

    two_points = TwoPointFunction(disc, **kwargs)

    couplings = two_points.couplings

    output_data = {
        pivot: {
            part: two_points.two_point_covariance(
                pivot,
                nbar=nbar,
                b_const=bias,
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

    SURVEY_SPECS = {
        'mask': unit_const,
        'selection': unit_const,
        'weight': unit_const,  # None
        'weight_derivative' : zero_const,  # None
    }

    COMM = MPI.COMM_WORLD
    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
