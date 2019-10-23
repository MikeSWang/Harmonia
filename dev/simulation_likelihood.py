"""Simulation spherical likelihood for primordial non-Gaussianity.

"""
import numpy as np
from nbodykit.lab import CSVCatalog
from scipy.interpolate import interp1d

from likelihood_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
)
from harmonia.mapper import SphericalMap
from harmonia.reader import TwoPointFunction
from spherical_likelihood import (
    spherical_map_f_nl_chi_square as f_nl_chi_square,
    # spherical_map_f_nl_likelihood as f_nl_likelihood,
)

PK_FILE_ROOT = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"
HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]


def initialise():
    """Initialise from input parameters and return runtime information.

    Returns
    -------
    runtime_info : str
        Runtime information.

    Raises
    ------
    AttributeError
        If a required input parameter is missing.

    """
    global prior_range, num_sample, pivot, nbar, bias, kmax, boxsize, \
        input_file

    try:
        prior_range = params.prior_range
        num_sample = params.num_sample
        pivot = params.pivot
        nbar = params.nbar
        bias = params.bias
        kmax = params.kmax
        boxsize = params.boxsize
        input_file = params.input_file
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global Plin, growth_rate

    k_points, Pk_points = \
        np.loadtxt("".join([PATHIN, script_name, "/", PK_FILE_ROOT, ".txt"])).T

    Plin = interp1d(k_points, Pk_points, assume_sorted=True)

    growth_rate = None

    runtime_info = "-(boxsize={},kmax={})".format(
        format_float(boxsize, 'intdot'),
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

    disc = DiscreteSpectrum(boxsize/2, 'Dirichlet', kmax)

    global overdensity

    catalogue = CSVCatalog(
        "".join([PATHIN, script_name, "/", input_file, ".txt"]),
        HEADINGS
    )

    catalogue.attrs['BoxSize'] = boxsize
    catalogue['Position'] = catalogue['x'][:, None] * [1, 0, 0] \
        + catalogue['y'][:, None] * [0, 1, 0] \
        + catalogue['z'][:, None] * [0, 0, 1]

    spherical_map = SphericalMap(disc, catalogue)

    n_coeff, nbar_coeff = spherical_map.transform()

    overdensity = [
        n_coeff[ell] - nbar_coeff[ell]
        for ell in np.sort(disc.degrees)
    ]

    field_vector = SphericalArray \
        .build(disc=disc, filling=overdensity) \
        .unfold(pivot, return_only='data')

    two_point_model = \
        TwoPointFunction(disc, f_0=growth_rate, power_spectrum=Plin)

    sample_parameters = np.linspace(*prior_range, num=num_sample+1)

    sample_chi_square = f_nl_chi_square(
        sample_parameters,
        field_vector,
        pivot,
        two_point_model,
        nbar,
        bias
    )

    output_data = {
        'f_nl': [sample_parameters],
        'chi_square': [sample_chi_square],
        # 'likelihood': [sample_likelihood],
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
        np.save("".join([base_path, "/", filename, "-dv.npy"]), overdensity)
        np.save("".join([base_path, "/", filename, ".npy"]), output_data)


if __name__ == '__main__':

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
