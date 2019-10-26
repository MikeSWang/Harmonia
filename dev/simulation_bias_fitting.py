"""Bias fitting for simulations without local primordial non-Gaussianity.

"""
import warnings

import numpy as np
from nbodykit.lab import CSVCatalog
from scipy.interpolate import interp1d

from likelihood_rc import PATHIN, PATHOUT, params
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import confirm_directory_path, format_float
from harmonia.mapper import SphericalMap
from harmonia.reader import TwoPointFunction

PK_FILE_ROOT = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"
HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]


def bias_chi_square(bias, dat_vector, two_point_model, pivot, nbar,
                    by_mode=False):
    """Compute chi-square for a zero-centred data vector and its variance.

    Parameters
    ----------
    bias : float, array_like
        Bias at which chi-square values are sampled.
    dat_vector : float or complex, array_like
        1-d data vector.
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function model without scale modification.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    by_element : bool, optional
        If `True` (default is `False`), return chi-square from each `data`
        element.

    Returns
    -------
    chi_square : float, array_like
        Chi-square value(s).

    """
    chi_square = np.zeros(len(np.atleast_1d(bias)))

    for idx, b in enumerate(bias):
        var_vector = two_point_model.mode_variance(pivot, nbar=nbar, b_const=b)

        chisq = np.abs(dat_vector)**2 / var_vector

        if not by_mode:
            chisq = np.sum(chisq)

        chi_square[idx] = chisq

    return chi_square


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
    global prior_range, num_sample, pivot, nbar, kmax, boxsize, input_file

    try:
        prior_range = params.prior_range
        num_sample = params.num_sample
        pivot = params.pivot
        nbar = params.nbar
        kmax = params.kmax
        boxsize = params.boxsize
        input_file = params.input_file
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global Plin

    k_points, Pk_points = \
        np.loadtxt("".join([PATHIN, SCRIPT_NAME, "/", PK_FILE_ROOT, ".txt"])).T

    Plin = interp1d(k_points, Pk_points, assume_sorted=True)

    runtime_info = "-(prior={},pivot={},nbar={},kmax={})".format(
        str(prior_range).replace(" ", ""),
        pivot,
        format_float(nbar, 'sci'),
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

    catalogue = CSVCatalog(
        "".join([PATHIN, SCRIPT_NAME, "/", input_file, ".txt"]),
        HEADINGS
    )
    catalogue.attrs['BoxSize'] = boxsize
    catalogue['Position'] = catalogue['x'][:, None] * [1, 0, 0] \
        + catalogue['y'][:, None] * [0, 1, 0] \
        + catalogue['z'][:, None] * [0, 0, 1]

    spherical_map = SphericalMap(disc, catalogue, mean_density_data=nbar)

    field_vector = SphericalArray \
        .build(disc=disc, filling=spherical_map.density_constrast()) \
        .unfold(pivot, return_only='data')

    two_point_model = TwoPointFunction(disc, power_spectrum=Plin)

    sample_parameters = np.linspace(*prior_range, num=num_sample+1)

    sample_chi_square = bias_chi_square(
        sample_parameters,
        field_vector,
        two_point_model,
        pivot,
        nbar,
        by_mode=True
    )

    output_data = {
        'bias': [sample_parameters],
        'chi_square': [sample_chi_square],
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
    base_path = f"{PATHOUT}simu_bias_fitting"
    assert confirm_directory_path(base_path)

    filename = f"{input_file}{program_tag}"
    if save:
        np.save("".join([base_path, "/", filename, ".npy"]), output_data)


if __name__ == '__main__':

    warnings.simplefilter("ignore", UserWarning)

    SCRIPT_NAME = "simulation_likelihood"

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
