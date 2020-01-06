"""Export sampled likelihood.

"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))

from view_likelihood import view_pdf, view_contour
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    collate_data_files,
    confirm_directory_path,
    harmony,
    overwrite_protection,
)


def safe_save(data, path, file):
    """Safely save data by checking overwrite protections.

    Parameters
    ----------
    data : array_like
        Data to be saved.
    path, file : str or :class:`pathlib.Path`
        Path and file name for the data to be saved.

    """
    if overwrite_protection(path, file):
        try:
            np.save(path/file, data)
        except TypeError:
            np.save(path + file, data)


def filter_data(full_data, remove_degrees=()):
    """Filter data.

    Parameters
    ----------
    full_data : dict
        Collated raw data.
    remove_degrees : int, array_like, optional
        If not an empty tuple (default), modes whose spherical degree is an
        element are removed from the data vector and parametrised
        covariance.

    Returns
    -------
    filtered_data : dict
        Filtered data.

    """
    disc = DiscreteSpectrum(BOXSIZE/2, 'dirichlet', KHYB)

    index_vector = SphericalArray\
        .build(disc=disc)\
        .unfold(PIVOT, return_only='index')

    excluded_deg = np.fromiter(
        map(lambda index: index[0] in remove_degrees, index_vector),
        dtype=bool
    )

    likelihood_contributions = \
        full_data['spherical_likelihood'][..., ~excluded_deg]

    filtered_data = full_data
    filtered_data.update(
        {'spherical_likelihood': np.sum(likelihood_contributions, axis=-1)}
    )

    return filtered_data


def read_data(collate_data=False, load_data=False, save=False):
    """Collate, load and export likelihood outputs.

    Parameters
    ----------
    collate_data, load_data, save : bool, optional
        If `True` (default is `False`), collate, load or save data.

    Returns
    -------
    output : dict
        Read output data.

    """
    scr_dir_path = PATHOUT/script_name
    collate_path = scr_dir_path/"collated"

    if MAP == "hybrid":
        search_root = f"map={MAP},*knots=[{KHYB},{KMAX}],*{PRIOR}{FIXED}"\
            .replace("=[", "=[[]").replace("],", "[]],")
        program_root = f"map={MAP},knots=[{KHYB},{KMAX}],{PRIOR}{FIXED}"
    else:
        search_root = f"map={MAP},*kmax={KMAX},*{PRIOR}{FIXED}"\
            .replace("=[", "=[[]").replace("],", "[]],")
        program_root = f"map={MAP},kmax={KMAX},{PRIOR}{FIXED}"

    file_name = f"{script_name}-{FILE_ROOT}-({program_root})"
    if collate_data:
        collated_output, _, _ = collate_data_files(
            f"{scr_dir_path}/*{FILE_ROOT}-*-*{search_root}*.npy", 'npy'
        )

        # NOTE: Change this before running.
        # collated_output['parameter'] = np.linspace(-200, 200, 401)
        collated_output['parameter_x'] = np.linspace(-175., 175., 351)
        collated_output['parameter_y'] = np.linspace(2.15, 2.55, 41)

        if MAP == "hybrid":
            filtered_output = filter_data(collated_output, remove_degrees=())
            collated_output['likelihood'] = np.squeeze(
                filtered_output['spherical_likelihood']
                + filtered_output['cartesian_likelihood']
            )
            del collated_output['spherical_likelihood']
            del collated_output['cartesian_likelihood']
        else:
            collated_output['likelihood'] = np.squeeze(
                collated_output[f'{MAP}_likelihood']
            )

        if save:
            confirm_directory_path(collate_path)
            safe_save(collated_output, collate_path, file_name + ".npy")

    if load_data:
        collated_output = np.load(str(collate_path/file_name) + ".npy").item()

    return collated_output, file_name


def view_data(data, savefig=False, **plot_kwargs):
    """Visualise output data.

    Parameters
    ----------
    data : dict
        Data to visualise.
    savefig : bool, optional
        If `True` (default is `False`), save plotted likelihood figure.
    **plot_kwargs
        Keyword arguments to be passed to the plotting routine.

    """
    if MAP == "hybrid":
        program_root = f"map={MAP},knots=[{KHYB},{KMAX}],{PRIOR}{FIXED}"
    else:
        program_root = f"map={MAP},kmax={KMAX},{PRIOR}{FIXED}"
    file_name = f"{script_name}-{FILE_ROOT}-({program_root}).pdf"

    plt.close('all')
    plt.style.use(harmony)

    visual_data = data
    if MAP != "hybrid":
        visual_data['likelihood'] = data[f'{MAP}_likelihood']
    view_contour(
        output, new_output,  r"$f_\mathrm{NL}$", r"$b_1$",
        truth=(0, None),
        precision=(0, 2),
        plot_ranges=([-150.0,150.0], [2.25, 2.55]),
    )
    '''
    view_pdf(
        visual_data,
        r"$f_\mathrm{NL}$", # r"$b_1$", #
        r"$\mathcal{L}(f_\mathrm{NL})$", # r"$\mathcal{L}(b_1)$", #
        **plot_kwargs
    )
    '''

    if savefig:
        plt.savefig(PATHOUT/script_name/file_name)


if __name__ == '__main__':

    PATHOUT = Path("./data/output/")
    FILE_ROOT = "(NG=0.,z=1.)"

    BOXSIZE = 1000.

    MAP = "cartesian"
    KHYB = 0.04
    KMAX = 0.1
    PIVOT = "spectral"
    PRIOR = "bias_prior=[2.15,2.55],fnl_prior=[-175.0,175.0]"
    FIXED = ""

    script_name = f"{MAP}_likelihood"

    output, filename = read_data(
        collate_data=True,
        load_data=False,
        save=True
    )
#    view_data(
#        output, new_output,
#        truth=(0, None),
#        precision=(0, 2),
#        plot_ranges=([-150.0,150.0], [2.25, 2.55]),
#        #savefig=True
#    )
    '''
    view_data(
        output,
        truth=0,
        precision=0,
        plot_range=(),
        scatter_plot=True,
        savefig=True
    )
    '''
