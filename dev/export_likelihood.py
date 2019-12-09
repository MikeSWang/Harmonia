"""Export sampled likelihood.

"""
import os
import sys
from pathlib import Path

_cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))

import numpy as np
import matplotlib.pyplot as plt

from view_likelihood import view_samples
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

    Raises
    ------
    AssertionError
        If path does not exist.
    AssertionError
        If overwrite permission is denied at the output path.

    """
    assert confirm_directory_path(path)
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
        full_data['spherical_likelihood'][:, :, ~excluded_deg]

    filtered_data = full_data
    filtered_data.update(
        {
            'spherical_likelihood': np.sum(likelihood_contributions, axis=-1)
        }
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
    program_root = f"map={MAP},kmax={KMAX},{PRIOR},{FIXED}"
    filename = f"{script_name}-{FILE_ROOT}-({program_root})"

    if collate_data:
        search_root = f"map={MAP},kmax={KMAX},*{PRIOR},{FIXED}"\
            .replace("=[", "=[[]").replace("],", "[]],")

        output, _, _ = collate_data_files(
            f"{scr_dir_path}/*{FILE_ROOT}-*-*{search_root}*.npy", 'npy'
        )

        output['parameters'] = np.linspace(-100, 300, 401)
        output[f'{MAP}_likelihood'] = np.squeeze(output[f'spherical_likelihood'])

        if save:
            safe_save(output, collate_path, filename + ".npy")

    if load_data:
        output = np.load(str(collate_path/filename) + ".npy").item()

    return output, filename


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
    program_root = f"map={MAP},kmax={KMAX},{PRIOR},{FIXED}"
    filename = f"{script_name}-{FILE_ROOT}-({program_root}).pdf"

    plt.close('all')
    plt.style.use(harmony)

    visual_data = data
    visual_data['likelihood'] = data[f'{MAP}_likelihood']

    view_samples(
        visual_data,
        r"$f_\mathrm{NL}$", # r"$b_1$", #
        r"$\mathcal{L}(f_\mathrm{NL})$", # r"$\mathcal{L}(b_1)$", #
        **plot_kwargs
    )

    if savefig:
        plt.savefig(PATHOUT/script_name/filename)


if __name__ == '__main__':

    PATHOUT = Path("./data/output/")
    FILE_ROOT = "(NG=-100.,z=1.)"

    BOXSIZE = 1000.

    MAP = "spherical"
    KHYB = 0.075
    KMAX = 0.075
    PIVOT = "spectral"
    PRIOR = "fnl_prior=[-300.0,100.0]"
    FIXED = "bias=2.36"

    script_name = f"{MAP}_likelihood"

    output, filename = read_data(
        collate_data=True,
        load_data=False,
        save=True
    )
    filtered_output = filter_data(output, remove_degrees=())
    view_data(
        filtered_output,
        precision=0,
        norm_range=(),
        scatter_plot=True,
        savefig=True
    )
