"""Export sampled spherical likelihood.

"""
import numpy as np
import matplotlib.pyplot as plt

from likelihood_rc import PATHOUT
from view_likelihood import view_samples
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import collate_data_files, harmony
from harmonia.cosmology import fiducial_distance


def safe_save(data, path, name, extension):
    """Safely save data by checking overwrite protections.

    Parameters
    ----------
    data : array_like
        Data to be saved.
    path, name, extension : str
        Path, file name and file extension for the data to be saved.

    Raises
    ------
    AssertionError
        If path does not exist.
    AssertionError
        If overwrite permission is denied at the output path.

    """
    from numpy import save
    from harmonia.collections import (
        confirm_directory_path,
        overwrite_protection
    )

    file = name + extension
    assert confirm_directory_path(path)
    assert overwrite_protection(path, file)
    save(path + file, data)


def filter_data(full_data, remove_degrees=(), renorm_range=()):
    """Filter data.

    Parameters
    ----------
    full_data : dict
        Collated raw data.
    remove_degrees : int, array_like, optional
        If not an empty tuple (default), modes whose spherical degree is an
        element are removed from the data vector and parametrised
        covariance.
    renorm_range : float, array_like, optional
        If not an empty tuple (default), renormalise likelihood in the new
        prior range specified.

    Returns
    -------
    filtered_data : dict
        Filtered data.

    """
    if ZMAX is None and BOXSIZE is not None:
        disc = DiscreteSpectrum(BOXSIZE/2, 'dirichlet', KMAX)
    elif ZMAX is not None and BOXSIZE is None:
        disc = DiscreteSpectrum(fiducial_distance(ZMAX), 'dirichlet', KMAX)

    index_vector = SphericalArray\
        .build(disc=disc)\
        .unfold(PIVOT, return_only='index')

    excluded_deg = np.fromiter(
        map(lambda index: index[0] in remove_degrees, index_vector),
        dtype=bool
    )
    if renorm_range:
        if len(renorm_range) != 2:
            raise ValueError(
                "Renormalisation prior range must be given "
                "exactly two end points. "
            )
        # Firs [0] for returning the first and only axis, and the second
        # [0] for the first and only index.
        try:
            included_range = slice(
                np.argwhere(full_data['parameters'] == renorm_range[0])[0][0],
                np.argwhere(full_data['parameters'] == renorm_range[1])[0][0] + 1
            )
        except IndexError:
            included_range = slice(None, None)
    else:
        included_range = slice(None, None)

    parameter_flat = full_data['parameters'][included_range]
    likelihood_contributions = \
        full_data['likelihood'][:, included_range, ~excluded_deg]

    filtered_data = {
        'parameters': parameter_flat,
        'likelihood': np.sum(likelihood_contributions, axis=-1)
    }

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
    scr_dir_path = f"{PATHOUT}{SCRIPT_NAME}/"
    collate_path = scr_dir_path + "collated/"

    if collate_data:
        search_root = \
            f"map={MAP},prior=[[]{PRIOR}[]],pivot={PIVOT},kmax={KMAX}"
        output, count, name_instance = collate_data_files(
            f"{scr_dir_path}{FILE_ROOT}-*{search_root}*.npy", 'npy'
        )

        output['parameters'] = output['parameters'][0]
        output['likelihood'] = np.concatenate(output['likelihood'])

        if save:
            file_tag = "".join(name_instance.split("(")[-1].split(")")[:-1])
            if FILE_ROOT == SCRIPT_NAME:
                file_tag += f"*{count}"
            safe_save(
                output, collate_path, f"{FILE_ROOT}-({file_tag})", ".npy"
            )

    if load_data:
        program_root = f"map={MAP},prior=[{PRIOR}],pivot={PIVOT},kmax={KMAX}"
        output = np.load(
            f"{collate_path}{FILE_ROOT}-({program_root},{PARAM_TAG}).npy"
        ).item()

    return output


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
    plt.close('all')
    plt.style.use(harmony)

    view_samples(
        data,
        r"$f_\mathrm{NL}$",
        r"$\mathcal{L}(f_\mathrm{NL})$",
        **plot_kwargs
    )
    if savefig:
        program_root = f"map={MAP},prior=[{PRIOR}],pivot={PIVOT},kmax={KMAX}"
        plt.savefig(
            f"{PATHOUT}log_likelihood-"
            f"{FILE_ROOT}-({program_root},{PARAM_TAG}).pdf"
        )


if __name__ == '__main__':

    SCRIPT_NAME = "simulation_likelihood"  # "realisation_likelihood"  #
    FILE_ROOT = "halos-(NG=0.,z=1.)"  # SCRIPT_NAME  #

    ZMAX = None
    BOXSIZE = 1000.

    MAP = "cartesian"
    PRIOR = "2.0,3.0"
    PIVOT = "spectral"
    KMAX = 0.075

    PARAM_TAG = "nbar=2.49e-4,b1=2.38,f0=none"
    # PARAM_TAG = (
    #     "gen=nbodykit,nbar=0.001,b1=2.,f0=none,"
    #     "rmax=293.,xpd=2.,mesh=256,niter=1000"
    # )

    output = read_data(
        collate_data=True,
        load_data=False,
        save=True
    )
#    clean = np.ones(42, dtype=bool)
#    clean[6] = False
#    clean[7] = False
#    output['likelihood'] = output['likelihood'][clean, :]
    processed_data = filter_data(
        output,
        remove_degrees=(),
        renorm_range=()
    )
    view_data(
        processed_data,
        scatter_plot=True,
        scaling='normalised',
        estimate='max',
        truth=0.
    )
