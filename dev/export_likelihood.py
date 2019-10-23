"""Export sampled spherical likelihood over many data realisations.

"""
import numpy as np
import matplotlib.pyplot as plt

from likelihood_rc import PATHOUT
from view_likelihood import view_chi_square
from harmonia.collections import (
    collate_data_files,
    confirm_directory_path as confirm_dir,
    harmony,
    overwrite_protection,
)


def aggregate_data(output):
    """Aggregate data into a single result.

    Parameters
    ----------
    output : dict
        Output data to be aggregated.

    Returns
    -------
    result : dict
        Aggregated output data.

    """
    result = {var: np.average(vals, axis=0) for var, vals in output.items()}

    return result


def safe_save(data, path, name):
    """Safely save data by checking overwrite path and protections.

    Raises
    ------
    AssertionError
        If path does not exist.
    AssertionError
        If overwrite permission is denied at the output path.

    """
    assert confirm_dir(path)
    assert overwrite_protection(path, name)
    np.save(path + name, data)


def process_data(collate_data=False, load_data=False, save=False):
    """Collate, load and export likelihood outputs.

    Parameters
    ----------
    collate_data, load_data, save : bool, optional
        If `True` (default is `False`), collate, load and/or save
        likelihood data.

    """
    data_outpath = f"{PATHOUT}{SCRIPT_NAME}/"
    collate_path = data_outpath + "collated/"

    global output, data

    if collate_data:
        output, count, _ = collate_data_files(
            f"{data_outpath}{FILE_ROOT}-*).npy",
            'npy'
        )
        data = aggregate_data(output)

        if save:
            save_str = "".join(_.split("(")[-1].split(")")[:-1]) + f"*{count}"
            safe_save(output, collate_path, f"{FILE_ROOT}-({save_str}).npy")

    if load_data:
        output = np.load(
            f"{collate_path}{FILE_ROOT}-({GEN_TAG},{PARAM_TAG}).npy"
        ).item()
        data = aggregate_data(output)


def view_data(savefig=False):
    """Visualise output data.

    Parameters
    ----------
    savefig : bool, optional
        If `True` (default is `False`), save plotted likelihood figure.

    """
    plt.close('all')
    plt.style.use(harmony)
    view_chi_square(data)
    if savefig:
        plt.savefig(f"{PATHOUT}likelihood-({GEN_TAG},{PARAM_TAG})")


if __name__ == '__main__':

    SCRIPT_NAME = "simulation_likelihood"
    FILE_ROOT = "halos-(NG=0.,z=1.)"  # SCRIPT_NAME

    PRIOR = "[-100,100]"
    GENERATOR = "nbodykit"
    PIVOT = "spectral"

    GEN_TAG = f"prior={PRIOR},pivot={PIVOT}" # f"gen={GENERATOR},pivot={PIVOT}"
    PARAM_TAG = \
        "(prior=[-100,100],pivot=spectral,kmax=0.04)"
    #     "nbar=0.001,b1=2.,f0=none,rmax=293.,kmax=0.04,xpd=2.,mesh=gc256,iter=1000"

    process_data(collate_data=True, load_data=False, save=True)
    view_data()
