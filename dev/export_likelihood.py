"""Export sampled spherical likelihood over many data realisations.

"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from likelihood_rc import PATHOUT
from view_likelihood import view_likelihood
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


def process_data(collate_data=False, load_data=False, save=True):
    """Collate, load and export likelihood outputs.

    Parameters
    ----------
    collate_data, load_data, save : bool, optional
        If `True` , collate, load and/or save likelihood data.

    """
    data_outpath = f"{PATHOUT}{SCRIPT_NAME}/"
    collate_path = data_outpath + "collated/"

    global output, data

    if collate_data:
        output, count, _ = \
            collate(f"{data_outpath}{SCRIPT_NAME}-*{GEN_TAG}*.npy", 'npy')
        data = aggregate_data(output)

        if save:
            save_str = "".join(_.split("(")[-1].split(")")[:-1]) + f"*{count}"
            safe_save(data, collate_path, f"{SCRIPT_NAME}-({save_str}).npy")

    if load_data:
        data = np.load(
            f"{collate_path}{SCRIPT_NAME}-({GEN_TAG},{PARAM_TAG}).npy"
        ).item()


def view_data(savefig=False):
    """Visualise output data.

    Parameters
    ----------
    savefig : bool, optional
        If `True` (default is `False`), save plotted likelihood figure.

    """
    plt.close('all')
    plt.style.use(harmony)
    view_likelihood(data)
    if savefig:
        plt.savefig(f"{PATHOUT}likelihood-({GEN_TAG},{PARAM_TAG})")


if __name__ == '__main__':

    SCRIPT_NAME = "build_likelihood"
    GENERATOR = "nbodykit"
    PIVOT = "k"

    GEN_TAG = f"gen={GENERATOR},pivot={PIVOT}"
    PARAM_TAG = \
        "nbar=0.001,b1=2.,f0=none,rmax=148.,kmax=0.1,xpd=2.,mesh=gc256,iter=250"

    process_data(collate_data=True, load_data=False)
    view_data()
