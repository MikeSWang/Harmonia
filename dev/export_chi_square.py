"""Export bias chi-square sampled from simulations..

"""
import numpy as np
import matplotlib.pyplot as plt

from likelihood_rc import PATHOUT
from harmonia.collections import (
    collate_data_files,
    confirm_directory_path,
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
    pass


def safe_save(data, path, name):
    """Safely save data by checking overwrite path and protections.

    Raises
    ------
    AssertionError
        If path does not exist.
    AssertionError
        If overwrite permission is denied at the output path.

    """
    assert confirm_directory_path(path)
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
            f"{data_outpath}{FILE_ROOT}-*.npy",
            'npy'
        )
        data = aggregate_data(output)

        if save:
            save_str = "".join(_.split("(")[-1].split(")")[:-1])
            safe_save(output, collate_path, f"{FILE_ROOT}-({save_str}).npy")

    if load_data:
        output = np.load(
            f"{collate_path}{FILE_ROOT}-({LOAD_TAG}).npy"
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


if __name__ == '__main__':

    SCRIPT_NAME = "simulation_bias_fitting"
    FILE_ROOT = "halos-(NG=0.,z=1.)"
    LOAD_TAG = ""

    process_data(collate_data=True, load_data=False, save=True)
    view_data()
