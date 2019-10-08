"""Export recovered power spectra.

"""
import numpy as np
from matplotlib import pyplot as plt

from recovery_rc import PATHOUT
from view_power import view_spectrum
from harmonia.collections import (
    collate as collate_data,
    confirm_directory_path as confirm_dir,
    harmony,
    overwrite_protection,
)


def aggregate_data(output_data):
    """Aggregate output data.

    Parameters
    ----------
    output_data : dict
        Output data to be aggregated.

    Returns
    -------
    results : dict
        Aggregated results.

    """
    dof = np.size(output_data['Pln'], axis=0) - 1
    results = {
        'Nk': np.sum(output_data['Nk'], axis=0),
        'k': np.average(output_data['k'], axis=0),
        'Pk': np.average(output_data['Pk'], axis=0),
        'Pshot': np.average(output_data['Pshot']),
        'ln': np.atleast_2d(output_data['ln'])[-1],
        'kln': np.atleast_2d(output_data['kln'])[-1],
        'Pln': np.average(output_data['Pln'], axis=0),
        'dk': np.std(output_data['k'], axis=0, ddof=1) / np.sqrt(dof),
        'dPk': np.std(output_data['Pk'], axis=0, ddof=1) / np.sqrt(dof),
        'dPln': np.std(output_data['Pln'], axis=0, ddof=1) / np.sqrt(dof),
    }

    # HACK: For spurious ``nbodykit`` binning only.
    _SELECT_IDX = slice(2, None)  # slice(None, None)
    for key in results:
        if key.endswith('k'):
            results[key] = results[key][_SELECT_IDX]

    return results


def main(collate=False, load=False, export=True, aggregate=True, save=True,
         savefig=False):
    """Perform data collation, loading and exporting.

    Parameters
    ----------
    collate, load, export, aggregate, save, savefig : bool, optional
        If `True`, collate, load, export, aggregate or save data, and/or
        save the plotted figure.  Default is `True` for exporting,
        aggregating and saving data, and `False` for collating and loading
        data or saving figure.

    Raises
    ------
    AssertionError
        If output path does not exist.
    AssertionError
        If overwrite permission is denied at the output path.

    """
    outpath = f"{PATHOUT}{SCRIPT_NAME}/"
    collate_path = f"{outpath}collated/"
    assert confirm_dir(outpath)

    global output

    if collate:
        output, count, _ = collate_data(f"{outpath}{FILE_PREFIX}*.npy", 'npy')
        if save:
            assert confirm_dir(collate_path)
            assert overwrite_protection(
                f"{outpath}collated/",
                f"{FILE_PREFIX}{FILE_TAG}.npy"
            )
            np.save(f"{outpath}collated/{FILE_PREFIX}{FILE_TAG}.npy", output)
        if aggregate:
            results = aggregate_data(output)

    if load:
        output = np.load(
            f"{outpath}collated/{FILE_PREFIX}{FILE_TAG}.npy"
        ).item()
        if aggregate: results = aggregate_data(output)

    if export:
        plt.style.use(harmony)
        plt.close('all')
        view_spectrum(results, case='error')
        if savefig: plt.savefig(f"{outpath}{FILE_PREFIX}{FILE_TAG}.pdf")


if __name__ == '__main__':

    SCRIPT_NAME = "realspace_power"
    FILE_PREFIX = "realspace_power"
    FILE_TAG = \
        "-(nbar=0.001,rmax=148.,kmax=0.1,xpd=2.,mesh=gc256,iter=1000)"

    COLLATE = True
    LOAD = False
    SAVEFIG = False

    main(collate=COLLATE, load=LOAD, savefig=SAVEFIG)
