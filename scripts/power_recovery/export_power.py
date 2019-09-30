"""Export recovered power spectra.

"""
import numpy as np
from matplotlib import pyplot as plt

from recovery_rc import PATHOUT, overwrite_protection
from view_power import view_spectrum
from harmonia.collections import (
    collate as collate_data,
    confirm_directory_path as confirm_dir,
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
    dof_k = np.size(output_data['k'], axis=-1) - 1
    dof_P = np.size(output_data['Pln'], axis=0) - 1

    results = {
        'Nk': np.sum(output_data['Nk'], axis=0),
        'k': np.average(output_data['k'], axis=0),
        'Pk': np.average(output_data['Pk'], axis=0),
        'Pshot': np.average(output_data['Pshot']),
        'ln': np.atleast_2d(output_data['ln'])[-1],
        'kln': np.atleast_2d(output_data['kln'])[-1],
        'Pln': np.average(output_data['Pln'], axis=0),
        'dk': np.std(output_data['k'], axis=0, ddof=1) / np.sqrt(dof_k),
        'dPk': np.std(output_data['Pk'], axis=0, ddof=1) / np.sqrt(dof_P),
        'dPln': np.std(output_data['Pln'], axis=0, ddof=1) / np.sqrt(dof_P),
    }
    return results


def main(collate=False, load=False, export=True, aggregate=True, save=True,
         savefig=False):
    """Perform data collation, loading and exporting.

    Parameters
    ----------
    collate, load, export, aggregate, save, savefig : bool, optional
        If `True`, collated, load, export, aggregate or save data, and/or save
        the plotted figure.

    """
    outpath = f"{PATHOUT}{script_name}/"
    assert confirm_dir(outpath)

    if collate:
        output, count, _ = collate_data(f"{outpath}{file_prefix}*.npy", 'npy')
        if save:
            assert overwrite_protection(outpath + "collated/")
            np.save(f"{outpath}collated/{file_prefix}{file_tag}.npy", output)
        if aggregate: results = aggregate_data(output)

    if load:
        output = np.load(
            f"{outpath}collated/{file_prefix}{file_tag}.npy",
        ).item()
        if aggregate: results = aggregate_data(output)

    if export:
        view_spectrum(results, case='error')
        if savefig: plt.savefig(f"{outpath}{file_prefix}{file_tag}.pdf")


if __name__ == '__main__':

    script_name = "nbodysim_power"
    file_prefix = "halos-(NG=0.,z=1.)"
    file_tag = "-(nbar=2.49e-4,bias=2.3415,kmax=0.04,boxsize=1000.,mesh=256)"

    collate = True
    load = False
    savefig = False

    main(collate=collate, load=load, savefig=savefig)
