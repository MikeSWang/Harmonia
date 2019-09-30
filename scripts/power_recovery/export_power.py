"""Export recovered power spectrum data and plots.

"""
import numpy as np
from matplotlib import pyplot as plt

from recovery_rc import PATHOUT, aggregate as aggregate_data
from view_power import view_spectrum
from harmonia.collections import (
    collate as collate_data,
    confirm_directory_path as confirm_dir,
)


def main(collate=False, load=False, export=True, aggregate=True, save=True,
         savefig=False):
    """Perform data collation, loading and exporting.

    Parameters
    ----------
    collate, load, export, aggregate, save, savefig : bool, optional
        If `True`, collated, load, export, aggregate or save data or save
        the plotted figure.

    """
    outpath = f"{PATHOUT}{script_name}/"
    assert confirm_dir(outpath)

    if collate:
        output, count, _ = collate_data(f"{outpath}{file_prefix}*.npy", 'npy')
        if save:
            assert confirm_dir(outpath + "collated/")
            np.save(f"{outpath}collated/{file_prefix}{file_tag}.npy", output)
        if aggregate: results = aggregate_data(output)

    if load and (file_tag is not None):
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
