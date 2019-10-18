"""Export catalogue summary data.

"""
import numpy as np
from matplotlib import pyplot as plt
from nbodykit import cosmology

from fidelity_rc import PATHOUT
from harmonia.collections import (
    collate_data_files as collate_data,
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
    dof = np.size(output_data['Pk'], axis=0) - 1
    results = {
        'Nk': np.sum(output_data['Nk'], axis=0),
        'k': np.average(output_data['k'], axis=0),
        'Pk': np.average(output_data['Pk'], axis=0),
        'Pshot': np.average(output_data['Pshot']),
        'dk': np.std(output_data['Pk'], axis=0, ddof=1) / np.sqrt(dof),
        'dPk': np.std(output_data['Pk'], axis=0, ddof=1) / np.sqrt(dof),
    }
    return results


def export_data(collate=False, load=False, save=False):
    """Perform data collation, loading and exporting.

    Parameters
    ----------
    collate, load, save : bool, optional
        If `True`, collate, load or save data (default is `False`).

    Returns
    -------
    results : dict
        Exported results.

    Raises
    ------
    InterruptedError
        If parameters and parameter tag mismatch and the user choose not
        continue.

    """
    mismatch = not all(
        [
            f"nbar={NBAR}" in param_tag,
            f"bias={BIAS}".rstrip("0") in param_tag,
            f"mesh={NUM_MESH}" in param_tag.replace("g", "").replace("c", ""),
        ]
    )
    if mismatch:
        continuation = input("Parameter mismatch... continue? [y/n] ")
        if not continuation.lower().startswith('y'):
            raise InterruptedError("User termination. ")

    global filename_root, outpath, outname

    filename_root = "-".join([obj_name, gen_name])
    outpath = "".join([PATHOUT, filename_root, "/"])
    outname = "".join([filename_root, "-(", param_tag, ")"])
    collate_path = f"{outpath}collated/"

    if collate:
        output, count, _ = \
            collate_data(f"{outpath}*{NBAR}*{NUM_MESH}*.npy", 'npy')
        if save:
            assert confirm_dir(collate_path)
            assert overwrite_protection(f"{collate_path}", f"{outname}.npy")
            np.save(f"{outpath}collated/{outname}.npy", output)
        results = aggregate_data(output)

    if load:
        output = np.load(f"{outpath}collated/{outname}.npy").item()
        results = aggregate_data(output)

    return results


def view(results, savefig=False):
    """View export results.

    Parameters
    ----------
    results : dict
        Exported results.
    savefig : bool, optional
        If `True` (default is `False`), save the plotted figure.

    """
    REDSHIFT = 0.
    YLIM = (22000, 120000)

    global k, Nk, Pk, dPk, Pk_model

    if np.isclose(results['k'][0], 0.):
        sel = slice(1, None)
    else:
        sel = slice(None)
    k = results['k'][sel]
    Nk = results['Nk'][sel]
    Pk = results['Pk'][sel]
    dPk = results['dPk'][sel]

    Plin = cosmology.LinearPower(
        cosmology.Planck15,
        redshift=REDSHIFT,
        transfer='CLASS'
    )
    Pshot = 1 / NBAR
    Pk_model = BIAS**2 * Plin(k) + Pshot
    dPk_model = Pk_model / np.sqrt(Nk - 1)

    global deviation

    deviation = Pk / Pk_model - 1
    if np.abs(np.average(deviation)) > 0.05:
        shot_noise_check = input(
            "Did you forget to add/subtract shot noise? [y/n] "
        )
        if shot_noise_check.lower().startswith('y'):
            shot_noise_action = input(
                "Do you want to add/subtract shot noise? [a/s] "
            )
            if shot_noise_action.lower().startswith('a'):
                Pk_model += Pshot
            elif shot_noise_action.lower().startswith('s'):
                Pk_model -= Pshot
            deviation = Pk / Pk_model - 1

    xlim = (0.95*min(k), 1.05*max(k))

    plt.style.use(harmony)
    plt.close('all')
    plt.figure("Catalogue fidelity")

    main_ax = plt.subplot2grid((5,6), (0,0), rowspan=4, colspan=6)

    plt.errorbar(k, Pk_model, dPk_model, ls='--', label="power spectrum input")
    main = plt.errorbar(k, Pk, dPk, label=f"{gen_name} {obj_name} output")

    plt.tick_params(axis='x', which='both', labelbottom=False)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(YLIM)
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()

    plt.subplot2grid((5,6), (4,0), colspan=6, sharex=main_ax)

    plt.plot(k, deviation, color=main[0].get_color(), ls='--')
    plt.fill_between(
        k,
        (Pk + dPk)/Pk_model - 1,
        (Pk - dPk)/Pk_model - 1,
        color=main[0].get_color(),
        alpha=1/5
    )
    plt.fill_between(
        xlim,
        [ERROR_PATCH_HEIGHT]*2,
        [-ERROR_PATCH_HEIGHT]*2,
        alpha=0.2
    )
    plt.axhline(y=0., lw=1., ls='--')

    plt.xlim(xlim)
    plt.ylim(-ERROR_PANEL_CEILING, ERROR_PANEL_CEILING)
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$\hat{P} \big/ P_\mathrm{model}(k) - 1$')

    plt.subplots_adjust(hspace=0)
    if savefig:
        plt.savefig(f"{outpath}{outname}.pdf")


if __name__ == '__main__':

    NBAR = 0.001
    BIAS = 2.
    NUM_MESH = 256

    obj_name = "catalogue"
    gen_name = "lognormal"
    param_tag = \
        "nbar=0.001,bias=2.,boxsize=593.,kmax=0.1,mesh=gc256,iter=1000"

    COLLATE = False
    LOAD = True

    results = export_data(collate=COLLATE, load=LOAD, save=True)

    ERROR_PANEL_CEILING = 0.05
    ERROR_PATCH_HEIGHT = 0.01

    view(results, savefig=False)
