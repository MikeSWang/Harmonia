"""Runtime configuration for power spectrum recovery in spherical basis.

"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def import_local_package():
    """Add package to Python module path.

    """
    _cwd = os.path.dirname(__file__)
    sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../../")))


def parse_cli_args():
    """Parse command line arguments.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameters.

    """
    cli_parser = ArgumentParser()

    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--contrast', type=float, default=None)

    cli_parser.add_argument('--bias', type=float, default=2.)
    cli_parser.add_argument('--redshift', type=float, default=0.)

    cli_parser.add_argument('--zmax', type=float, default=0.05)
    cli_parser.add_argument('--rmax', type=float, default=150.)

    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--dk', type=float, default=1e-2)

    cli_parser.add_argument('--boxsize', type=float, default=1000.)
    cli_parser.add_argument('--expand', type=float, default=2.)

    cli_parser.add_argument('--mesh-gen', type=int, default=256)
    cli_parser.add_argument('--mesh-cal', type=int, default=256)

    cli_parser.add_argument('--niter', type=int, default=10)
    cli_parser.add_argument('--prog-id', default="")
    cli_parser.add_argument('--input-file', default="halos-(NG=0.,z=1.)-0")

    return cli_parser.parse_args()


def overwrite_protection(outpath, outname, save=True):
    """Inspect and modify overwrite protection.

    Parameters
    ----------
    outpath : str
        Write-out directory path.
    outname : str
        Write-out filename.

    Returns
    -------
    overwrite_permission : bool
        Overwrite permission.

    """
    overwrite_permission = False
    while save:
        try:
            if not os.path.exists(outpath):
                raise FileNotFoundError(f"{outpath} does not exist. ")
            if not overwrite_permission:
                if os.path.exists(outpath + outname):
                    raise FileExistsError
            overwrite_permission = True
            break
        except FileExistsError:
            grant_permission = input(
                "Saving would overwrite existing file at destination. "
                "Do you want to continue? [y/n] "
            )
            if grant_permission.lower().startswith('y'):
                overwrite_permission = True
                break
            else:
                overwrite_permission = False
                raise FileExistsError(
                    "Overwrite permission denied. File not saved. "
                    )

    return overwrite_permission


def _view(output):

    dof_k = np.size(output['k'], axis=0) - 1
    dof_P= np.size(output['Pln'], axis=0) - 1

    results = {
        'Nk': np.sum(output['Nk'], axis=0),
        'k': np.average(output['k'], axis=0),
        'Pk': np.average(output['Pk'], axis=0),
        'Pshot': np.average(output['Pshot']),
        'ln': output['ln'][-1],
        'kln': output['kln'][-1],
        'Pln': np.average(output['Pln'], axis=0),
        'dk': np.std(output['k'], axis=0, ddof=1) / np.sqrt(dof_k),
        'dPk': np.std(output['Pk'], axis=0, ddof=1) / np.sqrt(dof_P),
        'dPln': np.std(output['Pln'], axis=0, ddof=1) / np.sqrt(dof_P),
    }

    plt.errorbar(
        results['k'],
        results['Pk'],
        xerr=results['dk'],
        yerr=results['dPk'],
        color='#0087BD',
        label='Cartesian',
    )

    spherical_result = plt.loglog(
        results['kln'],
        results['Pln'],
        color='#C40233',
        label='spherical',
    )

    for layer in [1, 2]:
        plt.fill_between(
            results['kln'],
            results['Pln'] - layer * results['dPln'],
            results['Pln'] + layer * results['dPln'],
            color=spherical_result[0].get_color(),
            alpha=(1/4)**layer,
        )

    for idx, dbl_indices in enumerate(results['ln']):
        if dbl_indices[0] == 0:
            plt.annotate(
                str(dbl_indices),
                xy=(results['kln'][idx], results['Pln'][idx]),
                verticalalignment='bottom',
                fontsize=6,
            )

    plt.xlim(left=0.99*results['kln'].min(), right=1.01*results['kln'].max())
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()


if not __name__ == '__main__':

    import_local_package()

    from harmonia.collections import get_filename

    script_name = get_filename(sys.argv[0])
    params = parse_cli_args()
