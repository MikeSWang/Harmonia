"""Recover real-space power spectrum from paired N-body simulations.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import CSVCatalog, FFTPower

from recovery_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    harmony,
)
from harmonia.mapper import SphericalMap

HEADINGS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass']


def read_parameters():
    """Read input parameters.

    """
    global input_file, kmax, boxsize, mesh_cal, niter, prog_id

    try:
        input_file = params.input_file
        kmax = params.kmax
        boxsize = params.boxsize
        mesh_cal = params.mesh_cal
        niter = params.niter
        prog_id = params.prog_id
    except AttributeError as attr_err:
        print(attr_err)


def program_tag():
    """Return program tag.

    Returns
    -------
    str
        Program tag.

    """
    param_tag = "rmax={},kmax={},mesh=c{},".format(
        format_float(boxsize/2, 'intdot'),
        format_float(kmax, 'sci'),
        mesh_cal,
    )
    return "".join(["-(", param_tag, ")-", "[", prog_id, "]"])


def process():
    """Program process.

    Returns
    -------
    output : dict
        Program output.

    """
    print(prog_tag.strip("-"))

    disc = DiscreteSpectrum(boxsize/2, 'Dirichlet', kmax)
    flat_order = np.concatenate(disc.wavenumbers).argsort()

    all_wavenumbers = np.concatenate(disc.wavenumbers)[flat_order]
    all_dbl_indices = np.concatenate(disc.dbl_indices)[flat_order]

    measurements = defaultdict(list)
    for pair_suffix in ["L", "R"]:
        catalogue = CSVCatalog(
            "".join(
                [PATHIN, script_name, "/", input_file, pair_suffix, ".txt"]
            ),
            HEADINGS,
        )
        catalogue.attrs['BoxSize'] = boxsize
        catalogue['Position'] = catalogue['x'][:, None] * [1, 0, 0] \
            + catalogue['y'][:, None] * [0, 1, 0] \
            + catalogue['z'][:, None] * [0, 0, 1]

        mesh = catalogue.to_mesh(
            Nmesh=mesh_cal,
            resampler='tsc',
            compensated=True,
            interlaced=True,
        )
        cartesian_power = FFTPower(mesh, mode='1d', kmax=kmax).power

        spherical_map = SphericalMap(disc, catalogue)
        spherical_power = spherical_map.spherical_power()

        measurements['k'].append([cartesian_power['k']])
        measurements['Nk'].append([cartesian_power['modes']])
        measurements['Pk'].append([cartesian_power['power'].real])
        measurements['Pshot'].append([cartesian_power.attrs['shotnoise']])
        measurements['Pln'].append(
            [np.concatenate(spherical_power)[flat_order]]
        )

    output = {var: np.concatenate(vals) for var, vals in measurements.items()}
    output.update({'ln': [all_dbl_indices], 'kln': [all_wavenumbers]})

    return output


def finalise(save=True, plot=True):
    """Program finalisation.

    Parameters
    ----------
    save : bool, optional
        If `True`, aggregate data over all iterations is saved as a dictionary.
    plot : bool, optional
        If `True`, plot the aggregate data and save as a .pdf file.

    """
    base_path = f"{PATHOUT}{script_name}",
    assert confirm_dir(base_path)

    filename = f"{input_file}{prog_tag}"
    if save:
        np.save("".join([base_path, "/", filename, ".npy"]), output)
    if plot:
        results = {
            'Nk': np.sum(output['Nk'], axis=0),
            'k': np.average(output['k'], axis=0),
            'Pk': np.average(output['Pk'], axis=0),
            'Pshot': np.average(output['Pshot']),
            'ln': output['ln'],
            'kln': output['kln'],
            'Pln': np.average(output['Pln'], axis=0),
        }
        try:
            plt.style.use(harmony)
            plt.close('all')

            plt.errorbar(
                results['k'],
                results['Pk'],
                yerr=results['Pk']/np.sqrt(results['Nk']),
                color='#C40233',
                label='Cartesian',
            )
            plt.loglog(
                results['kln'],
                results['Pln'],
                color='#0087BD',
                label='spherical',
                )

            POLE_NUM = 2  # number of poles
            POLE_COLOUR = ['#000000', '#FFD300',]  # colour of poles
            for ell in range(POLE_NUM):
                idx_sel = (results['ln'][:, 0] == ell)
                plt.scatter(
                    results['kln'][idx_sel],
                    results['Pln'][idx_sel],
                    color=POLE_COLOUR[ell],
                    label=r'$\ell={:d}$'.format(ell),
                )

            plt.xlim(0.99*results['kln'].min(), 1.01*results['kln'].max())
            plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
            plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
            plt.legend()

            plt.savefig("".join([base_path, "/", filename, ".pdf"]))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    read_parameters()
    prog_tag = program_tag()
    output = process()
    finalise()
