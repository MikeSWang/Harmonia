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


def initialise():
    """Initialise from input parameters and return runtime information.

    Returns
    -------
    runtime_info : str
        Runtime information.

    Raises
    ------
    AttributeError
        If a required input arameter is missing.

    """
    global input_file, kmax, boxsize, mesh_cal, niter, prog_id

    try:
        input_file = params.input_file
        kmax = params.kmax
        boxsize = params.boxsize
        mesh_cal = params.mesh_cal
        niter = params.niter
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    runtime_info = "-(rmax={},kmax={},mesh=c{})".format(
        format_float(boxsize/2, 'intdot'),
        format_float(kmax, 'sci'),
        mesh_cal,
    )
    return runtime_info


def process(runtime_info):
    """Program process.

    Parameters
    ----------
    runtime_info : str
        Program runtime information.

    Returns
    -------
    output_data : dict
        Program output.

    """
    print(runtime_info.strip("-"))

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

    output_data = {
        var: np.concatenate(vals)
        for var, vals in measurements.items()
    }
    output_data.update({'ln': [all_dbl_indices], 'kln': [all_wavenumbers]})
    return output_data


def finalise(output_data, save=True, plot=True):
    """Program finalisation with optional data and figure saving.

    Parameters
    ----------
    output_data : dict
        Program output.
    save : bool, optional
        If `True`, aggregate data over all iterations is saved as a dictionary.
    plot : bool, optional
        If `True`, plot the aggregate data and save as a .pdf file.

    Raises
    ------
    AssertionError
        If the output path does not exist.
    Exception
        If plotting fails.

    """
    base_path = f"{PATHOUT}{script_name}"
    assert confirm_dir(base_path)

    filename = f"{input_file}{program_tag}"
    if save:
        np.save("".join([base_path, "/", filename, ".npy"]), output_data)
    if plot:
        results = {
            'Nk': np.sum(output_data['Nk'], axis=0),
            'k': np.average(output_data['k'], axis=0),
            'Pk': np.average(output_data['Pk'], axis=0),
            'Pshot': np.average(output_data['Pshot']),
            'ln': output_data['ln'][-1],
            'kln': output_data['kln'][-1],
            'Pln': np.average(output_data['Pln'], axis=0),
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
            raise Exception(e)


if __name__ == '__main__':

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
