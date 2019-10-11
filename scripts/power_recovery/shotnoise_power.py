"""Recover constant shot noise power spectrum from unclustered catalogues.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import FFTPower

from recovery_rc import PATHOUT, params, _view, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    harmony,
    sort_dict_to_list,
)
from harmonia.mapper import RandomCatalogue, SphericalMap


def initialise():
    """Initialise from input parameters and return runtime information.

    Returns
    -------
    runtime_info : str
        Runtime information.

    Raises
    ------
    AttributeError
        If a required input parameter is missing.

    """
    global nbar, rmax, kmax, dk, mesh_cal, niter, prog_id

    try:
        nbar = params.nbar
        rmax = params.rmax
        dk = params.dk
        kmax = params.kmax
        mesh_cal = params.mesh_cal
        niter = params.niter
        prog_id = params.prog_id
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    param_tag = "nbar={},rmax={},kmax={},mesh=c{},iter={}".format(
        format_float(nbar, 'sci'),
        format_float(rmax, 'intdot'),
        format_float(kmax, 'sci'),
        mesh_cal,
        niter,
    )
    runtime_info = "".join(["-(", param_tag, ")-", "[", prog_id, "]"])
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

    disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
    flat_order = np.concatenate(sort_dict_to_list(disc.wavenumbers)).argsort()

    all_wavenumbers = np.concatenate(
        sort_dict_to_list(disc.wavenumbers)
    )[flat_order]
    all_root_indices = np.concatenate(
        sort_dict_to_list(disc.root_indices)
    )[flat_order]

    measurements = defaultdict(list)
    for run in range(niter):
        data_catalogue = RandomCatalogue(nbar, 2*rmax)
        mesh = data_catalogue.to_mesh(
            Nmesh=mesh_cal,
            resampler='tsc',
            compensated=True
        )

        cartesian_power = FFTPower(mesh, mode='1d', dk=dk, kmax=kmax).power
        spherical_map = SphericalMap(
            disc,
            data_catalogue,
            mean_density_data=nbar
        )
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
    output_data.update({'ln': [all_root_indices], 'kln': [all_wavenumbers]})
    return output_data


def finalise(output_data, save=True, plot=True):
    """Program finalisation with optional data and figure saving.

    Parameters
    ----------
    output_data : dict
        Program output.
    save : bool, optional
        If `True`, aggregate data is saved as :obj:`dict`.
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

    filename = f"{script_name}{program_tag}"
    if save:
        np.save("".join([base_path, "/", filename, ".npy"]), output_data)
    if plot:
        try:
            plt.style.use(harmony)
            plt.close('all')
            _view(output_data)
            plt.savefig("".join([base_path, "/", filename, ".pdf"]))
        except Exception as e:
            print(e)


if __name__ == '__main__':

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
