"""Recover constant shot noise power spectrum from unclustered catalogues.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import FFTPower

from recovery_rc import (
    PATHOUT,
    filename_root,
    params,
    quick_plot,
)
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    harmony,
)
from harmonia.mapper import RandomCatalogue, SphericalMap


def broadcast_parameters():
    """Broadcast input parameters.

    """
    global nbar, rmax, kmax, mesh_cal, niter, prog_id

    try:
        nbar = params.nbar
        rmax = params.rmax
        kmax = params.kmax
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
    param_tag = "nbar={},rmax={},kmax={},mesh=c{},iter={}".format(
        format_float(nbar, 'sci'),
        format_float(rmax, 'intdot'),
        format_float(kmax, 'sci'),
        mesh_cal,
        niter,
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

    disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
    flat_order = np.concatenate(disc.wavenumbers).argsort()

    all_wavenumbers = np.concatenate(disc.wavenumbers)[flat_order]
    all_dbl_indices = np.concatenate(disc.dbl_indices)[flat_order]

    measurements = defaultdict(list)
    for run in range(niter):
        data_catalogue = RandomCatalogue(nbar, 2*rmax)
        mesh = data_catalogue.to_mesh(
            Nmesh=mesh_cal,
            resampler='tsc',
            compensated=True,
        )

        cartesian_power = FFTPower(mesh, mode='1d', kmax=kmax).power
        spherical_map = SphericalMap(
            disc,
            data_catalogue,
            mean_density_data=nbar,
        )
        spherical_power = spherical_map.spherical_power()

        measurements['k'].append([cartesian_power['k']])
        measurements['Nk'].append([cartesian_power['modes']])
        measurements['Pk'].append([cartesian_power['power'].real])
        measurements['Pshot'].append([cartesian_power.attrs['shotnoise']])
        measurements['Pln'].append(
            [np.concatenate(spherical_power)[flat_order]],
        )

    output = {var: np.concatenate(vals) for var, vals in measurements.items()}
    output.update({'ln': [all_dbl_indices], 'kln': [all_wavenumbers]})

    return output


def finalise(save=True, plot=True):
    """Program finalisation.

    """
    base_path = f"{PATHOUT}{filename_root}",
    assert confirm_dir(base_path)

    filename = f"{filename_root}{prog_tag}"
    if save:
        np.save("".join([base_path, "/", filename, ".npy"]), output)
    if plot:
        try:
            plt.style.use(harmony)
            plt.close('all')
            quick_plot(output)
            plt.savefig("".join([base_path, "/", filename, ".pdf"]))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    broadcast_parameters()
    prog_tag = program_tag()
    output = process()
    finalise()
