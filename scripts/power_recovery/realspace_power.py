"""Recover real-space power spectrum from fixed-epoch log-normal catalogues.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower, ConvolvedFFTPower, FKPCatalog

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
from harmonia.mapper import NBKCatalogue, RandomCatalogue, SphericalMap


def broadcast_parameters():
    """Broadcast input parameters.

    """
    global nbar, contrast, bias, redshift, zmax, rmax, kmax, dk, expand, \
        mesh_gen, mesh_cal, niter, prog_id

    try:
        nbar = params.nbar
        contrast = params.contrast
        bias = params.bias
        redshift = params.redshift
        zmax = params.zmax
        rmax = params.rmax
        kmax = params.kmax
        dk = params.dk
        expand = params.expand
        mesh_gen = params.mesh_gen
        mesh_cal = params.mesh_cal
        niter = params.niter
        prog_id = params.prog_id
    except AttributeError as attr_err:
        print(attr_err)


def setup_cosmology():
    """Set up cosmological variables.

    """
    global cosmo, Plin, rmax

    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')
    rmax = cosmo.comoving_distance(zmax)


def program_tag():
    """Return program tag.

    """
    global case_is_mock

    try:
        ratio_tag = format_float(float(contrast), 'decdot')
        case_is_mock = True
    except (TypeError, ValueError):
        ratio_tag = "simu"
        case_is_mock = False

    if mesh_gen == mesh_cal:
        mesh_tag = f"mesh=gc{mesh_gen},"
    else:
        mesh_tag = f"mesh=[g{mesh_gen},c{mesh_cal}],"

    param_tag = "nbar={},contrast={},rmax={},kmax={},xpd={},".format(
        format_float(nbar, 'sci'),
        ratio_tag,
        format_float(rmax, 'intdot'),
        format_float(kmax, 'sci'),
        format_float(expand, 'decdot'),
    )
    iter_tag = "iter={}".format(niter)

    part_tags = ["-(", param_tag, mesh_tag, iter_tag, ")-", "[", prog_id, "]"]

    return "".join(part_tags)


def process():
    """Program process.

    Returns
    -------
    output : dict
        Program output.

    """
    print(prog_tag.strip("-"))

    boxsize = 2 * expand * rmax
    to_mesh_params = dict(Nmesh=mesh_cal, resampler='tsc', compensated=True)


    disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
    flat_order = np.concatenate(disc.wavenumbers).argsort()

    all_wavenumbers = np.concatenate(disc.wavenumbers)[flat_order]
    all_dbl_indices = np.concatenate(disc.dbl_indices)[flat_order]

    measurements = defaultdict(list)
    for run in range(niter):
        data_catalogue = NBKCatalogue(
            Plin,
            nbar,
            bias=bias,
            BoxSize=boxsize,
            Nmesh=mesh_gen,
        )
        if case_is_mock:
            rand_catalogue = RandomCatalogue(contrast*nbar, boxsize)
            data_catalogue['NZ'] = nbar * data_catalogue['Weight']
            rand_catalogue['NZ'] = nbar * rand_catalogue['Weight']
            pair_catalogue = FKPCatalog(data_catalogue, rand_catalogue)
        else:
            rand_catalogue = None

        if case_is_mock:
            mesh = pair_catalogue.to_mesh(**to_mesh_params)
            cartesian_power = ConvolvedFFTPower(
                mesh,
                poles=[0],
                dk=dk,
                kmax=all_wavenumbers.max()+dk,
            ).poles
        else:
            mesh = data_catalogue.to_mesh(**to_mesh_params)
            cartesian_power = FFTPower(
                mesh,
                mode='1d',
                dk=dk,
                kmax=all_wavenumbers.max()+dk,
            ).power

        spherical_map = SphericalMap(
            disc,
            data_catalogue,
            rand=rand_catalogue,
            mean_density_data=nbar,
            mean_density_rand=contrast*nbar,
        )
        spherical_power = spherical_map.spherical_power()

        measurements['k'].append([cartesian_power['k']])
        measurements['Nk'].append([cartesian_power['modes']])
        measurements['Pshot'].append([cartesian_power.attrs['shotnoise']])
        measurements['Pln'].append(
            [np.concatenate(spherical_power)[flat_order]]
        )
        if case_is_mock:
            measurements['Pk'].append([cartesian_power['power_0'].real])
        else:
            measurements['Pk'].append([cartesian_power['power'].real])

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
