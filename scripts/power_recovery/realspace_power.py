"""Recover real-space power spectrum from ``nbodykit`` log-normal
catalogues.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower, ConvolvedFFTPower, FKPCatalog

from recovery_rc import PATHOUT, params, _view, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    harmony,
    sort_dict_to_list,
)
from harmonia.mapper import (
    LogNormalCatalogue,
    NBKCatalogue,
    RandomCatalogue,
    SphericalMap,
)

GEN_CATALOGUE = {
    'lognormal': LogNormalCatalogue,
    'nbodykit': NBKCatalogue,
}


def initialise():
    """Initialise from input parameters, set up cosmology and return
    runtime information.

    Returns
    -------
    runtime_info : str
        Runtime information.

    Raises
    ------
    AttributeError
        If a required input arameter is missing.

    """
    global nbar, contrast, bias, redshift, zmax, kmax, dk, expand, \
        mesh_gen, mesh_cal, niter, prog_id

    try:
        generator = params.generator
        nbar = params.nbar
        contrast = params.contrast
        bias = params.bias
        redshift = params.redshift
        zmax = params.zmax
        kmax = params.kmax
        dk = params.dk
        expand = params.expand
        mesh_gen = params.mesh_gen
        mesh_cal = params.mesh_cal
        niter = params.niter
        prog_id = params.prog_id
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global cosmo, Plin, rmax, gen_name

    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')
    rmax = cosmo.comoving_distance(zmax)

    if generator.lower().startswith('g'):
        gen_name = "gaussian"
    elif generator.lower().startswith('l'):
        gen_name = "lognormal"
    elif generator.lower().startswith('n'):
        gen_name = "nbodykit"
    gen_tag = f"gen={gen_name},"

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

    part_tags = ["-(", gen_tag, param_tag, mesh_tag, iter_tag, ")-", "[", \
         prog_id, "]"]
    runtime_info = "".join(part_tags)
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

    boxsize = 2 * expand * rmax
    to_mesh_params = dict(Nmesh=mesh_cal, resampler='tsc', compensated=True)

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
        data_catalogue = GEN_CATALOGUE[gen_name](
            Plin,
            nbar,
            bias=bias,
            boxsize=boxsize,
            num_mesh=mesh_gen
        )
        if case_is_mock:
            rand_catalogue = RandomCatalogue(contrast*nbar, boxsize)
            data_catalogue['NZ'] = nbar * data_catalogue['Weight']
            rand_catalogue['NZ'] = nbar * rand_catalogue['Weight']
            pair_catalogue = FKPCatalog(data_catalogue, rand_catalogue)
            contrast_nbar = contrast * nbar
        else:
            rand_catalogue = None
            contrast_nbar = None

        if case_is_mock:
            mesh = pair_catalogue.to_mesh(**to_mesh_params)
            cartesian_power = ConvolvedFFTPower(
                mesh,
                poles=[0],
                dk=dk,
                kmax=kmax
            ).poles
        else:
            mesh = data_catalogue.to_mesh(**to_mesh_params)
            cartesian_power = FFTPower(
                mesh,
                mode='1d',
                dk=dk,
                kmax=kmax
            ).power

        spherical_map = SphericalMap(
            disc,
            data_catalogue,
            rand=rand_catalogue,
            mean_density_data=nbar,
            mean_density_rand=contrast_nbar
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
