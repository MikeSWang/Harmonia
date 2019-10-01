"""Create an ensemble of catalogues.

"""
import warnings
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower, LogNormalCatalog

from fidelity_rc import PATHOUT, params
from harmonia.algorithms.fields import (
    generate_gaussian_random_field,
    generate_lognormal_random_field,
    poisson_sample,
    _cal_isotropic_power_spectrum as cal_power,
)
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    harmony,
)
from harmonia.mapper import GaussianCatalogue, LogNormalCatalogue

MECHANISM = {
    'gaussian': generate_gaussian_random_field,
    'lognormal': generate_lognormal_random_field,
}
CATALOGUE = {
    'gaussian': GaussianCatalogue,
    'lognormal': LogNormalCatalogue,
    'nbodykit': LogNormalCatalog,
}


def initialise():
    """Initialise from input parameters, set up cosmology and return runtime
    information.

    Returns
    -------
    runtime_info : str
        Runtime information.

    """
    global generator, cataloguing, sampling, nbar, bias, redshift, kmax, dk, \
        nbins, boxsize, mesh_gen, mesh_cal, niter, prog_id

    try:
        generator = params.generator
        cataloguing = not params.non_catalogue
        sampling = not params.no_sampling

        nbar = params.nbar
        bias = params.bias
        redshift = params.redshift

        kmax = params.kmax
        dk = params.dk
        nbins = params.nbins

        boxsize = params.boxsize
        mesh_gen = params.mesh_gen
        mesh_cal = params.mesh_cal

        niter = params.niter
        prog_id = params.prog_id
    except AttributeError as attr_err:
        print(attr_err)

    global Plin

    Plin = cosmology.LinearPower(
        cosmology.Planck15,
        redshift=redshift,
        transfer='CLASS',
    )

    global filename_root

    if generator.lower().startswith('g'):
        gen_name = "gaussian"
    elif generator.lower().startswith('l'):
        gen_name = "lognormal"
    elif generator.lower().startswith('n'):
        gen_name = "nbodykit"
        if not cataloguing:
            cataloguing = True
            warnings.warn(
                "'--generator' flag value 'nbodykit' has "
                "overriden `cataloguing` to False. ",
                RuntimeWarning,
            )

    if cataloguing:
        obj_name = "catalogue"
        if not sampling:
            sampling = True
            warnings.warn(
                "'--non-catalogue' flag value False has "
                "overriden `sampling` to False. ",
                RuntimeWarning,
            )
    elif sampling:
        obj_name = "mesh"
    else:
        obj_name = "realisation"

    if mesh_gen == mesh_cal:
        mesh_tag = f"mesh=gc{mesh_gen}"
    else:
        mesh_tag = f"mesh=[g{mesh_gen},c{mesh_cal}]"

    filename_root = "-".join([obj_name, gen_name])

    param_tag = "nbar={},bias={},boxsize={},kmax={},".format(
        format_float(nbar, 'sci'),
        format_float(bias, 'decdot'),
        format_float(boxsize, 'intdot'),
        format_float(kmax, 'sci'),
    )

    iter_tag = "iter={}".format(niter)

    part_tags = ["-(", param_tag, mesh_tag, iter_tag, ")-", "[", prog_id, "]"]
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
    print(filename_root)
    print(runtime_info.strip("-"))

    suite = defaultdict(list)
    if cataloguing:
        for run in range(niter):
            catalogue = CATALOGUE[generator](
                Plin,
                nbar,
                bias=bias,
                boxsize=boxsize,
                num_mesh=mesh_gen,
            )
            mesh = catalogue.to_mesh(
                Nmesh=mesh_cal,
                resampler='tsc',
                compensated=True,
            )
            cartesian_power = FFTPower(mesh, mode='1d', kmax=kmax, dk=dk).power

            suite['k'].append([cartesian_power['k']])
            suite['Nk'].append([cartesian_power['modes']])
            suite['Pk'].append([cartesian_power['power'].real])
            suite['Pshot'].append([cartesian_power.attrs['shotnoise']])
    else:
        for run in range(niter):
            field = MECHANISM[generator](boxsize, mesh_gen, Plin, bias=bias)
            if sampling: field = poisson_sample(field, nbar, boxsize)
            k, Pk, Nk = cal_power(field, boxsize, kmax=kmax, nbins=nbins)

            suite['k'].append([k])
            suite['Nk'].append([Nk])
            suite['Pk'].append([Pk])
            suite['Pshot'].append([0.])

    output_data = {
        var: np.concatenate(vals)
        for var, vals in suite.items()
    }
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

    """
    basepath = f"{PATHOUT}{filename_root}"
    assert confirm_dir(basepath)

    filename = f"{filename_root}{program_tag}"
    if save:
        np.save("".join([basepath, "/", filename, ".npy"]), output_data)
    if plot:
        dof_k = np.size(output_data['k'], axis=-1) - 1
        dof_P = np.size(output_data['Pk'], axis=0) - 1

        results = {
            'Nk': np.sum(output_data['Nk'], axis=0),
            'k': np.average(output_data['k'], axis=0),
            'Pk': np.average(output_data['Pk'], axis=0),
            'Pshot': np.average(output_data['Pshot']),
            'dk': np.std(output_data['k'], axis=0, ddof=1) / np.sqrt(dof_k),
            'dPk': np.std(output_data['Pk'], axis=0, ddof=1) / np.sqrt(dof_P),
        }

        Pk_model = bias**2 * Plin(results['k']) + 1 / nbar

        try:
            plt.style.use(harmony)
            plt.close('all')

            plt.loglog(results['k'], Pk_model, label='model')
            plt.errorbar(
                results['k'],
                results['Pk'],
                xerr=results['dk'],
                yerr=results['dPk'],
                label='catalogue',
            )

            plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
            plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
            plt.legend()
            plt.savefig("".join([basepath, "/", filename, ".pdf"]))
        except Exception as e:
            print(e)


if __name__ == '__main__':

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
