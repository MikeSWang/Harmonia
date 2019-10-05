"""Measure 2-point function values from a suite of realisations.

"""
from collections import defaultdict

import numpy as np
from nbodykit.lab import cosmology

from agreement_rc import PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
)
from harmonia.mapper import (
    GaussianCatalogue,
    LogNormalCatalogue,
    NBKCatalogue,
    SphericalMap,
)

GEN_CATALOGUE = {
    'gaussian': GaussianCatalogue,
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
    global pivots, rsd_flag, nbar, contrast, bias, redshift, zmax, kmax, \
        expand, mesh_gen, mesh_cal, niter, prog_id

    try:
        pivots = params.structure.split(",")
        rsd_flag = params.rsd
        generator = params.generator

        nbar = params.nbar
        bias = params.bias
        redshift = params.redshift
        zmax = params.zmax
        kmax = params.kmax
        expand = params.expand
        mesh_gen = params.mesh_gen
        mesh_cal = params.mesh_cal
        niter = params.niter
        prog_id = params.prog_id
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global Plin, rmax, beta, gen_name

    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')
    rmax = cosmo.comoving_distance(zmax)
    beta = cosmo.scale_independent_growth_rate(redshift) / bias

    if generator.lower().startswith('g'):
        gen_name = "gaussian"
    elif generator.lower().startswith('l'):
        gen_name = "lognormal"
    elif generator.lower().startswith('n'):
        gen_name = "nbodykit"
    gen_tag = f"{gen_name},"

    if rsd_flag:
        rsd_tag = "{:.2f}".format(beta)
    else:
        rsd_tag = 'none'

    if len(pivots) > 1:
        pivot_tag = "pivots={},".format(pivots).replace("'", "")
    else:
        pivot_tag = "pivots={},".format(params.structure).replace("'", "")

    param_tag = "nbar={},bias={},beta={},rmax={},kmax={},xpd={},".format(
        format_float(nbar, 'sci'),
        format_float(bias, 'decdot'),
        rsd_tag,
        format_float(rmax, 'intdot'),
        format_float(kmax, 'sci'),
        format_float(expand, 'decdot'),
    )

    if mesh_gen == mesh_cal:
        mesh_tag = f"mesh=gc{mesh_gen},"
    else:
        mesh_tag = f"mesh=[g{mesh_gen},c{mesh_cal}],"

    iter_tag = "iter={}".format(niter)

    tags = (gen_tag, pivot_tag, param_tag, mesh_tag, iter_tag)
    runtime_info = "".join(["-(", *tags, ")-", "[", prog_id, "]"])
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
    disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)

    output_data = defaultdict(list)
    for run in range(niter):
        catalogue = GEN_CATALOGUE[gen_name](
            Plin,
            nbar,
            bias=bias,
            boxsize=boxsize,
            num_mesh=mesh_gen,
            add_RSD=rsd_flag
        )
        spherical_map = SphericalMap(disc, catalogue, mean_density_data=nbar)

        for key in pivots:
            output_data[key].append(
                spherical_map.two_points_pivoted(pivot=key)
            )

    return output_data


def finalise(output_data, save=True):
    """Program finalisation with optional data and figure saving.

    Parameters
    ----------
    output_data : dict
        Program output.
    save : bool, optional
        If `True`, aggregate data is saved as :obj:`dict`.

    Raises
    ------
    AssertionError
        If the output path does not exist.

    """
    base_path = f"{PATHOUT}{script_name}"
    assert confirm_dir(base_path)

    filename = f"{script_name}{program_tag}"
    if save:
        np.save("".join([base_path, "/", filename, ".npy"]), output_data)


if __name__ == '__main__':

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
