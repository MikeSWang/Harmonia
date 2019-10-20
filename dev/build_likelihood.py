"""Sample spherical likelihood for primordial non-Gaussianity.

"""
import numpy as np
from nbodykit.lab import cosmology

from likelihood_rc import PATHIN, PATHOUT, params, script_name
from spherical_likelihood import (
    _f_nl_parametrised_chi_square as chi_square,
#    spherical_map_f_nl_likelihood as likelihood,
)
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
)
from harmonia.cosmology import fiducial_cosmology, fiducial_distance
from harmonia.mapper import LogNormalCatalogue, NBKCatalogue, SphericalMap
from harmonia.reader import TwoPointFunction

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
        If a required input parameter is missing.

    """
    global prior_range, num_sample, pivot, generator, rsd_flag, nbar, bias, \
        redshift, rmax, kmax, expand, mesh_gen, mesh_cal, niter, prog_id

    try:
        prior_range = params.prior_range
        num_sample = params.num_sample
        pivot = params.pivot

        if params.generator.lower().startswith('l'):
            generator = "lognormal"
        elif params.generator.lower().startswith('n'):
            generator = "nbodykit"
        rsd_flag = params.rsd

        nbar = params.nbar
        bias = params.bias
        redshift = params.redshift
        rmax = fiducial_distance(params.zmax)
        kmax = params.kmax

        expand = params.expand
        mesh_gen = params.mesh_gen
        mesh_cal = params.mesh_cal
        niter = params.niter
        prog_id = params.prog_id
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global external_couplings, Plin, growth_rate

    external_couplings = np.load(
        f"{PATHIN}predict_twopt"
        "-(pivots=[natural,spectral],nbar=0.001,b1=2.,f0=none,"
        "rmax=148.,kmax=0.1)-"
        "couplings.npy"
    ).item()

    Plin = cosmology.LinearPower(fiducial_cosmology, redshift=redshift)

    if rsd_flag:
        growth_rate = fiducial_cosmology.scale_independent_growth_rate(redshift)
        rsd_tag = "{:.2f}".format(growth_rate)
    else:
        growth_rate = None
        rsd_tag = 'none'

    param_tag = "gen={},pivot={},nbar={},b1={},f0={},rmax={},kmax={},xpd={}," \
        .format(
            generator,
            pivot,
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

    runtime_info = "".join(["-(", param_tag, mesh_tag, iter_tag, ")-",
        "[", prog_id, "]"])
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

    twopt_args = disc, nbar, bias
    twopt_kwargs = dict(
        f_0=growth_rate,
        cosmo=fiducial_cosmology,
        couplings=external_couplings,
    )

    two_point_model = TwoPointFunction(*twopt_args, **twopt_kwargs)

    sample_parameters = np.linspace(*prior_range, num=num_sample)

    chi_square_samples = []
#    likelihood_samples = []
    for run in range(niter):
        catalogue = GEN_CATALOGUE[generator](
            Plin,
            nbar,
            bias=bias,
            boxsize=2*expand*rmax,
            num_mesh=mesh_gen,
            add_RSD=rsd_flag
        )

        spherical_map = SphericalMap(disc, catalogue, mean_density_data=nbar)

        n_coeff, nbar_coeff = spherical_map.transform()
        overdensity = [
            n_coeff[ell] - nbar_coeff[ell]
            for ell in np.sort(disc.degrees)
        ]

        field_vector = SphericalArray.build(disc=disc, filling=overdensity) \
            .unfold(pivot, return_only='data')

        sample_chi_square = chi_square(
            sample_parameters,
            field_vector,
            pivot,
            two_point_model
        )

#        sample_likelihood = likelihood(
#            sample_parameters,
#            field_vector,
#            pivot,
#            two_point_model
#        )

        chi_square_samples.append(sample_chi_square)
#        likelihood_samples.append(sample_likelihood)

    output_data = {
        'f_nl': [sample_parameters],
        'chi_square': chi_square_samples,
#        'likelihood': likelihood_samples,
    }

    return output_data


def finalise(output_data, save=True):
    """Program finalisation with optional data and figure saving.

    Parameters
    ----------
    output_data : dict
        Program output.
    save : bool, optional
        If `True`, aggregate data over all iterations is saved as a
        dictionary.

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
