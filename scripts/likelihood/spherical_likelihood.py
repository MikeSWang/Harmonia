"""Sample the spherical likelihood from simulations.

"""
from collections import defaultdict
from pprint import pprint

import numpy as np
from mpi4py import MPI
from nbodykit.cosmology import Cosmology

from likelihood_rc import PATHIN, PATHOUT, DATAPATH, logger, script_name
from likelihood_rc import parse_external_args
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import confirm_directory_path
from harmonia.reader import TwoPointFunction
from harmonia.reader import spherical_map_log_likelihood as sph_likelihood

# Cosmological input.
COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"

# Catalogue map input.
MAP_PATH = DATAPATH/"spherical_map"

# Survey specfications input.
SPECS_PATH = PATHIN/"specifications"
COUPLINGS_FILE = "couplings-(fsky={:.2f},kmax={}).npy"

# Likelihood input.
FIXED_PARAMS_FILE = PATHIN/"fixed_parameters.txt"
SAMPLED_PARAMS_FILE = PATHIN/"sampled_parameters.txt"

# Global quantities.
simu_cosmo = None
external_couplings = None

fixed_params = None
sampled_params = None


def initialise():
    """Initialise program by setting parameters.

    Returns
    -------
    ini_params : dict
        Initialised parameters.
    ini_tag :
        Initialised program tag.

    """
    ini_params = parsed_params.__dict__

    # Extract fixed parameter values and sampled parameter ranges.
    global fixed_params, sampled_params

    fixed_tag = ""
    fixed_params = {}
    with open(FIXED_PARAMS_FILE, 'r') as fixed_par_file:
        for name, values in eval(fixed_par_file.read()).items():
            fixed_tag += "{}={},".format(name, values[0])
            if name == 'fnl':
                name = 'non_gaussianity'
            fixed_params.update(
                {name : dict(zip(['spherical', 'cartesian'], values))}
            )
    ini_params.update({'fixed_params': fixed_tag.strip(",")})

    sampled_tag = ""
    sampled_params = {}
    with open(SAMPLED_PARAMS_FILE, 'r') as samp_par_file:
        for name, (ranges, num_sample) in eval(samp_par_file.read()).items():
            sampled_tag += "{}_prior=[{},{}],".format(name, *ranges)
            if name == 'fnl':
                name = 'non_gaussianity'
            sampled_params.update(
                {name: np.linspace(*ranges, num=num_sample+1)}
            )
    ini_params.update({'sampled_params': sampled_tag.strip(",")})

    growth_rate = None if parsed_params.rsd else 0
    ini_params.update({'growth_rate': growth_rate})

    ini_tag = "map={},fsky={:.2f},knots=[{},{}],rsd={},{}{}".format(
        parsed_params.map, parsed_params.fsky,
        parsed_params.kmin, parsed_params.kmax,
        parsed_params.rsd, sampled_tag, fixed_tag,
    ).strip(",")

    # Extract cosmology and survey specifications.
    global simu_cosmo, external_couplings

    with open(COSMOLOGY_FILE) as cosmology_file:
        cosmological_parameters = eval(cosmology_file.read())
        simu_cosmo = Cosmology(
            h=cosmological_parameters['h'],
            Omega0_b=cosmological_parameters['Omega0_b'],
            Omega_cdm=cosmological_parameters['Omega_cdm']
        ).match(cosmological_parameters['sigma8'])

    ini_params.update({'sph_mode_independence': True})
    if parsed_params.load_couplings:
        try:
            external_couplings = np.load(
                SPECS_PATH/COUPLINGS_FILE.format(
                    parsed_params.fsky, str(parsed_params.kmax).rstrip("0"),
                )
            ).item()
        except FileNotFoundError:
            pass
        else:
            ini_params.update({'sph_mode_independence': False})

    if comm.rank == 0:
        pprint(ini_params)
        print("\n")

    return ini_params, ini_tag


def process():
    """Process program.

    Returns
    -------
    dict
        Program output.

    """
    disc = DiscreteSpectrum(
        params['boxsize']/2, 'dirichlet', params['kmax'], comm=comm
    )

    two_point_model = TwoPointFunction(
        disc,
        redshift=params['redshift'],
        growth_rate=params['growth_rate'],
        cosmo=simu_cosmo,
        couplings=external_couplings,
        comm=comm
    )

    return two_point_model

#    map_file = params['input_catalogue'] \
#        + "-(map={},fsky={:.2f},knots=[{},{}],rsd={}).npy".format(
#            params['map'], params['fsky'],
#            params['kmin'], params['kmax'],
#            params['rsd']
#        )
#    map_data = np.load(MAP_PATH/map_file).item()
#
#    output_data = defaultdict(list)
#    for file_suffix in ["L.txt", "R.txt"]:
#        # Load map data.
#        spherical_data = SphericalArray.build(
#            disc=disc, filling=map_data[file_suffix]
#        )
#
#        # Construct spherical likelihood.
#        sph_likelihood_kwargs = dict(
#            two_point_model=two_point_model,
#            spherical_data=spherical_data,
#            mean_number_density=params['nbar'],
#            contrast=params['contrast'],
#            pivot=params['spherical_pivot'],
#            breakdown=params['breakdown'],
#            independence=params['sph_mode_independence'],
#            logger=logger,
#            comm=comm,
#        )
#        for par_name, par_values in fixed_params.items():
#            sph_likelihood_kwargs.update({par_name: par_values['spherical']})
#        for par_name, par_values in sampled_params.items():
#            sph_likelihood_kwargs.update({par_name: par_values})
#
#        output_data['spherical_likelihood'].append(
#            [sph_likelihood(**sph_likelihood_kwargs)]
#        )
#
#    return output_data


def finalise(results, filetag):
    """Finalise program.

    Parameters
    ----------
    results : dict
        Output data.
    filetag : dict
        Output file tag.

    """
    assert confirm_directory_path(PATHOUT/script_name)

    filename = f"{params['input_catalogue']}-({filetag}).npy"

    if comm.rank == 0:
        np.save(PATHOUT/script_name/filename, results)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    parsed_params = parse_external_args()
    params, tag = initialise()
    output = process()
    # finalise(output, tag)
