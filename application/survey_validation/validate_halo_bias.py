"""Validate simulation halo bias values from rapid full-box likelihood
inference.

"""
import os
import sys
from argparse import ArgumentParser

# pylint: disable=no-name-in-module
import numpy as np
from nbodykit.algorithms import FFTPower
from nbodykit.cosmology import LinearPower
from scipy.special import loggamma
from tqdm import tqdm

try:
    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.mapper import SourceCatalogue
    from harmonia.cosmology import BaseModel, scale_dependence_modification
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.mapper import SourceCatalogue
    from harmonia.cosmology import BaseModel, scale_dependence_modification


@display_args(logger=logger)
def initialise_parameters():
    """Initialise the program parameters passed from ``stdin``.

    Returns
    -------
    parsed_args : :class:`argparse.Namespace`
        Parsed parameter namespace.

    """
    parser = ArgumentParser()

    parser.add_argument(
        '--series', type=str, required=True, help="catalogue source series"
    )
    parser.add_argument(
        '--serial-number', type=str, required=True,
        help="catalogue source serial number"
    )
    parser.add_argument(
        '--cosmology-file', type=str, required=True,
        help="catalogue source cosmological parameter file"
    )
    parser.add_argument(
        '--mass-cut', type=float, default=None,
        help="mass cut in solar masses"
    )
    parser.add_argument(
        '--boxsize', type=float, default=1000.0, help="catalogue box size"
    )
    parser.add_argument(
        '--density', type=float, required=True,
        help="catalogue mean number density"
    )
    parser.add_argument(
        '--NG', type=str, required=True, help="catalogue source PNG value"
    )
    parser.add_argument(
        '--bias', type=float, nargs=3, help="bias sampling values"
    )
    parser.add_argument(
        '--kmin', type=float, default=None, help="minimum wavenumber"
    )
    parser.add_argument(
        '--kmax', type=float, default=0.1, help="maximum wavenumber"
    )

    parsed_args = parser.parse_args()

    return parsed_args


def make_catalogue_measurements(kmin=None, kmax=None):
    """Measure catalogue power spectrum at binned wavenumbers.

    Parameters
    ----------
    kmin : float or None, optional
        Minimum wavenumber (default is `None`).
    kmax : float or None, optional
        Maximum wavenumber (default is `None`).

    Returns
    -------
    dict
        Power spectrum measured at binned wavenumbers.

    """
    kmin, kmax = kmin or KMIN, kmax or KMAX

    catalogue = SourceCatalogue(
        input_dir/input_filename,
        headings=["x", "y", "z", "vx", "vy", "vz", "mass"],
        mean_density=progrc.density, boxsize=progrc.boxsize
    )
    if progrc.mass_cut:
        catalogue['Selection'] = np.greater_equal(
            catalogue['mass'], progrc.mass_cut
        )

    fftmesh = catalogue.to_mesh(Nmesh=NMESH, resampler='tsc', compensated=True)

    results = FFTPower(fftmesh, mode='1d', kmin=kmin, kmax=kmax)

    k = results.power['k']
    pk = results.power['power'].real
    nk = results.power['modes']
    shotnoise = results.power.attrs['shotnoise']

    valid_bins = nk > 1

    return {
        'k': k[valid_bins], 'pk': pk[valid_bins], 'nk': nk[valid_bins],
        'shotnoise': [shotnoise],
    }


def log_likelihood(b_1):
    """Evaluate the Gaussianised log-likelihood at a given bias.

    Parameters
    ----------
    b_1 : float
        Model scale-independent linear bias.

    Returns
    -------
    float
        Gaussnianised log-likelihood value.

    """
    model_vector = 1 / progrc.density \
        + (b_1 + (b_1 - TRACER_P) * mode_modifications) ** 2 * mode_powers

    gaussianised_model_vector = gaussianised_expectation_factor \
        * model_vector ** (1./3.)
    gaussianised_model_variance = gaussianised_variance_factor \
        * model_vector ** (2./3.)

    return - 1./2. * (
        np.sum(np.log(gaussianised_model_variance)) +
        np.sum(
            (gaussianised_data_vector - gaussianised_model_vector) ** 2
            / gaussianised_model_variance
        )
    )


KMIN = 1.e-4
KMAX = 0.1
Z = 1.
TRACER_P = 1.
NMESH = 256

if __name__ == '__main__':

    progrc = initialise_parameters()

    # Set I/O paths.
    cosmo_dir = data_dir/"external"/"cosmology"

    input_dir = data_dir/"external"/"simulations"
    input_filename = progrc.series + '-{}.txt'.format(progrc.serial_number)

    output_dir = data_dir/"raw"/"survey_validation"
    output_filename = "likelihood-({})".format(",".join([
        "source={}".format(input_filename.replace('.txt', '')),
        "map=cubic", "scale=[{},None,{}]".format(progrc.kmin, progrc.kmax),
        "orders=[0]", "excl_monop=False", "rsd=False",
        "mask=None", "selection=None"
    ]))

    confirm_directory(output_dir)

    # Make catalogue measurements.
    measurements = \
        make_catalogue_measurements(kmin=progrc.kmin, kmax=progrc.kmax)

    np.savez(
        output_dir/output_filename.replace("likelihood", "pk"),
        **measurements
    )

    # Make model predictions.
    cosmo = BaseModel(cosmo_dir/progrc.cosmology_file)

    mode_modifications = float(progrc.NG) \
        * scale_dependence_modification(cosmo, Z)(measurements['k'])
    mode_powers = LinearPower(cosmo, Z)(measurements['k'])

    gaussianised_data_vector = measurements['pk'] ** (1./3.)
    gaussianised_expectation_factor = np.array([
        np.exp(loggamma(n + 1./3.) - loggamma(n)) / n ** (1./3.)
        for n in measurements['nk']
    ])
    gaussianised_variance_factor = np.array([
        np.exp(loggamma(n + 2./3.) - loggamma(n)) / n ** (2./3.)
        - np.exp(loggamma(n + 1./3.) - loggamma(n)) ** 2 / n ** (2./3.)
        for n in measurements['nk']
    ])

    # Evaluate likelihood and export results.
    bias_start, bias_end, ninterval = progrc.bias

    bias_grid = np.linspace(bias_start, bias_end, num=ninterval+1)
    likelihood_grid = np.zeros_like(bias_grid)

    for idx, bias in tqdm(enumerate(bias_grid)):
        likelihood_grid[idx] = log_likelihood(bias)

    np.savez(
        output_dir/output_filename,
        likelihood=likelihood_grid, b_1=bias_grid, f_nl=[float(progrc.NG)]
    )
