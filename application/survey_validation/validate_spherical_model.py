"""Validate spherical modelling.

"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpi4py import MPI

try:
    from application import confirm_directory, data_dir
    from harmonia.algorithms import DiscreteSpectrum
    from harmonia.cosmology import BaseModel, modified_power_spectrum
    from harmonia.reader import Couplings, SphericalCorrelator
    from harmonia.utils import covar_to_corr
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import confirm_directory, data_dir
    from harmonia.algorithms import DiscreteSpectrum
    from harmonia.cosmology import BaseModel, modified_power_spectrum
    from harmonia.reader import Couplings, SphericalCorrelator
    from harmonia.utils import covar_to_corr


def validate_fullsky_spherical_model(b_1=2.3415, f_nl=100., nbar=2.5e-4,
                                     contrast=10., use_shortcut=True):
    """Validate full-sky spherical modelling against baseline power
    spectrum model.

    Parameters
    ----------
    b_1, f_nl, nbar, contrast : float, optional
        Cosmological and model parameters to pass to the model.  Defaults
        are 2.3415, 100., 2.5e-4 and 10. respectively.
    use_shortcut : bool, optional
        If `True` (default), short cuts (Kronecker delta reduction in the
        absence of couplings and diagonalisation of correlator matrix)
        are used.

    """
    # External information.
    simulation_cosmo = BaseModel(cosmo_dir/cosmo_file)

    if use_shortcut:
        fullsky_couplings = None
        disc = DiscreteSpectrum(rmax, 'dirichlet', kmax)
    else:
        fullsky_couplings = Couplings.load(
            survey_product_dir/couplings_file.format(kmax, "1.0", "None")
        )
        disc = fullsky_couplings.disc

    sort_order = np.argsort(list(disc.wavenumbers.values()))
    wavenumbers = np.array(list(disc.wavenumbers.values()))[sort_order]
    normalisations = np.array(list(disc.normalisations.values()))[sort_order]

    # Spherical model.
    # pylint: disable=global-statement
    global spherical_model_fullsky
    spherical_model_fullsky = SphericalCorrelator(
        disc, REDSHIFT,
        cosmo=simulation_cosmo, growth_rate=0., couplings=fullsky_couplings,
        comm=comm
    )

    spherical_correlator_matrix = spherical_model_fullsky.correlator_matrix(
        "spectral", b_1=b_1, f_nl=f_nl, nbar=nbar, contrast=contrast,
        diagonal=True
    )

    _, unique_ind = np.unique(
        np.diag(np.around(spherical_correlator_matrix, decimals=0)),
        return_index=True
    )
    spherical_spectrum = normalisations * \
        np.diag(spherical_correlator_matrix)[np.sort(unique_ind)]

    # Cartesian model.
    cartesian_spectrum = modified_power_spectrum(
        b_1=b_1, f_nl=f_nl, cosmo=simulation_cosmo, redshift=REDSHIFT
    )(wavenumbers) + (1 + 1 / contrast) / nbar

    _ratios = spherical_spectrum / cartesian_spectrum - 1

    sns.set(style='ticks', font='serif')
    plt.figure("full sky spherical model validation")
    plt.loglog(wavenumbers, spherical_spectrum, label='spherical')
    plt.loglog(wavenumbers, cartesian_spectrum, ls='--', label='Cartesian')
    plt.xlabel(r"$k\ \  [h/\mathrm{{Mpc}}]$")
    plt.ylabel(r"$P(k)\ \ [(\mathrm{{Mpc}}/h)^3]$")
    plt.legend()
    plt.title("Full-sky spherical model validation")

    output_file = output_dir/output_filename.format("1.0", "None")
    plt.savefig(output_file.with_suffix(output_file.suffix + '.pdf'))

    return _ratios


def validate_spherical_correlator_model(b_1=1., f_nl=0., nbar=2.5e-4,
                                        contrast=10.):
    """Validate generic spherical correlator predictions against estimates
    from large sample sets.

    Parameters
    ----------
    b_1, f_nl, nbar, contrast : float, optional
        Cosmological and model parameters to pass to the model.  Defaults
        are 1., 0., 2.5e-4 and 10. respectively.

    """
    # External information.
    simulation_cosmo = BaseModel(cosmo_dir/cosmo_file)

    couplings = Couplings.load(
        survey_product_dir/couplings_file.format(kmax, mask_tag, selection_tag)
    )

    spherical_correlation_estimate = covar_to_corr(np.load(
        raw_product_dir/corr_estimate_file
    ))

    # Spherical model.
    # pylint: disable=global-statement
    global spherical_model_generic
    spherical_model_generic = SphericalCorrelator(
        couplings.disc, REDSHIFT,
        cosmo=simulation_cosmo, growth_rate=0., couplings=couplings,
        comm=comm
    )

    spherical_correlation_model = covar_to_corr(
        spherical_model_generic.correlator_matrix(
            "natural", b_1=b_1, f_nl=f_nl, nbar=nbar, contrast=contrast,
            report_progress=True
        )
    )

    _ratios = spherical_correlation_estimate / spherical_correlation_model - 1

    sns.set(style='ticks', font='serif')
    plt.figure("partial sky spherical model validation", figsize=(10, 2.75))
    plt.subplot2grid((1, 3), (0, 0))
    sns.heatmap(
        spherical_correlation_model.real,
        rasterized=True, square=True, center=0., vmin=-1., vmax=1.
    )
    plt.subplot2grid((1, 3), (0, 1))
    sns.heatmap(
        spherical_correlation_estimate.real,
        rasterized=True, square=True, center=0., vmin=-1., vmax=1.
    )
    plt.subplot2grid((1, 3), (0, 2))
    sns.heatmap(
        spherical_correlation_estimate.real - spherical_correlation_model.real,
        cmap='coolwarm', rasterized=True, square=True, center=0.
    )

    plt.subplots_adjust(
        wspace=0.4, bottom=0.175, top=0.825, left=0.05, right=0.95
    )

    output_file = output_dir/output_filename.format(mask_tag, selection_tag)
    plt.savefig(output_file.with_suffix(output_file.suffix + '.pdf'))

    return _ratios


REDSHIFT = 1.

spherical_model_fullsky, spherical_model_generic = None, None

if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rmax = 500.
    kmax = 0.04
    mask_tag = "0.5" # "random0_BOSS_DR12v5_CMASS_North"
    selection_tag = "None" # "[100.0,500.0]"

    # Set I/O paths.
    cosmo_dir = data_dir/"external"/"cosmology"
    cosmo_file = "simulation.txt"

    survey_product_dir = data_dir/"processed"/"survey_products"
    couplings_file = "couplings-({}).npz".format(
        ",".join(["kmax={}", "mask={}", "selection={}"])
    )

    raw_product_dir = data_dir/"raw"/"survey_products"
    corr_estimate_file = "covar-estimate-({}).npy".format(",".join([
        "source=1-2500", "map=spherical",
        "scale=[None,{}]".format(kmax),
        "orders=None",
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    output_dir = data_dir/"raw"/"survey_validation"
    output_filename = "spherical-validation-({})".format(",".join([
        "scale=[None,{}]".format(kmax),
        "rsd=False", "mask={}", "selection={}"
    ]))

    # Validate spherical modelling.
    confirm_directory(output_dir)  # NOTE: Not used yet.
    ratios_fullsky = validate_fullsky_spherical_model()
    ratios_partsky = validate_spherical_correlator_model()
    print("Spherical-to-Cartesian relative difference:", ratios_fullsky)
