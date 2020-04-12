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
    from harmonia.cosmology import BaseModel, modified_power_spectrum
    from harmonia.reader import Couplings, SphericalCorrelator
    from harmonia.utils import covar_to_corr
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import confirm_directory, data_dir
    from harmonia.cosmology import BaseModel, modified_power_spectrum
    from harmonia.reader import Couplings, SphericalCorrelator
    from harmonia.utils import covar_to_corr


def validate_fullsky_spherical_model(b_1=2.3415, f_nl=100., nbar=2.5e-4,
                                     contrast=10.):
    """Validate full-sky spherical modelling against baseline power
    spectrum model.

    """
    # External information.
    simulation_cosmo = BaseModel(cosmo_dir/cosmo_file)

    fullsky_couplings = Couplings.load(
        survey_product_dir/couplings_file.format(kmax, "1.0", "None")
    )

    sort_order = np.argsort(
        list(fullsky_couplings.disc.wavenumbers.values())
    )
    wavenumbers = np.array(
        list(fullsky_couplings.disc.wavenumbers.values())
    )[sort_order]
    normalisations = np.array(
        list(fullsky_couplings.disc.normalisations.values())
    )[sort_order]

    # Spherical model.
    global spherical_model

    spherical_model = SphericalCorrelator(
        fullsky_couplings.disc, REDSHIFT,
        cosmo=simulation_cosmo, growth_rate=0., couplings=fullsky_couplings,
        comm=comm
    )

    spherical_correlator_matrix = spherical_model.correlator_matrix(
        "spectral", b_1=b_1, f_nl=f_nl, nbar=nbar, contrast=contrast,
        radialise=True
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
    plt.loglog(wavenumbers, spherical_spectrum, label='spherical')
    plt.loglog(wavenumbers, cartesian_spectrum, ls='--', label='Cartesian')
    plt.xlabel(r"$k\ \  [h/\mathrm{{Mpc}}]$")
    plt.ylabel(r"$P(k)\ \ [(\mathrm{{Mpc}}/h)^3]$")
    plt.legend()
    plt.title("Full-sky spherical model validation")

    return _ratios


def validate_spherical_correlator_model(b_1=2.3415, f_nl=100., nbar=2.5e-4,
                                        contrast=10.):
    """Validate generic spherical correlator predictions against estimates
    from large sample sets.

    """
    # External information.
    simulation_cosmo = BaseModel(cosmo_dir/cosmo_file)

    couplings = Couplings.load(
        survey_product_dir/couplings_file.format(kmax, mask_tag, selection_tag)
    )

    spherical_correlator_estimate = covar_to_corr(np.load(
        raw_product_dir/covar_estimate_file
    ))

    # Spherical model.
    global spherical_model

    spherical_model = SphericalCorrelator(
        couplings.disc, REDSHIFT,
        cosmo=simulation_cosmo, growth_rate=0., couplings=couplings,
        comm=comm
    )

    spherical_correlation_model = covar_to_corr(
        spherical_model.correlator_matrix(
            "natural", b_1=b_1, f_nl=f_nl, nbar=nbar, contrast=contrast,
            report_progress=True
        )
    )

    _ratios = spherical_correlator_estimate / spherical_correlation_model - 1

    sns.set(style='ticks', font='serif')
    plt.figure(figsize=(10, 2.75))
    plt.subplot2grid((1, 3), (0, 0))
    sns.heatmap(
        spherical_correlation_model.real,
        rasterized=True, square=True, center=0., vmin=-1., vmax=1.
    )
    plt.subplot2grid((1, 3), (0, 1))
    sns.heatmap(
        spherical_correlator_estimate.real,
        rasterized=True, square=True, center=0., vmin=-1., vmax=1.
    )
    plt.subplot2grid((1, 3), (0, 2))
    sns.heatmap(
        spherical_correlator_estimate.real - spherical_correlation_model.real,
        cmap='coolwarm', rasterized=True, square=True, center=0.
    )

    plt.subplots_adjust(
        wspace=0.4, bottom=0.175, top=0.825, left=0.05, right=0.95
    )

    return _ratios


REDSHIFT = 1.

spherical_model = None

if __name__ == '__main__':

    comm = MPI.COMM_WORLD

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
    covar_estimate_file = "covar-estimate-({}).npy".format(
        ",".join([
            "source=1-2500", "map=spherical",
            "scale=[None,{}]".format(kmax), "orders=None",
            "mask={}".format(mask_tag), "selection={}".format(selection_tag)
        ])
    )

    # NOTE: Not used yet.
    output_dir = data_dir/"processed"/"survey_validation"

    # Validate spherical modelling.
    confirm_directory(output_dir)  # NOTE: Not used yet.
    ratios_fullsky = validate_fullsky_spherical_model()
    ratios_partsky = validate_spherical_correlator_model()
