"""Validate Cartesian modelling.

"""
import os
import sys
from collections import defaultdict
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from application import confirm_directory, data_dir
    from harmonia.algorithms import CartesianArray
    from harmonia.cosmology import BaseModel
    from harmonia.reader import CartesianMultipoles
    from harmonia.surveyor.synthesis import CovarianceEstimator
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import confirm_directory, data_dir
    from harmonia.algorithms import CartesianArray
    from harmonia.cosmology import BaseModel
    from harmonia.reader import CartesianMultipoles
    from harmonia.surveyor.synthesis import CovarianceEstimator


def collate_map_data():
    """Collate Cartesian map data.

    Returns
    -------
    wavenumbers : float :class:`numpy.ndarray`
        Wavenumber of Cartesian map multipoles.
    multipoles : dict{int: float :class:`numpy.ndarray`}
        Power multipoles of different Legendre orders at `wavenumbers`.

    """
    wavenumbers = []
    multipoles = defaultdict(list)
    for source_tag in source_tags:
        map_data = CartesianArray.load(
            map_data_dir/map_data_file.format(*source_tag)
        )

        wavenumbers.append(np.unique(map_data.array['wavenumber']))
        orders = np.unique(map_data.array['order'])
        for ell in orders:
            power = map_data.array['power'][map_data.array['order'] == ell]
            multipoles[ell].append(power)

    wavenumbers = np.mean(wavenumbers, axis=0)
    multipoles = {
        ell: np.mean(multipoles[ell], axis=0)
        for ell in multipoles.keys()
    }

    return wavenumbers, multipoles


def compare_with_cartesian_model(wavenumbers, multipole_data):
    """Compare the map data with Cartesian model predictions.

    Parameters
    ----------
    wavenumbers : float :class:`numpy.ndarray`
        Wavenumbers.
    multipole_data : dict{int: float :class:`numpy.ndarray`}
        Measured multipoles from map data.

    Returns
    -------
    ratios : dict{int: float :class:`numpy.ndarray`}
        Ratio of the measured multipoles to model predictions.

    """
    global cartesian_model
    simulation_cosmo = BaseModel(cosmo_dir/cosmo_file)

    mask_multipoles = np.load(survey_product_dir/mask_file)

    window_multipoles = CartesianArray.load(
        survey_product_dir/window_file
    )

    covariance_estimator = CovarianceEstimator.load(
        survey_product_dir/covariance_estimate_file
    )

    std_estimate = np.sqrt(np.diag(
        covariance_estimator.get_fiducial_covariance(pivot='order')
    )) / (1 / 2.5e-4)

    cartesian_model = CartesianMultipoles(
        wavenumbers, redshift=REDSHIFT, cosmo=simulation_cosmo, growth_rate=0.,
        mask_multipoles=mask_multipoles, window_multipoles=window_multipoles
    )

    multipole_model = {
        order: cartesian_model.convolved_power_multipoles(
            orders=[order], b_1=2.3415, f_nl=NG, nbar=2.5e-4, contrast=10.
        ).array['power']
        for order in multipole_data.keys()
    }

    ratios = {
        order: multipole_data[order] / multipole_model[order] - 1
        for order in multipole_data.keys()
    }

    sns.set(style='ticks', font='serif')
    plt.figure()
    for order in multipole_data.keys():
        plt.loglog(
            wavenumbers, multipole_model[order],
            ls='--', label='model'
        )
        plt.errorbar(
            wavenumbers, multipole_data[order],
            yerr=std_estimate*multipole_data[order]/np.sqrt(len(source_tags)),
            marker='s', label='measurements'
        )
    plt.xlabel(r"$k\ \  [h/\mathrm{{Mpc}}]$")
    plt.ylabel(r"$P(k)\ \ [(\mathrm{{Mpc}}/h)^3]$")
    plt.legend()
    plt.title(",  ".join([
        r"$f_\mathrm{{NL}} = {}$".format(NG),
        "mask={}".format(mask_tag), "selection={}".format(selection_tag)
    ]))

    return ratios


REDSHIFT = 1.
NG = 0

if __name__ == '__main__':

    mask_tag = "random0_BOSS_DR12v5_CMASS_North"
    selection_tag = "[100.0,500.0]"
    source_tags = list(product(("NG={}.".format(NG),), range(1, 25)))

    # Set I/O paths.
    ## Map data.
    map_data_dir = data_dir/"raw"/"catalogue_maps"

    map_data_file = "catalogue-map-({}).npz".format(
        ",".join([
            "source=halo-({},z=1.)-{}",
            "map=cartesian", "scale=[None,0.1]", "orders=[0]", "rsd=False",
            "mask={}".format(mask_tag), "selection={}".format(selection_tag)
        ])
    )

    ## Survey products.
    survey_product_dir = data_dir/"processed"/"survey_products"

    mask_or_file_info = ",".join([
        "orders=[0,2,4,6,8]", "boxsize=1000.0", "expansion=70.0", "mesh=768",
        "mask={}".format(mask_tag), "selection={}".format(selection_tag)
    ])

    mask_file = "mask-({}).npy".format(mask_or_file_info)

    window_file = "window-({}).npz".format(mask_or_file_info)

    covariance_estimate_info = ",".join([
        "source=1-2500", "map=cartesian", "scale=[None,0.1]", "orders=[0]",
        "mask={}".format(mask_tag), "selection={}".format(selection_tag)
    ])

    covariance_estimate_file = "covar-estimate-({}).npz".format(
        covariance_estimate_info
    )

    ## Cosmological inputs.
    cosmo_dir = data_dir/"external"/"cosmology"

    cosmo_file = "simulation.txt"

    ## Outputs.
    output_dir = data_dir/"processed"/"survey_validation"

    # Validate Cartesian modelling.
    confirm_directory(output_dir)

    k, pk = collate_map_data()
    ratios = compare_with_cartesian_model(k, pk)
