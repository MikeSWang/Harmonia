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
    float :class:`numpy.ndarray`
        Wavenumber of Cartesian map multipoles.
    dict{int: float :class:`numpy.ndarray`}
        Power multipoles of different Legendre orders at `wavenumbers`.

    """
    _wavenumbers = []
    _multipoles = defaultdict(list)
    for source_tag in source_tags:
        map_data = CartesianArray.load(
            map_data_dir/map_data_file.format(*source_tag)
        )

        orders = np.unique(map_data.array['order'])
        _wavenumbers.append(np.unique(map_data.array['wavenumber']))
        for ell in orders:
            power = map_data.array['power'][map_data.array['order'] == ell]
            _multipoles[ell].append(power)

    _wavenumbers = np.mean(_wavenumbers, axis=0)
    _multipoles = {
        ell: np.mean(_multipoles[ell], axis=0) for ell in _multipoles.keys()
    }

    return _wavenumbers, _multipoles


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
    dict{int: float :class:`numpy.ndarray`}
        Ratio of the measured multipoles to model predictions.

    """
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

    null_cartesian_model = CartesianMultipoles(
        wavenumbers, redshift=REDSHIFT, cosmo=simulation_cosmo, growth_rate=0.
    )

    orders = multipole_data.keys()

    multipole_model = {
        order: cartesian_model.convolved_power_multipoles(
            orders=[order], b_1=BIAS, f_nl=NG, nbar=2.5e-4, contrast=10.
        ).array['power']
        for order in orders
    }

    multipole_null_model = {
        order: null_cartesian_model.convolved_power_multipoles(
            orders=[order], b_1=BIAS, f_nl=NG, nbar=2.5e-4, contrast=10.
        ).array['power']
        for order in orders
    }

    _ratios = {
        order: multipole_data[order] / multipole_model[order] - 1
        for order in orders
    }

    output_file = output_dir/output_filename

    sns.set(style='ticks', font='serif')
    plt.figure()
    subplots = []
    for ord_idx, order in enumerate(orders):
        sharex = None if ord_idx == 0 else subplots[0]
        ax = plt.subplot2grid((len(orders), 1), (ord_idx, 0), sharex=sharex)
        if ord_idx == 0:
            subplots.insert(0, ax)
        else:
            subplots.append(ax)

        model_label = 'convolved model' if ord_idx == 0 else None
        null_model_label = 'underlying model' if ord_idx == 0 else None
        plt.loglog(
            wavenumbers, multipole_model[order],
            ls='--', label=model_label
        )
        plt.loglog(
            wavenumbers, multipole_null_model[order],
            ls=':', label=null_model_label
        )

        data_label = 'measurements' if ord_idx == 0 else None
        plt.errorbar(
            wavenumbers, multipole_data[order],
            yerr=std_estimate*multipole_data[order]/np.sqrt(len(source_tags)),
            marker='s', label=data_label
        )

        if ord_idx == len(orders) - 1:
            plt.xlabel(r"$k\ \  [h/\mathrm{{Mpc}}]$")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        plt.ylabel(r"$P_{}(k)\ \ [(\mathrm{{Mpc}}/h)^3]$".format(order))

        if ord_idx == 0:
            plt.legend()

    plt.suptitle(",  ".join([
        r"$f_\mathrm{{NL}} = {}$".format(NG),
        "mask={}".format(mask_tag), "selection={}".format(selection_tag)
    ]))
    plt.subplots_adjust(hspace=0)
    plt.savefig(output_file.with_suffix('.pdf'))

    return _ratios


REDSHIFT = 1.
NG = 0
BIAS = 2.324  # 2.354

if __name__ == '__main__':

    mask_tag = "random0_BOSS_DR12v5_CMASS_North"
    selection_tag = "[100.0,500.0]"

    scale_tag = "[None,0.1]"
    order_tag = "[0]"
    rsd_tag = "False"

    source_tags = list(product(("NG={}.".format(NG),), range(1, 25)))

    # Set I/O paths.
    ## Map data.
    map_data_dir = data_dir/"raw"/"catalogue_maps"
    map_data_file = "catalogue-map-({}).npz".format(",".join([
        "source=halos-({},z=1.)-standard-{}",
        "map=cartesian",
        "scale={}".format(scale_tag),
        "orders={}".format(order_tag),
        "rsd={}".format(rsd_tag),
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    ## Survey products.
    survey_product_dir = data_dir/"processed"/"survey_products"

    mask_or_file_info = ",".join([
        "orders=[0,2,4,6,8]", "boxsize=1000.0", "expansion=70.0", "mesh=768",
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ])
    mask_file = "mask-({}).npy".format(mask_or_file_info)

    window_file = "window-({}).npz".format(mask_or_file_info)

    covariance_estimate_info = ",".join([
        "source=1-2500", "map=cartesian", "boxsize=1000.0",
        "scale={}".format(scale_tag),
        "orders={}".format(order_tag),
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ])
    covariance_estimate_file = "covar-estimate-({}).npz".format(
        covariance_estimate_info
    )

    ## Cosmological inputs.
    cosmo_dir = data_dir/"external"/"cosmology"
    cosmo_file = "simulation-GadgetAHF.txt"

    ## Outputs.
    output_dir = data_dir/"raw"/"survey_validation"
    output_filename = "cartesian-model-validation-({})".format(",".join([
        "NG={}.".format(NG),
        "scale={}".format(scale_tag),
        "rsd={}".format(rsd_tag),
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    # Validate Cartesian modelling.
    confirm_directory(output_dir)

    k, multipoles = collate_map_data()
    ratios = compare_with_cartesian_model(k, multipoles)
    print("Measurement-to-model relative difference for orders:", ratios)
