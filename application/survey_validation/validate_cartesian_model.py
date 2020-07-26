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

    mask_multipoles = np.load(survey_product_dir/mask_file, allow_pickle=True)

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
        wavenumbers, redshift=1., cosmo=simulation_cosmo, growth_rate=0.,
        mask_multipoles=mask_multipoles, window_multipoles=window_multipoles
    )

    null_cartesian_model = CartesianMultipoles(
        wavenumbers, redshift=1., cosmo=simulation_cosmo, growth_rate=0.
    )

    orders = multipole_data.keys()

    multipole_model = {
        order: cartesian_model.convolved_power_multipoles(
            orders=[order], b_1=BIAS, f_nl=NG, nbar=DENSITY, contrast=CONTRAST
        ).array['power']
        for order in orders
    }

    multipole_null_model = {
        order: null_cartesian_model.convolved_power_multipoles(
            orders=[order], b_1=BIAS, f_nl=NG, nbar=DENSITY, contrast=CONTRAST
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
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG)
    ]))
    plt.subplots_adjust(hspace=0)
    plt.savefig(output_file.with_suffix('.pdf'))

    return _ratios


if __name__ == '__main__':

    TRACER = "halos"
    SERIES = '-cut_2'
    NG = 0
    MAP_SERIALS = range(1, 1+24)
    BIAS = 3.55  # 3.50
    DENSITY = 4.75e-5  # 4.91e-5
    CONTRAST = 50.

    MASK_TAG = "1.0"  # "random0_BOSS_DR12v5_CMASS_North"  #
    SELECTION_TAG = "None"  # "[100.0,500.0]"  #

    SCALE_TAG = "[0.04,0.09]"
    ORDER_TAG = "[0]"
    RSD_TAG = "False"

    source_tags = list(product(
        [TRACER,], ["NG={}.".format(NG),], [SERIES,], range(1, 25)
    ))

    # Set I/O paths.
    ## Map data.
    map_data_dir = data_dir/"raw"/"catalogue_maps"
    map_data_file = "catalogue-map-({}).npz".format(",".join([
        "source={}-({},z=1.){}-{}",
        "map=cartesian",
        "scale={}".format(SCALE_TAG),
        "orders={}".format(ORDER_TAG),
        "rsd={}".format(RSD_TAG),
        "mask={}".format(MASK_TAG),
        "selection={}".format(SELECTION_TAG),
    ]))

    ## Survey products.
    survey_product_dir = data_dir/"processed"/"survey_products"

    mask_or_file_info = ",".join([
        "orders=[0,2,4,6,8]", "boxsize=1000.0", "expansion=70.0", "mesh=768",
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
    ])
    mask_file = "mask-({}).npy".format(mask_or_file_info)

    window_file = "window-({}).npz".format(mask_or_file_info)

    covariance_estimate_info = ",".join([
        "source=1-2500", "map=cartesian", "boxsize=1000.0",
        "scale={}".format(SCALE_TAG), "orders={}".format(ORDER_TAG),
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
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
        "scale={}".format(SCALE_TAG), "rsd={}".format(RSD_TAG),
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
    ]))

    # Validate Cartesian modelling.
    confirm_directory(output_dir)

    k, multipoles = collate_map_data()
    ratios = compare_with_cartesian_model(k, multipoles)
    print("Measurement-to-model relative difference for orders:", ratios)
