"""Export recovered 2-point function values.

"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from agreement_rc import PATHIN, PATHOUT
from view_twopt import view_covariance
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    collate,
    confirm_directory_path as confirm_dir,
    overwrite_protection,
    sort_dict_to_list,
)
from harmonia.cosmology import fiducial_distance

DATA_NAME_ROOT = "measure_twopt"
MODEL_NAME_ROOT = "predict_twopt"
NBODY_NAME_ROOT = "nbodymod_twopt"
NBODY_REFNAME_ROOT = "halos"


def aggregate_data(output):
    """Aggregate data into a single result.

    Parameters
    ----------
    output : dict
        Output data to be aggregated.

    Returns
    -------
    result : dict
        Aggregated output data.

    """
    result = {var: np.average(vals, axis=0) for var, vals in output.items()}

    return result


def setup_cosmology(boxsize, zmax=None):
    """Set up cosmology.

    """
    global disc, index_vector, k_ordered_normalisations

    if boxsize is None:
        boxsize = fiducial_distance(zmax)
    disc = DiscreteSpectrum(boxsize, 'Dirichlet', KMAX)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        index_vector = SphericalArray.build(disc=disc)\
            .unfold(PIVOT, return_only='index')

    flat_order = np.concatenate(sort_dict_to_list(disc.wavenumbers)).argsort()

    k_ordered_normalisations = np.concatenate(
        sort_dict_to_list(disc.normalisations)
    )[flat_order]


def main(collate_data=False, load_data=False, load_model=False, load_nbody=False,
         save_fig=False):
    """Collate data, load data, model and n-body result files and/or export
    two-point plots.

    Parameters
    ----------
    load_data, load_model, load_nbody, savefig : bool, optional
        If `True` (default is `False`), load data, load model, load n-body
        result files or save the plotted figure.

    Raises
    ------
    AssertionError
        If output path does not exist.
    AssertionError
        If overwrite permission is denied at the output path.

    """
    _SAVE = 'agg'  # 'full', 'agg', ''

    data_outpath = f"{PATHOUT}{DATA_NAME_ROOT}/"
    model_outpath = f"{PATHOUT}{MODEL_NAME_ROOT}/"
    nbody_outpath = f"{PATHOUT}{NBODY_NAME_ROOT}/"
    nbody_inpath = f"{PATHIN}{NBODY_NAME_ROOT}/"
    collate_path = data_outpath + "collated/"

    if collate_data:
        output, count, _ = \
            collate(f"{data_outpath}{DATA_NAME_ROOT}-*.npy", 'npy')
        data = aggregate_data(output)

        save_str = "".join(_.split("(")[-1].split(")")[:-1]) + f"*{count}"
        confirm_dir(collate_path)
        if _SAVE == 'full':
            save_name = f"{DATA_NAME_ROOT}-({save_str})-all.npy"
            assert overwrite_protection(collate_path, save_name)
            np.save(collate_path+save_name, output)
        elif _SAVE == 'agg':
            save_name = f"{DATA_NAME_ROOT}-({save_str})-agg.npy"
            assert overwrite_protection(collate_path, save_name)
            np.save(collate_path+save_name, data)

    if load_data:
        results = np.load(
            f"{collate_path}{DATA_NAME_ROOT}{DATA_TAG}.npy"
        ).item()
        if DATA_TAG.endswith("agg"):
            data = results
        elif DATA_TAG.endswith("all"):
            data = aggregate_data(results)

    if load_model:
        model = np.load(
            f"{model_outpath}{MODEL_NAME_ROOT}{MODEL_TAG}.npy"
        ).item()[PIVOT]

    if load_nbody:
        nbody_model = np.load(
            f"{nbody_outpath}{NBODY_NAME_ROOT}{NBODY_TAG}.npy"
        ).item()[PIVOT]
        nbody_power = aggregate_data(
            np.load(
                f"{nbody_inpath}/{NBODY_REFNAME_ROOT}{NBODY_REFTAG}.npy"
            ).item()
        )

    if load_data or collate_data:
        data_covar = data[PIVOT]
    if load_model:
        model_covar = model['signal'] + model['shotnoise']
    if load_nbody:
        model_covar = nbody_model['signal'] + nbody_model['shotnoise']

        data_covar = np.zeros(disc.nmode)
        for vec_idx, triplet_idx in enumerate(index_vector):
            condition = np.logical_and(
                nbody_power['ln'][:, 0] == triplet_idx[0],
                nbody_power['ln'][:, 1] == triplet_idx[-1]
            )
            ref_idx = np.where(condition)[0][0]
            data_covar[vec_idx] = nbody_power['Pln'][ref_idx] / k_ordered_normalisations[ref_idx]
        data_covar = np.diag(data_covar)

    global data_2pt, model_2pt

    if load_data or load_nbody or collate:
        data_2pt = np.abs(data_covar)
    if load_model or load_nbody:
        model_2pt = np.abs(model_covar)
    else:
        model_2pt = None

    global view

    index_range = slice(len(index_vector))  # len(index_vector), None
    view = view_covariance(
        data_2pt,
        model=model_2pt,
        ratio=RATIO,
        diag='only',  # False, 'only', 'off'
        select_idx=index_range,
        tick_labels='auto'  # 'auto', index_vector[index_range]
    )
    if save_fig:
        plt.savefig(
            f"{PATHOUT}twopt-"
            f"(pivot={PIVOT},beta={BETA},kmax={KMAX},diff={str(RATIO)}).png",
            format='png',
            dpi=500,
            transparent=False
        )


if __name__ == '__main__':

    PIVOT = 'natural'
    BETA = 'none'
    KMAX = 0.1
    ZMAX = 0.05
    BOXSIZE = None  # 1000.

    DATA_TAG = ""
    MODEL_TAG = f"-(pivots=natural,nbar=0.001,bias=2.,beta=none,rmax=148.,kmax=0.1)"
    NBODY_TAG = ""
    NBODY_REFTAG = (
        "-(NG=0.,z=1.)-"
        "(nbar=2.49e-4,bias=2.3415,kmax=0.04,boxsize=1000.,mesh=c256,npair=11)"
        )

    EXPORT_OPT = dict(
        collate_data=True,
        load_data=False,
        load_model=True,
        load_nbody=False,
        save_fig=False,
    )
    RATIO = False  # False, 'd2m', 'm2d'

    setup_cosmology(BOXSIZE, zmax=ZMAX)
    main(**EXPORT_OPT)
