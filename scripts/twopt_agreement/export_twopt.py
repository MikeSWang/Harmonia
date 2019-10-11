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
    harmony,
    overwrite_protection,
    sort_dict_to_list,
)
from harmonia.cosmology import fiducial_distance

DATA_NAME_ROOT = "measure_twopt"
MODEL_NAME_ROOT = "predict_twopt"
NBODY_NAME_ROOT = "nbodymod_twopt"
NBODY_REF_NAME_ROOT = "halos"


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
    global disc, index_vector, normalisations

    if boxsize is None:
        boxsize = 2*fiducial_distance(zmax)
    disc = DiscreteSpectrum(boxsize/2, 'Dirichlet', KMAX)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        index_vector = SphericalArray.build(disc=disc)\
            .unfold(PIVOT, return_only='index')

    flat_order = np.concatenate(sort_dict_to_list(disc.wavenumbers)).argsort()

    normalisations = np.concatenate(
        sort_dict_to_list(disc.normalisations)
    )[flat_order]


def process_data(collate_data=False, load_data=False, load_model=False,
                 load_nbody=False):
    """Collate data, load data, model and n-body result files and/or export
    two-point plots.

    Parameters
    ----------
    collate_data, load_data, load_model, load_nbody : bool, optional
        If `True` (default is `False`), collate data, load data,
        load model, or load n-body result files.

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
        output, count, _ = collate(
            f"{data_outpath}{DATA_NAME_ROOT}-{DATA_SEARCH_TAG}.npy",
            'npy'
        )
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
        ).item()

    if load_nbody:
        nbody_model = np.load(
            f"{nbody_outpath}{NBODY_NAME_ROOT}{NBODY_TAG}.npy"
        ).item()[PIVOT]
        nbody_power = aggregate_data(
            np.load(
                f"{nbody_inpath}/{NBODY_REF_NAME_ROOT}{NBODY_REF_TAG}.npy"
            ).item()
        )

    if load_data or collate_data:
        data_covar = data[PIVOT]
    if load_model:
        model = model[PIVOT]
        model_covar = model['signal'] + model['shotnoise']
    if load_nbody:
        nbody_model = nbody_model[PIVOT]
        model_covar = nbody_model['signal'] + nbody_model['shotnoise']

        data_covar = np.zeros(len(index_vector))
        for vec_idx, triplet_idx in enumerate(index_vector):
            condition = np.logical_and(
                nbody_power['ln'][:, 0] == triplet_idx[0],
                nbody_power['ln'][:, 1] == triplet_idx[-1]
            )
            ref_idx = np.where(condition)[0][0]
            data_covar[vec_idx] = nbody_power['Pln'][ref_idx] \
                / normalisations[ref_idx]
        data_covar = np.diag(data_covar)

    global k, k_normalisations

    k = np.zeros(len(index_vector))
    k_normalisations = np.zeros(len(index_vector))
    for vec_idx, triplet_idx in enumerate(index_vector):
        k[vec_idx] = sort_dict_to_list(disc.wavenumbers)\
            [triplet_idx[0]][triplet_idx[-1]-1]
        k_normalisations[vec_idx] = sort_dict_to_list(disc.normalisations)\
            [triplet_idx[0]][triplet_idx[-1]-1]

    global data_2pt, model_2pt

    if load_data or load_nbody or collate:
        data_2pt = np.abs(data_covar)
    if load_model or load_nbody:
        model_2pt = np.abs(model_covar)
    else:
        model_2pt = None


def view_data():
    """Visualise output data.

    """
    plt.close('all')
    plt.style.use(harmony)
    view_covariance(
        data_2pt,
        model=model_2pt,
        ratio=RATIO,
        diag=DIAG,
        select_idx=INDEX_RANGE,
        tick_labels=TICK_LABELS
    )
    if SAVE_FIG:
        plt.savefig(
            f"{PATHOUT}twopt-"
            f"(pivot={PIVOT},f0={BETA},kmax={KMAX},diff={str(RATIO)}).png",
            format='png',
            dpi=500,
            transparent=False
        )


if __name__ == '__main__':

    PIVOT = 'k'
    BETA = 'none'
    KMAX = 0.1
    ZMAX = 0.05
    BOXSIZE = None  # 1000.

    DATA_SEARCH_TAG = "*nbodykit*"

    GEN_NAME = "nbodykit"
    PIVOT_NAMES = "[natural,k]"
    PARAMS_TAG = "nbar=0.001,b1=2.,f0=none,rmax=148.,kmax=0.1"
    BOX_TAG = "xpd=2.,mesh=gc256,iter=50*100"

    DATA_TAG = "-(gen={},pivots={},{},{})-agg"\
        .format(GEN_NAME, PIVOT_NAMES, PARAMS_TAG, BOX_TAG)
    MODEL_TAG = "-(pivots={},{})".format(PIVOT_NAMES, PARAMS_TAG)
    NBODY_TAG = ""

    NBODY_REF_TAG = "-(NG=0.,z=1.)-(" + \
        "nbar=2.49e-4,b1=2.3415,kmax=0.04," + \
        "boxsize=1000.,mesh=c256,npair=11" + \
        ")"

    setup_cosmology(BOXSIZE, zmax=ZMAX)

    PROCESS_OPTS = dict(
        collate_data=False,
        load_data=True,
        load_model=True,
        load_nbody=False,
    )

    process_data(**PROCESS_OPTS)

    RATIO = 'd2m'  # False, 'd2m', 'm2d'
    DIAG = 'only' # None, 'only', 'off'
    INDEX_RANGE = slice(None, None)  # None, None
    TICK_LABELS = 'auto'  # 'auto', index_vector[index_range]
    SAVE_FIG = False

    view_data()
