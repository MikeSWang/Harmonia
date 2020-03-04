"""Generate correlated random maps.

"""
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from map_rc import PATHOUT
from harmonia.algorithms import CartesianArray, SphericalArray
from harmonia.collections import harmony
from harmonia.collections import \
    covar_to_corr, overwrite_protection, sort_dict_to_list

CATALOGUE_SERIAL_NUM = list(chain(range(0, 11), range(12, 22)))
CATALOGUE_SERIAL_IND = ["L.txt", "R.txt"]


def safe_save(data, path, file):
    """Safely save data by checking overwrite protections.

    Parameters
    ----------
    data : array_like
        Data to be saved.
    path, file : str or :class:`pathlib.Path`
        Path and file name for the data to be saved.

    """
    if overwrite_protection(path, file):
        try:
            np.save(path/file, data)
        except TypeError:
            np.save(path + file, data)


def get_hybrid_maps():
    """Load and process hybrid map data files.

    Returns
    -------
    hybrid_map_data : list of :class:`numpy.ndarray`
        Hybrid map data vectors.

    """
    output_filename = (
        "halos-(NG={},z=1.)-(map=hybrid,fsky={},"
        "pivots=[{},{}],knots=[{},{}],orders={},rsd={}).npy"
    ).format(NG, FSKY, SPIVOT, CPIVOT, KHYB, KMAX, ORDERS, RSD)

    catalogue_root = "halos-(NG={},z=1.)-{}-"
    smap_suffix = "(map=spherical,fsky={},knots=[{},{}],rsd={}).npy"
    cmap_suffix = "(map=cartesian,fsky={},knots=[{},{}],orders={},rsd={}).npy"

    if COLLATE_DATA:
        hybrid_map_data = []
        for serial_num in CATALOGUE_SERIAL_NUM:
            catalogue_prefix = catalogue_root.format(NG, serial_num)

            smap_file = catalogue_prefix \
                + smap_suffix.format(FSKY, None, KHYB, RSD)
            cmap_file = catalogue_prefix \
                + cmap_suffix.format(FSKY, KHYB, KMAX, ORDERS, RSD)

            smap_data = np.load(PATHOUT/"spherical_map"/smap_file).item()
            cmap_data = np.load(PATHOUT/"cartesian_map"/cmap_file).item()

            for serial_index in CATALOGUE_SERIAL_IND:
                smap_vector = SphericalArray.build(
                    filling=sort_dict_to_list(smap_data[serial_index])
                ).unfold(SPIVOT, return_only='data')

                cmap_vector = CartesianArray(
                    cmap_data[serial_index], 'k', 'power_'
                ).unfold(CPIVOT, return_only='data')

                hybrid_map_data.append(
                    np.concatenate((smap_vector, cmap_vector))
                )
        safe_save(hybrid_map_data, PATHOUT/"hybrid_map", output_filename)
    else:
        hybrid_map_data = np.load(PATHOUT/"hybrid_map"/output_filename)

    return hybrid_map_data


def inspect_hybrid_map(thredshold=0., savefig=False, zoom=False):
    """Inspect statistical properties of hybrid map data.

    Returns
    -------
    sample_corr : :class:`numpy.ndarray`
        Hybrid map data sample correlation matrix.

    """
    mean_data = np.average(hybrid_data, axis=0)
    sample_cov = np.conj(hybrid_data - mean_data).T \
        @ np.asarray(hybrid_data - mean_data) \
        / len(hybrid_data)
    sample_corr = covar_to_corr(sample_cov)

    reflection_mask = np.triu_indices(len(sample_cov), k=1)
    view_corr = sample_corr.real
    view_corr[reflection_mask] = sample_corr.imag[reflection_mask]

    output_filename = "hybrid_corr-(fsky={},pivots=[{},{}],rsd={}).pdf"\
        .format(FSKY, SPIVOT, CPIVOT, RSD)

    plt.close('all')
    plt.style.use(harmony)
    sns.set(style='ticks', font='serif')

    view_offcorr = view_corr[zoom:, :zoom]  # real part only

    view_corr[np.abs(view_corr) < thredshold] = 0.
    sns.heatmap(
        (view_corr), square=True,
        cmap='coolwarm', rasterized=True
    )
    if savefig:
        plt.savefig(
            PATHOUT/"hybrid_map"/output_filename.replace("_corr", "_corr"),
            format='pdf', transparency=True
        )
    if zoom:
        plt.figure()
        sns.heatmap((view_offcorr), cmap='coolwarm')
        if savefig:
            plt.savefig(
                PATHOUT/"hybrid_map"/(
                    output_filename.replace("_corr", "_offcorr")
                ),
                format='pdf', transparency=True
            )

    return sample_corr


COLLATE_DATA = True

NG = "0."
FSKY = "1.00"
KHYB, KMAX = 0.04, 0.1
ORDERS = "[0]"
SPIVOT, CPIVOT = 'natural', 'variable'
RSD = False

if __name__ == '__main__':
    hybrid_data = get_hybrid_maps()
    hybrid_corr = inspect_hybrid_map(thredshold=0., savefig=True, zoom=456)
