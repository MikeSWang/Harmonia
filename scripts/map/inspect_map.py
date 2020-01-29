"""Inspect statistical properties of hybrid maps.

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
        f"halos-(NG={NG},z=1.)-(map=hybrid,"
        f"fsky={FSKY},knots=[{KHYB},{KMAX}],orders={ORDERS},rsd={RSD}"
        ").npy"
    )

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
                )
                cmap_vector = CartesianArray(
                    cmap_data[serial_index], 'k', 'power_'
                )
                hmap_vector = np.concatenate(
                    (
                        smap_vector.unfold(SPIVOT, return_only='data'),
                        cmap_vector.unfold(CPIVOT, return_only='data')
                    )
                )
                hybrid_map_data.append(hmap_vector)
        safe_save(hybrid_data, PATHOUT/"hybrid_map", output_filename)
    else:
        hybrid_map_data = np.load(PATHOUT/"hybrid_map"/output_filename)

    return hybrid_map_data


def inspect_hybrid_map():
    """Inspect statistical properties of hybrid map data.

    Returns
    -------
    sample_corr : :class:`numpy.ndarray`
        Hybrid map data sample correlation matrix.

    """
    sample_cov = np.cov(hybrid_data, rowvar=False)
    sample_corr = covar_to_corr(sample_cov).real

    plt.style.use(harmony)
    sns.set(style='ticks', font='serif')
    sns.heatmap(sample_corr, square=True, cmap='YlGn')

    return sample_corr


COLLATE_DATA = False

NG = "0."
FSKY = "1.00"
KHYB, KMAX = 0.04, 0.1
ORDERS = "[0]"
SPIVOT, CPIVOT = 'spectral', 'variable'
RSD = False

if __name__ == '__main__':
    hybrid_data = get_hybrid_maps()
    hybrid_corr = inspect_hybrid_map()
