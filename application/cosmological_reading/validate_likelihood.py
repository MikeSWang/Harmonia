"""Validate spherical likelihood evaluations on a coarse grid.

"""
import os
import sys

import numpy as np
import seaborn as sns

try:
    from application import data_dir
    from harmonia.algorithms import SphericalArray
    from harmonia.cosmology import BaseModel
    from harmonia.reader import Couplings, LogLikelihood, SphericalCorrelator
    from harmonia.surveyor.synthesis import generate_compression_matrix
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import data_dir
    from harmonia.algorithms import SphericalArray
    from harmonia.cosmology import BaseModel
    from harmonia.reader import Couplings, LogLikelihood, SphericalCorrelator
    from harmonia.surveyor.synthesis import generate_compression_matrix


def load_map_data(map_serial_num):
    """Load spherical map data used for likelihood evaluation.

    Parameters
    ----------
    map_serial_num : int
        The serial number of the map to use.

    Returns
    -------
    :class:`harmonia.algorithms.arrays.SphericalArray`
        Spherical map data.

    """
    map_dir = data_dir/"raw"/"catalogue_maps"

    return SphericalArray.load(
        map_dir/"catalogue-map-({}).npz".format(",".join([
            f"source=halos-(NG={NG}.,z=1.){SERIES}-{map_serial_num}",
            "map=spherical", "scale=[None,0.04]", "orders=None", "rsd=False",
            f"mask={MASK_TAG}", f"selection={SELECTION_TAG}"
        ]))
    )


def set_base_model(disc):
    """Set the base cosmological model used for likelihood evaluation.

    Parameters
    ----------
    disc : :class:`harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum object.

    Returns
    -------
    :class:`harmonia.reader.models.SphericalCorrelator`
        Spherical correlator model.

    """
    cosmo_dir = data_dir/"external"/"cosmology"
    cosmo = BaseModel(cosmo_dir/"simulation-GadgetAHF.txt")

    product_dir = data_dir/"processed"/"survey_products"
    couplings = Couplings.load(
        product_dir/(
            "couplings-(rmax=500.0,kmax={},mask={},selection={}).npz"
            .format(KCUT, MASK_TAG, SELECTION_TAG)
        )
    )

    return SphericalCorrelator(
        disc, redshift=1., growth_rate=0., couplings=couplings, cosmo=cosmo
    )


def likelihood_evaluation(bias, png, map_data, map_model, discard=None):
    """Evaluate the map likelihood on a 9-grid.

    Parameters
    ----------
    bias, png : list of float
        Bias and PNG parameters.
    map_data : :class:`harmonia.algorithms.arrays.SphericalArray`
        Spherical map data.
    map_model : :class:`harmonia.reader.models.SphericalCorrelator`
        Spherical correlator model.
    discard : int or None, optional
        If not `None`, the least informative `discard` number of modes
        are discarded by data compression.

    Returns
    -------
    likelihood : float :class:`numpy.ndarray`
        Log-likelihood evaluated on a grid.

    """
    log_likelihood = LogLikelihood(
        spherical_data=map_data, base_spherical_model=map_model,
        spherical_pivot=PIVOT, nbar=NBAR, contrast=CONTRAST
    )

    diagonal = (MASK_TAG is None or MASK_TAG.startswith('1.')) \
        and (SELECTION_TAG is None or SELECTION_TAG.upper() == 'NONE')

    if diagonal:
        comp_mat = None
    else:
        comp_mat = generate_compression_matrix(
            {
                'pivot': PIVOT, 'spherical_model': map_model,
                'b_1': np.median(bias), 'f_nl': np.median(png)
            },
            {
                'pivot': PIVOT, 'spherical_model': map_model,
                'b_1': np.max(bias), 'f_nl': np.max(png)
            },
            discard=discard
        )

    likelihood = np.reshape(
        [
            log_likelihood.spherical_map_likelihood(
                b_1=b_1, f_nl=f_nl,
                diagonal=diagonal, compression_matrix=comp_mat
            )
            for b_1 in bias for f_nl in png
        ],
        (len(bias), len(png))
    )

    return np.real_if_close(likelihood, tol=10**10)


PIVOT = 'natural'
SERIES = '-cut_2'

if __name__ == '__main__':

    MAP_SERIAL_NUM = 1

    NG = 0
    NBAR, CONTRAST = 4.75e-5, 50.  # 4.91e-5
    KCUT = 0.04
    MASK_TAG = "1.0"  # "random0_BOSS_DR12v5_CMASS_North"
    SELECTION_TAG = "None"  # "[100.0,500.0]"

    spherical_data = load_map_data(MAP_SERIAL_NUM)
    spherical_model = set_base_model(spherical_data.disc)

    bias_grid = np.flip([3., 4., 3.])
    png_grid = [NG - 100., NG, NG + 100.]

    likelihood_grid = likelihood_evaluation(
        bias_grid, png_grid, spherical_data, spherical_model,
        # discard=376
    )

    sns.heatmap(
        likelihood_grid,
        xticklabels=png_grid, yticklabels=bias_grid, annot=True
    )
