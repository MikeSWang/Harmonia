"""Smooth diagonal 2-point function values for *n*-body simulation pairs.

"""
import warnings

import numpy as np
from matplotlib import pyplot as plt

from agreement_rc import PATHIN, PATHOUT
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import harmony, sort_dict_to_list


def _aggregate_data(output):
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


def setup_cosmology():
    """Set up cosmology.

    """
    global disc, index_vector, k_ordered_normalisations

    disc = DiscreteSpectrum(RMAX, 'Dirichlet', KMAX)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        index_vector = SphericalArray.build(disc=disc)\
            .unfold(PIVOT, return_only='index')

    flat_order = np.concatenate(sort_dict_to_list(disc.wavenumbers)).argsort()

    k_ordered_normalisations = np.concatenate(
        sort_dict_to_list(disc.normalisations)
    )[flat_order]


def process_data():

    reference = _aggregate_data(
        np.load(f"{PATHIN}{SCRIPT_NAME}/{REFERENCE_FILE}.npy").item()
    )
    model = np.load(
        f"{PATHOUT}{SCRIPT_NAME}/{SCRIPT_NAME}{MODEL_TAG}.npy"
    ).item()[PIVOT]

    dim_covar = len(index_vector)

    global diagonal_normalisations, diagonal_k_coords, reference_covar, \
        model_covar

    diagonal_normalisations = np.zeros(dim_covar)
    diagonal_k_coords = np.zeros(dim_covar)
    diagonal_covar = np.zeros(dim_covar)
    for vec_idx, triplet_idx in enumerate(index_vector):
        condition = np.logical_and(
            reference['ln'][:, 0] == triplet_idx[0],
            reference['ln'][:, 1] == triplet_idx[-1]
        )
        ref_idx = np.where(condition)[0][0]

        diagonal_normalisations[vec_idx] = k_ordered_normalisations[ref_idx]
        diagonal_k_coords[vec_idx] = reference['kln'][ref_idx]
        this_val = reference['Pln'][ref_idx] \
            / k_ordered_normalisations[ref_idx]
        if np.isinf(this_val):
            print(this_val, vec_idx, ref_idx)
        diagonal_covar[vec_idx] = this_val

    reference_covar = diagonal_covar
    model_covar = np.abs(np.diag(model['signal'] + model['shotnoise']))


def diagonal_smoothing():
    """Smooth diagonal data entries.

    """
    global smooth_data

    rough_data = {
        'measurements': diagonal_normalisations * reference_covar,
        'predictions': diagonal_normalisations * model_covar,
    }
    smooth_data = {}

    spherical_wavenumber_zero_depth = np.array(
        [wavenumbers[0] for deg, wavenumbers in disc.wavenumbers.items()]
    )
    spherical_wavenumber_selected_deg = [0, 4, 7, 9, 12]

    bins = np.concatenate(
        [
            spherical_wavenumber_zero_depth[spherical_wavenumber_selected_deg],
            [0.04],
        ]
    )
    # HACK: manual rebinning.
    bins = [0.00675, 0.016, 0.025, 0.032, 0.036, 0.040]

    global bin_coords, counts

    counts, _ = np.histogram(diagonal_k_coords, bins=bins)
    bin_coords = np.histogram(
        diagonal_k_coords,
        bins=bins,
        weights=diagonal_k_coords
    )[0] / counts
    for key, val in rough_data.items():
        bin_value_total, _ = np.histogram(
            diagonal_k_coords,
            bins=bins,
            weights=val
        )  # aggregate values
        smooth_data[key] = bin_value_total / counts

    global correction_ratio, correction_tag

    correction_ratio = np.average(
        smooth_data['measurements'] / smooth_data['predictions']
    )
    if np.isclose(correction_ratio, 1, atol=1.e-2):
        correction_tag = ''
    else:
        correction_tag = r"${:.2f} \times$".format(correction_ratio)
        smooth_data['measurements'] = \
            smooth_data['measurements'] / correction_ratio
        warnings.warn(
            "2-point measurements downscaled by {:.2f}. "
            .format(correction_ratio),
            RuntimeWarning
        )


def view_result():
    """Visualise output result.

    """
    plt.style.use(harmony)
    plt.close('all')

    main_ax = plt.subplot2grid((4, 8), (0, 0), rowspan=3, colspan=8)

    plt.loglog(
        bin_coords,
        smooth_data['measurements'],
        ls='-',
        marker='+',
        label='measurements'
    )
    plt.loglog(
        bin_coords,
        smooth_data['predictions'],
        ls=':',
        marker='+',
        label=f'{correction_tag} predictions'
    )

    plt.tick_params(axis='x', which='both', labelbottom=False)
    plt.ylabel(
        r'$\kappa \left\langle\delta \delta^*\right\rangle$ '
        r'[$(\textrm{Mpc}/h)^2$]'
        )
    plt.legend()

    xlim = plt.gca().get_xlim()

    plt.subplot2grid((4, 8), (3, 0), rowspan=1, colspan=8, sharex=main_ax)

    plt.plot(
        bin_coords,
        smooth_data['measurements'] / smooth_data['predictions'] - 1,
        '--'
    )

    plt.fill_between(
        xlim,
        [ERROR_PATCH_HT]*2,
        [-ERROR_PATCH_HT]*2,
        alpha=0.2
    )

    plt.axhline(y=0, lw=1, ls='--')
    plt.xlim(xlim)
    plt.ylim(bottom=-ERROR_PANEL_HT,top=ERROR_PANEL_HT)
    plt.xlabel(r"$k$ [$h/\textrm{Mpc}$]")
    plt.ylabel(
        r"$\langle\delta\delta^*\rangle_\mathrm{{data}} \big/$"
        + r"{}".format(correction_tag)  # \big(
        + r"$\langle\delta\delta^*\rangle_\mathrm{{model}}$"  # \big)
        + r"$-1$"
    )

    plt.subplots_adjust(wspace=0, hspace=0)


if __name__ == '__main__':

    SCRIPT_NAME = "nbodymod_twopt"
    MODEL_TAG = \
        "-(pivots=natural,nbar=2.49e-04,b1=2.40,f0=0.,rmax=500.,kmax=0.04)"
    REFERENCE_FILE = (
        "halos-(NG=0.,z=1.)"
        "-(nbar=2.49e-4,b1=2.3415,kmax=0.04,boxsize=1000.,mesh=c256,pair=21)"
    )

    PIVOT = 'natural'
    RMAX = 500.
    KMAX = 0.04

    ERROR_PANEL_HT = 0.05
    ERROR_PATCH_HT = 0.01

    setup_cosmology()
    process_data()  # upscale bias by 1.027
    diagonal_smoothing()
    view_result()
