"""Smooth diagonal 2-point function values for *n*-body simulation pairs.

"""
import warnings

import numpy as np
from matplotlib import pyplot as plt

from agreement_rc import PATHIN, PATHOUT, aggregate
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import harmony, sort_dict_to_list


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


def view_result(result):
    """Visualise output result.

    """
    plt.style.use(harmony)
    plt.close('all')

    plt.subplot2grid((4, 8), (0, 0), rowspan=3, colspan=8)

    plt.loglog(
        bincoord, smooth_data['measurements'], ls='-', marker='+',
        label='measurements'
        )
    plt.loglog(
        bincoord, smooth_data['predictions'], ':', marker='+',
        label=f'{corrct_tag} predictions'
        )
    xlim = plt.gca().get_xlim()

    plt.tick_params(axis='x', which='both', labelbottom=False)
    plt.ylabel(
        r'$\kappa \left\langle\delta \delta^*\right\rangle$ [$(\textrm{Mpc}/h)^2$]'
        )
    plt.legend()

    plt.subplot2grid((4, 8), (3, 0), rowspan=1, colspan=8)

    plt.plot(
        bincoord, smooth_data['measurements']/smooth_data['predictions']-1, '--'
        )

    yceil = 0.01
    plt.fill_between(xlim, [yceil,]*2, [-yceil]*2, alpha=0.2)
    plt.axhline(y=0, lw=1, ls='--')

    plt.xscale('log')
    plt.xlim(xlim)
    plt.ylim(bottom=-0.05,top=0.05)
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(
        r'$\langle\delta\delta^*\rangle_\mathrm{{dat}} \big/$'  # \big(
        + r'{}$\langle\delta\delta^*\rangle_\mathrm{{crt}}$'.format(corrct_tag)
        + r'$-1$'  # \big)
        )

    plt.subplots_adjust(wspace=0, hspace=0)


def setup_cosmology():
    """Set up cosmology.

    """
    global index_vector, norms

    disc = DiscreteSpectrum(BOXSIZE, 'Dirichlet', KMAX)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        index_vector = SphericalArray.build(disc=disc)\
            .unfold(PIVOT, return_only='index')
    flat_order = np.concatenate(sort_dict_to_list(disc.wavenumbers)).argsort()
    norms = np.concatenate(sort_dict_to_list(disc.normalisations))[flat_order]


def main():

    model = np.load(
        f"{PATHOUT}{SCRIPT_NAME}/{SCRIPT_NAME}{MODEL_TAG}.npy"
    ).item()
    reference = aggregate(
        np.load(f"{PATHIN}{SCRIPT_NAME}/{REFERENCE_FILE}.npy").item()
    )

# Set up indexing.

kappa = np.zeros(DISC.mode_count)
coord = np.zeros(DISC.mode_count)
sp2pt = np.zeros(DISC.mode_count)
for idx, greek in enumerate(indx_vec):
    condition = np.logical_and(
        ref['ln'][:,0] == greek[0],
        ref['ln'][:,1] == greek[-1]
        )
    refidx = np.where(condition)[0][0]
    kappa[idx] = norms[refidx]
    coord[idx] = ref['kln'][refidx]
    sp2pt[idx] = (ref['Pln'][refidx] - ref['Pshot']) / norms[refidx]

# Post-process imported data.
Covar = np.abs(sp2pt)
covar = np.abs(np.diag(REBIAS2**2*mod['signal']))

rough_data = {
    'measurements': kappa * Covar,
    'predictions': kappa * covar,
    }
smooth_data = {}
bins = np.concatenate([
    np.array([k_ell[0] for k_ell in DISC.wavenumbers])[[0, 4, 7, 9, 12,]],
    [0.04,]
    ])
# bins = np.linspace(6e-3, 4e-2, 6), logspace(-2.2, -1.4, 5)

counts, _ = np.histogram(coord, bins=bins)
bincoord = np.histogram(coord, bins=bins, weights=coord)[0] / counts
for key, val in rough_data.items():
    bindat, _ = np.histogram(coord, bins=bins, weights=val)  # aggregate values
    smooth_data[key] = bindat / counts

ratio = np.average(
    smooth_data['measurements'] / smooth_data['predictions']
    )
if np.isclose(ratio, 1, atol=1.e-2):
    corrct_tag = ''
else:
    corrct_tag = r'${:.2f} \times$'.format(ratio)
    smooth_data['measurements'] = smooth_data['measurements'] / ratio
    warnings.warn(
        "2-point measurements downscaled by {:g}. ".format(ratio),
        RuntimeWarning
        )

# Visualise data product.


if __name__ == '__main__':

    REFERENCE_FILE = (
        "halos-(NG=0.,z=1.)"
        "-(nbar=2.49e-4,bias=2.3415,kmax=0.04,boxsize=1000.,mesh=c256,pair=11)"
    )
    SCRIPT_NAME = "nbodymod_twopt"
    MODEL_TAG = (
        "-(NG=0.,z=1.)-"
        "(nbar=2.49e-4,b=2.3415,beta=none,kmax=0.04,side=1000.,nmesh=256,npair=11)"
        )

    PIVOT = 'natural'
    BETA = 'none'  # float
    KMAX = 0.04
    BOXSIZE = 500.
    UPSCALE_BIAS = 1.027

    main()
