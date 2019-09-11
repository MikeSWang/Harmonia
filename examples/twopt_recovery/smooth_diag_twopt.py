"""Smooth diagonal 2-point function values.

"""
import warnings

import numpy as np
from matplotlib import pyplot as plt

from twoptrc import PATHIN, PATHOUT, aggregate, clean_warnings
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import harmony

warnings.formatwarning = clean_warnings


# == CONFIGURATION ============================================================

REFDIR = "nbodymod_twopt/"
REFNAME = "halos"
SUBDIR = "nrsd/"
STRUCT = 'k'

BETA = 'none'  # float
KMAX = 0.04

TAG_REF = (
    "-(NG=0.,z=1.)-"
    "(nbar=2.49e-4,b=2.3415,beta=none,kmax=0.04,side=1000.,nmesh=256,npair=11)"
    )


# == OPERATION ================================================================

# Import data.
mod = np.load(
    f"{PATHOUT}{REFDIR}{SUBDIR}nbodymod_2pt-(struct={STRUCT}).npy"
    ).item()
ref = aggregate(np.load(f"{PATHIN}{REFDIR}{REFNAME}{TAG_REF}.npy").item())

# Set up indexing.
DISC = DiscreteSpectrum(500, 'Dirichlet', KMAX)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    indx_vec = SphericalArray.build(disc=DISC).unfold(STRUCT, retonly='index')
order = np.concatenate(DISC.wavenumbers).argsort()
norms = np.concatenate(DISC.normcoeff)[order]

kappa = np.zeros(DISC.nmode)
coord = np.zeros(DISC.nmode)
sp2pt = np.zeros(DISC.nmode)
for idx, greek in enumerate(indx_vec):
    condition = np.logical_and(
        ref['ln'][:,0] == greek[0],
        ref['ln'][:,1] == greek[-1]
        )
    refidx = np.where(condition)[0][0]
    kappa[idx] = norms[refidx]
    coord[idx] = ref['kln'][refidx]
    sp2pt[idx] = ref['Pln'][refidx] / norms[refidx]

# Post-process imported data.
Covar = np.abs(sp2pt)
covar = np.abs(np.diag(mod['signal'] + mod['shotnoise']))

rough_data = {
    'measurements': kappa * Covar,
    'predictions': kappa * covar,
    }
smooth_data = {}
bins = np.linspace(6e-3, 4e-2, 8)  # [k_ell[0] for k_ell in DISC.wavenumbers]

counts, _ = np.histogram(coord, bins=bins)
bincoord = np.histogram(coord, bins=bins, weights=coord)[0] / counts
for key, val in rough_data.items():
    bindat, _ = np.histogram(coord, bins=bins, weights=val)  # aggregated
    smooth_data[key] = bindat / counts

ratio = np.average(
    smooth_data['measurements'][1:] / smooth_data['predictions'][1:]
    )
if np.isclose(ratio, 1, rtol=1e-2):
    corrct_tag = ''
else:
    corrct_tag = r'${:g} \times$ '.format(ratio)
    smooth_data['measurements'] = smooth_data['measurements'] / ratio
    warnings.warn(
        "2-point measurements downscaled by {:g}. ".format(ratio),
        RuntimeWarning
        )

# Visualise data product.
plt.style.use(harmony)
plt.close('all')

plt.subplot2grid((4, 8), (0, 0), rowspan=3, colspan=8)
plt.loglog(
    bincoord, smooth_data['measurements'], '-', marker='+',
    label='{:.2f}x measurements'.format(1/ratio))
plt.loglog(
    bincoord, smooth_data['predictions'], ':', marker='+', label='predictions'
    )

xlim = plt.gca().get_xlim()
yceil = 0.02

plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel(
    r'$\kappa \left\langle\delta \delta^*\right\rangle$ [$(\textrm{Mpc}/h)^2$]'
    )
plt.legend()

plt.subplot2grid((4, 8), (3, 0), rowspan=1, colspan=8)
plt.plot(
    bincoord, smooth_data['measurements']/smooth_data['predictions']-1, '--'
    )

yceil = 0.02

plt.fill_between(xlim, [yceil,]*2, [-yceil]*2, alpha=0.25)
plt.xlim(xlim)
plt.ylim(bottom=-0.1,top=0.04)
plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
plt.ylabel(
    r'${:.2f} \left\langle \delta\delta^* \right\rangle /$'.format(1/ratio)
    + r'$\left\langle \delta\delta^* \right\rangle_\mathrm{{model}} - 1$'
    )

plt.subplots_adjust(wspace=0, hspace=0)
