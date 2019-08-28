"""Export recovered two-point values.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology as cosmo

from twopointrc import PATHOUT, harmony
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import collate


# == DEFINITION ===============================================================

def aggregate(data):
    return {key: np.average(val, axis=0) for key, val in data.items()}


def view_twopoint(d2pt, m2pt=None, diff=False, diag=None, lb=None, ind=None):  # !!!

    plt.style.use(harmony)
    plt.close('all')
    sns.set(style='ticks', font='serif')

    if ind is not None:
        d2pt = d2pt[ind, ind]
        if m2pt is not None: m2pt = m2pt[ind, ind]

    if lb is None: lb = 10

    if m2pt is None:
        if diag == 'off': np.fill_diagonal(d2pt, np.nan)
        sns.heatmap(
            d2pt, center=0, square=True, xticklabels=lb, yticklabels=lb
            )
        return

    if diff:
        if diag == 'only':
            mask = np.diag(np.ones(len(d2pt)))
        else:
            mask0 = ~np.isclose(m2pt, 0)
            mask1 = m2pt / d2pt > 1e-3
            mask = np.logical_and(mask0, mask1)

        diff = np.where(mask, m2pt/d2pt, np.nan)  # m2pt/d2pt, d2pt/m2pt
        sns.heatmap(
            diff, center=1, annot=False,# vmin=0.9, vmax=1.1,
            square=True, xticklabels=lb, yticklabels=lb
            )
        return diff

    if diag == 'off':
        np.fill_diagonal(d2pt, np.nan)
        np.fill_diagonal(m2pt, np.nan)
    if diag == 'only':
        mask = np.diag(np.ones(len(d2pt)))
        d2pt = np.where(mask, d2pt, np.nan)
        m2pt = np.where(mask, m2pt, np.nan)

    plt.subplot2grid((4, 9), (0, 0), rowspan=4, colspan=4)
    sns.heatmap(
        d2pt, center=0,
        square=True, xticklabels=lb, yticklabels=lb, cbar_kws={'shrink': 0.6}
        )

    plt.subplot2grid((4, 9), (0, 5), rowspan=4, colspan=4)
    sns.heatmap(
        m2pt, center=0,
        square=True, xticklabels=lb, yticklabels=lb, cbar_kws={'shrink': 0.6}
        )

    return


# == EXECUTION ================================================================

CASE = "nRSD/"
BIASSTR = "b=2"
STRUCT = 'k'

PREFIX_DATA = "measurements"
PREFIX_MODEL = "predictions"

DIR_DATA = f"{PREFIX_DATA}/"
DIR_MODEL = f"{PREFIX_MODEL}/"

TAG_DATA = ("2pt-("
            "nbar=0.001,ratio=simu,b=2.,rsd=none,"
            "rmax=148.,xpd=2.,nmesh=[cp256],niter=25*100"
            ")-agg")
TAG_MODEL = f"2pt-(nbar=0.001,rmax=148.,b=2.,rsd=none,ord={STRUCT})"

COLLATE = False
SAVE = 'agg'  # 'full', 'agg', ''

LOAD_DATA = True
LOAD_MODEL = True

SAVEFIG = False

# Collate or load data.
if COLLATE:
    output_data, count, _ = collate(
        f"{PATHOUT}{DIR_DATA}2pt/{CASE}{PREFIX_DATA}-*{BIASSTR}*.npy", 'npy'
        )
    data = aggregate(output_data)

    save_str = "".join(_.split("(")[-1].split(")")[:-1])
    if SAVE == 'full':
        np.save(f"{PATHOUT}{DIR_DATA}collated/{CASE}"
                + f"{PREFIX_DATA}-2pt-({save_str}*{count}).npy",
                output_data)
    elif SAVE == 'agg':
        np.save(f"{PATHOUT}{DIR_DATA}collated/{CASE}"
                + f"{PREFIX_DATA}-2pt-({save_str}*{count})-agg.npy",
                data
                )

if LOAD_DATA:
    output_data = np.load(
        f"{PATHOUT}{DIR_DATA}collated/{CASE}{PREFIX_DATA}-{TAG_DATA}.npy"
        ).item()
    if TAG_DATA.endswith("agg"):
        data = output_data
    else:
        data = aggregate(output_data)

# Load model.
if LOAD_MODEL:
    model = np.load(
        f"{PATHOUT}{DIR_MODEL}{CASE}{PREFIX_MODEL}-{TAG_MODEL}.npy"
        ).item()

# Indexing set up.
rmax = cosmo.Planck15.comoving_distance(0.05)
Plin = cosmo.LinearPower(cosmo.Planck15, redshift=0., transfer='CLASS')

disc = DiscreteSpectrum(rmax, 'Dirichlet', 0.1)

order = np.concatenate(disc.wavenumbers).argsort()
waves = np.concatenate(disc.wavenumbers)[order]
modes = np.concatenate(disc.waveindices)[order]
norms = np.concatenate(disc.normcoeff)[order]

indx_arr = SphericalArray.build(disc=disc)
_, indx_vec = indx_arr.unfold(axis_order=STRUCT)

ref = np.load('./data/input/ref.npy').item()

# Visualise comparisons.
Covar = data[STRUCT]
covar = model['signal'] + model['shotnoise']
ind = slice(len(indx_vec))  # len(indx_vec)

d2pt = np.abs(Covar)

if LOAD_MODEL: m2pt = np.abs(covar)
else: m2pt = None

#m2pt = np.zeros(disc.nmodes)
#for idx, greek in enumerate(indx_vec):
#    condition = np.logical_and(
#        ref['ln'][:,0] == greek[0],
#        ref['ln'][:,1] == greek[-1]
#        )
#    refidx = np.where(condition)[0][0]
#    m2pt[idx] = ref['Pln'][refidx] / norms[refidx]
#m2pt = np.diag(m2pt)

# None / indx_vec[ind], True / False, None / 'only' / 'off'
diff = view_twopoint(d2pt, m2pt=m2pt, ind=ind,
                     lb=None, diff=True, diag='only')
if SAVEFIG:
    plt.savefig(
        f"{PATHOUT}2pt_{STRUCT}-(b=2.,rsd=none)-m_rel_d.png",
        format='png', dpi=500, transparent=False
        )
