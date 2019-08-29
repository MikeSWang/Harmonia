"""Export recovered two-point values.

"""
import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology as cosmo

from twopointrc import PATHOUT
from view_twopt import view_twopoint
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import collate


# == DEFINITION ===============================================================

def aggregate(data):
    return {key: np.average(val, axis=0) for key, val in data.items()}


# == EXECUTION ================================================================

CASE = "nRSD/"
BIASSTR = "b=2"
STRUCT = 'natural'

PREFIX_DATA = "measure2pt"
PREFIX_MODEL = "predict2pt"

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
LOAD_MODEL = False

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

disc = DiscreteSpectrum(500, 'Dirichlet', 0.1)

order = np.concatenate(disc.wavenumbers).argsort()
waves = np.concatenate(disc.wavenumbers)[order]
modes = np.concatenate(disc.waveindices)[order]
norms = np.concatenate(disc.normcoeff)[order]

indx_arr = SphericalArray.build(disc=disc)
_, indx_vec = indx_arr.unfold(axis_order=STRUCT)

ref = np.load('./data/input/halos-(NG=0.,z=1.)-(side=1000.,nmesh=p256,npair=11)-inscribed.npy').item()

# Visualise comparisons.
#Covar = data[STRUCT]
#covar = model['signal'] + model['shotnoise']

ind = slice(50)  # len(indx_vec)

#d2pt = np.abs(Covar)
#
#if LOAD_MODEL: m2pt = np.abs(covar)
#else: m2pt = None

m2pt = np.zeros(disc.nmodes)
for idx, greek in enumerate(indx_vec):
    condition = np.logical_and(
        ref['ln'][:,0] == greek[0],
        ref['ln'][:,1] == greek[-1]
        )
    refidx = np.where(condition)[0][0]
    m2pt[idx] = ref['Pln'][refidx] / norms[refidx]
m2pt = np.diag(m2pt)

# None / indx_vec[ind], True / False, None / 'only' / 'off'
diff = view_twopoint(m2pt, m2pt=None, ind=ind,
                     lb=indx_vec[ind], diff=True, diag='only')
if SAVEFIG:
    plt.savefig(
        f"{PATHOUT}2pt_{STRUCT}-(b=2.,rsd=none)-m_rel_d.png",
        format='png', dpi=500, transparent=False
        )
