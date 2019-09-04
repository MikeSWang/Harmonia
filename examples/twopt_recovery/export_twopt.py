"""Export recovered 2-point function values.

"""
import warnings

import numpy as np
from matplotlib.pyplot import savefig

from twoptrc import PATHIN, PATHOUT, aggregate, confirm_dir, clean_warnings
from view_twopt import view_covariance
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import collate

warnings.formatwarning = clean_warnings

DATADIR = "measure_twopt/"
MODELDIR = "predict_twopt/"
REFDIR = "nbodymod_twopt/"

DATANAME = "measure_twopt"
MODELNAME = "predict_2pt"
REFNAME = "halos"

# == CONFIGURATION ============================================================

SUBDIR = "nrsd/"
STRUCT = 'natural'

TAG_DATA = "-()-agg"
TAG_MODEL = f"-(ord={STRUCT})"
TAG_REF = (
    "-(NG=0.,z=1.)-"
    "(nbar=2.49e-4,b=2.3415,beta=none,kmax=0.04,side=1000.,nmesh=256,npair=11)"
    )

COLLATE = False
SAVE = 'agg'  # 'full', 'agg', ''

LOAD_DATA = False
LOAD_MODEL = False
LOAD_REF = True

SAVEFIG = False

DISC = DiscreteSpectrum(500, 'Dirichlet', 0.04)


# == OPERATION ================================================================

datpathful = f"{PATHOUT}{DATADIR}{SUBDIR}"
modpathful = f"{PATHOUT}{MODELDIR}{SUBDIR}"
collatepath = datpathful + "collated/"

# I/O files.
if COLLATE:
    output, count, _ = collate(f"{datpathful}{DATANAME}-*.npy", 'npy')
    data = aggregate(output)

    save_str = "".join(_.split("(")[-1].split(")")[:-1]) + f"*{count}"
    confirm_dir(collatepath)
    if SAVE == 'full':
        np.save(f"{collatepath}{DATANAME}-({save_str})-all.npy", output)
    elif SAVE == 'agg':
        np.save(f"{collatepath}{DATANAME}-({save_str})-agg.npy", data)

if LOAD_DATA:
    output = np.load(f"{collatepath}{DATANAME}{TAG_DATA}.npy").item()
    if TAG_DATA.endswith("agg"):
        data = output
    elif TAG_DATA.endswith("all"):
        data = aggregate(output)

if LOAD_MODEL:
    model = np.load(f"{modpathful}{MODELNAME}{TAG_MODEL}.npy").item()

if LOAD_REF:
    mod = np.load(
        f"{PATHOUT}{REFDIR}{SUBDIR}nbodymod_2pt-(struct={STRUCT}).npy"
        ).item()
    ref = np.load(f"{PATHIN}{REFDIR}{REFNAME}{TAG_REF}.npy").item()
    ref = aggregate(ref)

# Set up indexing.
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    indx_vec = SphericalArray.build(disc=DISC).unfold(STRUCT, retonly='index')
order = np.concatenate(DISC.wavenumbers).argsort()
norms = np.concatenate(DISC.normcoeff)[order]

# Visualise comparisons.
if LOAD_DATA or COLLATE:
    Covar = data[STRUCT]
if LOAD_MODEL:
    covar = model['signal'] + model['shotnoise']
if LOAD_REF:
    covar = mod['signal'] + mod['shotnoise']

    Covar = np.zeros(DISC.nmode)
    for idx, greek in enumerate(indx_vec):
        condition = np.logical_and(
            ref['ln'][:,0] == greek[0],
            ref['ln'][:,1] == greek[-1]
            )
        refidx = np.where(condition)[0][0]
        Covar[idx] = ref['Pln'][refidx] / norms[refidx]
    Covar = np.diag(Covar)

if LOAD_DATA or LOAD_REF or COLLATE:
    d2pt = np.abs(Covar)
if LOAD_MODEL or LOAD_REF:
    m2pt = np.abs(covar)
else:
    m2pt = None

indrange = slice(len(indx_vec))  #  len(indx_vec)
view = view_covariance(
    d2pt, m2pt=m2pt, indices=indrange,
    label_interval=indx_vec[indrange],  # None, indx_vec[indrange]
    diff=True,  # True, False
    diag='only'  # 'none', 'only', 'off'
    )
if SAVEFIG:
    savefig(
        f"{PATHOUT}" +
        f"{REFNAME}-cov-(beta=none,kmax=0.04,struct={STRUCT})-m2d.png",
        format='png', dpi=500, transparent=False
        )
