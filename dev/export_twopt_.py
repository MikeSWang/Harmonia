"""Export recovered 2-point function values.

"""
import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology as cosmo

from twopointrc import PATHOUT, aggregate, confirm_dir
from view_twopt_ import view_covariance_ as view_covariance
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import collate


# == CONFIGURATION ============================================================

STRUCT = 'natural'

DATADIR = "measure_twopt/"
MODELDIR = "predict_twopt/"
SUBDIR = "nRSD/"

DATANAME = "measure_twopt"
MODELNAME = "predict_2pt"

TAG_DATA = (
    "-("
    ")-agg"
    )
TAG_MODEL = f"-(ord={STRUCT})"

COLLATE = False
SAVE = 'agg'  # 'full', 'agg', ''

LOAD_DATA = False
LOAD_MODEL = False

SAVEFIG = False


# == OPERATION ================================================================

collate_path = f"{PATHOUT}{DATADIR}{SUBDIR}collated/"
confirm_dir(collate_path)

# Load files.
if COLLATE:
    output_data, count, _ = collate(
        f"{PATHOUT}{DATADIR}{SUBDIR}{DATANAME}-*.npy", 'npy'
        )
    data = aggregate(output_data)

    save_str = "".join(_.split("(")[-1].split(")")[:-1])
    if SAVE == 'full':
        np.save(
            f"{collate_path}{DATANAME}-({save_str}*{count}).npy",
            output_data
            )
    elif SAVE == 'agg':
        np.save(
            f"{collate_path}{DATANAME}-({save_str}*{count})-agg.npy",
            data
            )

if LOAD_DATA:
    output_data = np.load(f"{collate_path}{DATANAME}{TAG_DATA}.npy").item()
    if TAG_DATA.endswith("agg"):
        data = output_data
    else:
        data = aggregate(output_data)

if LOAD_MODEL:
    model = np.load(
        f"{PATHOUT}{MODELDIR}{SUBDIR}{MODELNAME}{TAG_MODEL}.npy"
        ).item()

# Set up indexing. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
diff = view_covariance(m2pt, m2pt=None, ind=ind,
                     lb=indx_vec[ind], diff=True, diag='only')
if SAVEFIG:
    plt.savefig(
        f"{PATHOUT}2pt_{STRUCT}-(b=2.,rsd=none)-m_rel_d.png",
        format='png', dpi=500, transparent=False
        )
