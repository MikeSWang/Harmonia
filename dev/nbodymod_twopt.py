"""Predict 2-point function values for fixed-epoch N-body simulations.

"""
import warnings
from functools import partial

import numpy as np
from mpi4py import MPI
from scipy.interpolate import interp1d

from twoptrc import PATHIN, PATHOUT, fname, fdir, params
from twoptrc import clean_warnings, confirm_dir, mpicomp
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.reader import coupling_list, twopoint_signal, twopoint_shotnoise


# == INITIALISATION ===========================================================

warnings.formatwarning = clean_warnings

# -- Runtime parameters -------------------------------------------------------

struct = params.struct
nbar = params.nbar
bias = params.bias
kmax = params.kmax
rmax = params.boxside/2

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

FILE = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"

# -- Cosmology ----------------------------------------------------------------

k_samp, Pk_samp = np.loadtxt("".join([PATHIN, fdir, FILE, ".txt"])).T
Plin = interp1d(k_samp, Pk_samp, assume_sorted=True)
beta = 0.

# -- Program identifier -------------------------------------------------------

if beta:
    subdir, rsd_tag = "wrsd/", "{:.2f}".format(beta)
else:
    subdir, rsd_tag = "nrsd/", 'none'


# == PROCESSING ===============================================================

# Set up discretisation and indexing.
disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
indx_vec = SphericalArray.build(disc=disc).unfold(struct, retonly='index')

# Compute couplings with parallel processes.
in2co = {
    coupletype: partial(coupling_list, coupletype=coupletype, disc=disc)
    for coupletype in ['ang', 'rad', 'rsd']
    }
couplings = mpicomp(indx_vec, in2co, comm)

# Compute 2-pt values.
if rank == 0:
    cov_signal = np.zeros((disc.nmodes, disc.nmodes), dtype=complex)
    cov_shotnoise = np.zeros((disc.nmodes, disc.nmodes), dtype=complex)
    for rowidx in range(disc.nmodes):
        for colidx in range(rowidx+1):
            cov_signal[rowidx, colidx] = bias**2 * twopoint_signal(
                Plin, beta, disc,
                M_mu_all=couplings['ang'][rowidx],
                M_nu_all=couplings['ang'][colidx],
                Phi_mu_all=couplings['rad'][rowidx],
                Phi_nu_all=couplings['rad'][colidx],
                Upsilon_mu_all=couplings['rsd'][rowidx],
                Upsilon_nu_all=couplings['rsd'][colidx]
                )

            mu, nu = indx_vec[rowidx], indx_vec[colidx]
            M_munu = couplings['ang'][rowidx][nu[0]][nu[1]+nu[0]]
            cov_shotnoise[rowidx, colidx] = twopoint_shotnoise(
                mu, nu, nbar, disc, M_munu
                )

    idx_upper = np.triu_indices(disc.nmodes, k=1)
    cov_signal[idx_upper] = cov_signal.T[idx_upper]
    cov_shotnoise[idx_upper] = cov_shotnoise.T[idx_upper]


# == FINALISATION =============================================================

if rank == 0:
    fpathful, fnameroot = f"{PATHOUT}{fdir}{subdir}", fname.split("_").pop()
    confirm_dir(fpathful)

    np.save(
        "".join([
            fpathful, fnameroot, "_couplings", f"-(struct={struct})", ".npy"
        ]),
        couplings
        )
    np.save(
        "".join([fpathful, fnameroot, "_2pt", f"-(struct={struct})", ".npy"]),
        {'signal': cov_signal,
         'shotnoise': cov_shotnoise,
         'covar': cov_signal + cov_shotnoise,
         }
        )
