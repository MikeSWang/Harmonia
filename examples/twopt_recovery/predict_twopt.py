"""Predict 2-point function values from a set of cosmological parameters.

"""
import warnings
from functools import partial

import numpy as np
from mpi4py import MPI
from nbodykit.lab import cosmology as cosmo

from twoptrc import PATHOUT, fdir, fname, params, confirm_dir, mpicomp
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import format_float as ff
from harmonia.reader import coupling_list, twopoint_signal, twopoint_shotnoise


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

nbar = params.nbar
bias = params.bias
redshift = params.redshift
rsd = params.rsd
zmax = params.zmax
kmax = params.kmax
struct = params.struct

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# -- Cosmology ----------------------------------------------------------------

rmax = cosmo.Planck15.comoving_distance(zmax)
Plin = cosmo.LinearPower(cosmo.Planck15, redshift=redshift, transfer='CLASS')
beta = rsd * cosmo.Planck15.scale_independent_growth_rate(redshift) / bias

# -- Program identifier -------------------------------------------------------

if rsd:
    subdir, rsd_tag = "wrsd/", "{:.2f}".format(beta)
else:
    subdir, rsd_tag = "nrsd/", 'none'

ftag = (
    f"-("
    f"nbar={ff(nbar, 'sci')},b={ff(bias, 'decdot')},rsd={rsd_tag},"
    f"rmax={ff(rmax, 'intdot')},kmax={ff(kmax, 'sci')},ord={struct}"
    f")"
    )


# == PROCESSING ===============================================================

if rank == 0: print(ftag)

# Set up discretisation and indexing.
disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    indx_vec = SphericalArray.build(disc=disc).unfold(struct, retonly='index')

# Compute couplings with parallel processes.
in2co = {
    coupletype: partial(coupling_list, coupletype=coupletype, disc=disc)
    for coupletype in ['ang', 'rad', 'rsd']
    }
couplings = mpicomp(indx_vec, in2co, comm)

# Compute 2-pt values.
if rank == 0:
    cov_signal = np.zeros((disc.nmode, disc.nmode), dtype=complex)
    cov_shotnoise = np.zeros((disc.nmode, disc.nmode), dtype=complex)
    for rowidx in range(disc.nmode):
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

    idx_upper = np.triu_indices(disc.nmode, k=1)
    cov_signal[idx_upper] = cov_signal.T[idx_upper]
    cov_shotnoise[idx_upper] = cov_shotnoise.T[idx_upper]


# == FINALISATION =============================================================

if rank == 0:
    fpathful, fnameroot = f"{PATHOUT}{fdir}{subdir}", fname.split("_")[0]
    confirm_dir(fpathful)

    np.save(
        "".join([fpathful, fnameroot, "_couplings", ftag, ".npy"]),
        couplings
        )
    np.save(
        "".join([fpathful, fnameroot, "_2pt", ftag, ".npy"]),
        {'signal': cov_signal,
         'shotnoise': cov_shotnoise,
         'covar': cov_signal + cov_shotnoise,
         }
        )
