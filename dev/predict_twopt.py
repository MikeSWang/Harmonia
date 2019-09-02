"""Predict 2-point function values from a set of cosmological parameters.

"""
import numpy as np
from mpi4py import MPI
from nbodykit.lab import cosmology as cosmo

from twoptrc import PATHOUT, fdir, fname, params, confirm_dir
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    allocate_segments, unitconst, format_float as ff
    )
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
    f"-(nbar={ff(nbar, 'sci')},rmax={ff(rmax, 'intdot')},"
    f"b={ff(bias, 'decdot')},rsd={rsd_tag},ord={struct})"
    )


# == PROCESSING ===============================================================

if rank == 0: print(ftag)

# Set up discretisation and indexing.
disc = DiscreteSpectrum(rmax, 'dirichlet', kmax)

indx_arr = SphericalArray.build(disc=disc)
_, indx_vec = indx_arr.unfold(struct)

# Compute couplings with parallel processes.
segment = allocate_segments(ntask=len(indx_vec), nproc=size)
indx_segment = indx_vec[segment[rank]]

M_seg, Phi_seg, Upsi_seg = [], [], []
for mu in indx_segment:
    M_seg.append(coupling_list(mu, 'ang', disc))
    Phi_seg.append(coupling_list(mu, 'rad', disc))
    Upsi_seg.append(coupling_list(mu, 'rsd', disc))

comm.Barrier()

M_all = comm.gather(M_seg, root=0)
Phi_all = comm.gather(Phi_seg, root=0)
Upsi_all = comm.gather(Upsi_seg, root=0)

# Compute 2-pt values.
if rank == 0:
    M_all = np.concatenate(M_all, axis=0).tolist()
    Phi_all = np.concatenate(Phi_all, axis=0).tolist()
    Upsi_all = np.concatenate(Upsi_all, axis=0).tolist()

    cov_signal = np.zeros((disc.nmodes, disc.nmodes), dtype=complex)
    cov_shotnoise = np.zeros((disc.nmodes, disc.nmodes), dtype=complex)
    for rowidx in range(disc.nmodes):
        for colidx in range(rowidx+1):
            mu, nu = indx_vec[rowidx], indx_vec[colidx]
            cov_signal[rowidx, colidx] = bias**2 * twopoint_signal(
                mu, nu, Plin, beta, disc,
                M_mu_all=M_all[rowidx], M_nu_all=M_all[colidx],
                Phi_mu_all=Phi_all[rowidx], Phi_nu_all=Phi_all[colidx],
                Upsilon_mu_all=Upsi_all[rowidx],
                Upsilon_nu_all=Upsi_all[colidx]
                )

            ellidx_nu, midx_nu = nu[0], nu[1] + nu[0]
            M_munu = M_all[rowidx][ellidx_nu][midx_nu]
            cov_shotnoise[rowidx, colidx] = twopoint_shotnoise(
                mu, nu, nbar, disc, M_munu, sel=unitconst
                )

    idx_upper = np.triu_indices(disc.nmodes, k=1)
    cov_signal[idx_upper] = cov_signal.T[idx_upper]
    cov_shotnoise[idx_upper] = cov_shotnoise.T[idx_upper]


# == FINALISATION =============================================================

if rank == 0:
    fpathful, fnameroot = f"{PATHOUT}{fdir}{subdir}", fname.split("_").pop()
    confirm_dir(fpathful)

    output_couplings = {
        'ang': M_all,
        'rad': Phi_all,
        'rsd': Upsi_all,
        }
    np.save(
        "".join([fpathful, fnameroot, "_couplings", ftag, ".npy"]),
        output_couplings
        )

    output_2pt = {
        'signal': cov_signal,
        'shotnoise': cov_shotnoise,
        'covar': cov_signal + cov_shotnoise,
        }
    np.save(
        "".join([fpathful, fnameroot, "_2pt", ftag, ".npy"]),
        output_2pt
        )