import numpy as np
from nbodykit.lab import cosmology as cosmo

from twopointrc import PATHOUT
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.reader import twopoint_signal, twopoint_shotnoise

# Runtime constants.
NBAR, ZMAX, STRUCT = 1e-3, 0.05, 'natural'
BIAS = 1.
KMAX = 0.1
Z = 0.

rmax = cosmo.Planck15.comoving_distance(ZMAX)
Plin = cosmo.LinearPower(cosmo.Planck15, redshift=Z, transfer='CLASS')
beta0 = cosmo.Planck15.scale_independent_growth_rate(0.) / BIAS

# Set up discretisation and indexing.
disc = DiscreteSpectrum(rmax, 'Dirichlet', KMAX)

order = np.concatenate(disc.wavenumbers).argsort()
modes = np.concatenate(disc.waveindices)[order]
waves = np.concatenate(disc.normcoeff)[order]

indx_arr = SphericalArray.build(disc=disc)
_, indx_vec = indx_arr.unfold(axis_order=STRUCT)

# Load 2-pt values.
output_2pt = np.load(
    f"{PATHOUT}predictions/wRSD/predictions-2pt-"
    "(nbar=0.001,rmax=148.,b=1.,rsd=0.52,ord=natural)"
    ".npy"
    ).item()
covar = output_2pt['covar']

# Read couplings.
output_couplings = np.load(
    f"{PATHOUT}predictions/wRSD/predictions-couplings-"
    "(nbar=0.001,rmax=148.,b=1.,rsd=0.52,ord=natural)"
    ".npy"
    ).item()
M_all = output_couplings['ang']
Phi_all = output_couplings['rad']
Upsi_all = output_couplings['rsd']

# Compute predicted 2-pt values.
Cov_signal2pt = np.zeros((disc.nmodes, disc.nmodes), dtype=complex)
Cov_shotnoise2pt = np.zeros((disc.nmodes, disc.nmodes), dtype=complex)
for rowidx in range(disc.nmodes):
    for colidx in range(rowidx+1):
        mu, nu = indx_vec[rowidx], indx_vec[colidx]
        Cov_signal2pt[rowidx, colidx] = BIAS**2 * twopoint_signal(
            mu, nu, Plin, beta0, disc,
            M_mu_all=M_all[rowidx], M_nu_all=M_all[colidx],
            Phi_mu_all=Phi_all[rowidx], Phi_nu_all=Phi_all[colidx],
            Upsilon_mu_all=Upsi_all[rowidx], Upsilon_nu_all=Upsi_all[colidx]
            )
        ellidx_nu, midx_nu = nu[0], nu[1] + nu[0]
        M_munu = M_all[rowidx][ellidx_nu][midx_nu]
        Cov_shotnoise2pt[rowidx, colidx] = twopoint_shotnoise(
            mu, nu, NBAR, disc, M_munu
            )

idx_upper = np.triu_indices(disc.nmodes, k=1)
Cov_signal2pt[idx_upper] = Cov_signal2pt.T[idx_upper]
Cov_shotnoise2pt[idx_upper] = Cov_shotnoise2pt.T[idx_upper]

Covar = Cov_signal2pt + Cov_shotnoise2pt
