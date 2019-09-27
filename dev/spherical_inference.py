"""Spherical likelihood inference for primordial non-Gaussianity.

"""
import warnings

import numpy as np
from astropy import constants
from nbodykit.lab import cosmology as cosmo

from inference_rc import PATHOUT, confirm_dir, filename, mpicomp, params
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import format_float
from harmonia.cosmology import fiducial_cosmology, fiducial_distance
from harmonia.mapper import NBKCatalogue, SphericalMap
from harmonia.reader import Couplings, two_point_signal, two_point_shot_noise


# == INITIALISATION ===========================================================

# -- External parameters ------------------------------------------------------

pivot = params.pivot

nbar = params.nbar
bias = params.bias
redshift = params.redshift
zmax = params.zmax
kmax = params.kmax

expand = params.expand
meshgen = params.meshgen
niter = params.niter
progid = params.progid


# -- Internal parameters ------------------------------------------------------

BETA = 0
SPHERICAL_COLLAPSE_THRESHOLD = 1.686

SURVEY_SPECIFICATION = {
    'mask': None,
    'selection': None,
    'weight': None,
    'weight_derivative': None,
}

COSMOLOGY_SPECIFICATION = {
    'r2z': None,
    'z2chi': None,
    'evolution': None,
    'AP_distortion': None,
}


# -- Derived parameters -------------------------------------------------------

rmax = fiducial_distance(zmax)
boxsize = 2 * expand * rmax

transfer_func = cosmo.power.transfers.CLASS(fiducial_cosmology, redshift)
linear_power = cosmo.LinearPower(fiducial_cosmology, redshift=redshift, transfer='CLASS')

disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    indices_flat = SphericalArray.build(disc=disc).unfold(
        pivot,
        return_only='index',
    )

couplings_kwargs = dict(
    disc=disc,
    survey_specs=SURVEY_SPECIFICATION,
    cosmo_specs=COSMOLOGY_SPECIFICATION,
)

info = "nbar={},bias={},rmax={},kmax={},xpd={},mesh={},iter={}".format(
    format_float(nbar, 'sci'),
    format_float(bias, 'decdot'),
    format_float(rmax, 'intdot'),
    format_float(kmax, 'sci'),
    format_float(expand, 'decdot'),
    meshgen,
    niter,
)
program_tag = "".join(["-(", info, ")-", "[", progid, "]"])

print(program_tag)


# == PROCESSING ===============================================================

# -- Data generation ----------------------------------------------------------

catalogue = NBKCatalogue(linear_power, nbar, boxsize, bias=bias, Nmesh=meshgen)
spherical_map = SphericalMap(disc, catalogue, mean_density_data=nbar)
two_point_measurements = spherical_map.twopoint(pivot=pivot)


# -- Model building -----------------------------------------------------------

poisson_kernel = lambda k: 3 \
    * (100*fiducial_cosmology.h / constants.c.to('km/s'))**2 \
    * SPHERICAL_COLLAPSE_THRESHOLD * fiducial_cosmology.Omega0_m \
    / transfer_func(k)

scale_dependent_bias = lambda k, f_NL: bias \
    + f_NL * (bias - 1) * poisson_kernel(k) / k**2

power_spectrum_model = lambda k, f_NL: scale_dependent_bias(k)**2 \
    * linear_power(k)

couplings = Couplings(**couplings_kwargs)
couplings_to_coefficients = {
    coupling_type: lambda mu: Couplings.compile_over_index(mu, coupling_type)
    for coupling_type in ['angular', 'radial', 'RSD']
}
coefficients = mpicomp(indices_flat, couplings_to_coefficients)

cov_signal = np.zeros((disc.nmode, disc.nmode), dtype=complex)
cov_shotnoise = np.zeros((disc.nmode, disc.nmode), dtype=complex)
for rowidx in range(disc.nmode):
    for colidx in range(rowidx+1):
        cov_signal[rowidx, colidx] = two_point_signal(
            power_spectrum_model, BETA, disc,
            M_mu_all=coefficients['angular'][rowidx],
            M_nu_all=coefficients['angular'][colidx],
            Phi_mu_all=coefficients['radial'][rowidx],
            Phi_nu_all=coefficients['radial'][colidx],
            Upsilon_mu_all=coefficients['RSD'][rowidx],
            Upsilon_nu_all=coefficients['RSD'][colidx]
            )

        mu, nu = indices_flat[rowidx], indices_flat[colidx]
        M_munu = couplings['ang'][rowidx][nu[0]][nu[1]+nu[0]]
        cov_shotnoise[rowidx, colidx] = two_point_shot_noise(
            mu, nu, nbar, disc, M_munu
            )

idx_upper = np.triu_indices(disc.nmode, k=1)
cov_signal[idx_upper] = cov_signal.T[idx_upper]
cov_shotnoise[idx_upper] = cov_shotnoise.T[idx_upper]
