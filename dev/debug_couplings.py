import numpy as np
from nbodykit.lab import cosmology as cosmo

from twopointrc import PATHOUT
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import unitconst
from harmonia.reader import coupling_list

# Indexing set up.
rmax = cosmo.Planck15.comoving_distance(0.05)
disc = DiscreteSpectrum(rmax, 'Dirichlet', 0.1)

indx_arr = SphericalArray.build(disc=disc)
_, indx_vec = indx_arr.unfold(axis_order='natural')

# Load results computed with hard integration.
couplings = np.load(
    f"{PATHOUT}predictions/wRSD/predictions-couplings-"
    "(nbar=0.001,rmax=148.,b=1.,rsd=0.52,ord=natural)"
    ".npy"
    ).item()

# Calculate model with shortcut integration results.
M_all, Phi_all, Upsi_all = [], [], []
for mu in indx_vec:
    M_all.append(coupling_list(mu, 'ang', disc))
    Phi_all.append(coupling_list(mu, 'rad', disc, sel=unitconst))
    Upsi_all.append(coupling_list(mu, 'rsd', disc, sel=unitconst))

Couplings = {'ang': M_all, 'rad': Phi_all, 'rsd': Upsi_all}

# Check agreement.
closeness = {'ang': [], 'rad': [], 'rsd':[]}
for key in closeness:
    for idx in range(disc.nmodes):
        close_idx = []
        for Mu_list, mu_list in zip(Couplings[key][idx], couplings[key][idx]):
            close_list = []
            for Ell_line, ell_line in zip(Mu_list, mu_list):
                close_list.append(np.allclose(Ell_line, ell_line))
            close_idx.append(close_list)
        closeness[key].append(close_idx)
