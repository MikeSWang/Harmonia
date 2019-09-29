"""Recover constant power spectrum from unclustered catalogues.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import FFTPower

from powerrc import PATHOUT, confirm_dir, filename, params, quick_plot
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import format_float, harmony
from harmonia.mapper import RandomCatalogue, SphericalMap


# == INITIALISATION ===========================================================

nbar = params.nbar
rmax = params.rmax
kmax = params.kmax
meshcal = params.meshcal
niter = params.niter
progid = params.progid

prog_tag = "-(nbar={},rmax={},kmax={},mesh={},iter={})-".format(
    format_float(nbar, 'sci'),
    format_float(rmax, 'intdot'),
    format_float(kmax, 'sci'),
    meshcal,
    niter,
)
prog_tag += "[" + progid + "]"

print(prog_tag.lstrip("-"))


# == PROCESSING ===============================================================

disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
order = np.concatenate(disc.wavenumbers).argsort()
waves = np.concatenate(disc.wavenumbers)[order]
modes = np.concatenate(disc.dbl_indices)[order]

measurements = defaultdict(list)
for run in range(niter):
    data_catalogue = RandomCatalogue(nbar, 2*rmax)
    mesh = data_catalogue.to_mesh(
        Nmesh=meshcal,
        resampler='tsc',
        compensated=True,
    )

    cartesian_power = FFTPower(mesh, mode='1d', kmax=kmax).power
    spherical_map = SphericalMap(disc, data_catalogue, mean_density_data=nbar)
    spherical_power = spherical_map.spherical_power()

    measurements['k'].append([cartesian_power['k']])
    measurements['Nk'].append([cartesian_power['modes']])
    measurements['Pk'].append([cartesian_power['power'].real])
    measurements['Pshot'].append([cartesian_power.attrs['shotnoise']])
    measurements['Pln'].append([np.concatenate(spherical_power)[order]])


# == FINALISATION =============================================================

base_path, root_name = f"{PATHOUT}{filename}", f"{filename}{prog_tag}"
assert confirm_dir(base_path)

output = {var: np.concatenate(vals) for var, vals in measurements.items()}
output.update({'ln': [modes], 'kln': [waves]})
np.save("".join([base_path, "/", root_name, ".npy"]), output)

try:
    plt.style.use(harmony)
    plt.close('all')
    quick_plot(output)
    plt.savefig("".join([base_path, "/", root_name, ".pdf"]))
except Exception as e:
    print(e)
