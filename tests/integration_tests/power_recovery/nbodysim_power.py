"""Recover real-space power spectrum from paired N-body simulations.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import CSVCatalog, FFTPower

from powerrc import (
    PATHIN,
    PATHOUT,
    confirm_dir,
    filename,
    get_filename,
    params
)
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import format_float, harmony
from harmonia.mapper import SphericalMap


# == INITIALISATION ===========================================================

in_file = params.infile
kmax = params.kmax
boxsize = params.boxsize
meshcal = params.meshcal

HEADINGS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass']

prog_tag = "-(rmax={},kmax={},mesh={})".format(
    format_float(boxsize/2, 'intdot'),
    format_float(kmax, 'sci'),
    meshcal,
)


# == PROCESSING ===============================================================

disc = DiscreteSpectrum(boxsize/2, 'Dirichlet', kmax)
order = np.concatenate(disc.wavenumbers).argsort()
waves = np.concatenate(disc.wavenumbers)[order]
modes = np.concatenate(disc.dbl_indices)[order]

measurements = defaultdict(list)
for file_suffix in ["L.txt", "R.txt"]:
    catalogue = CSVCatalog(
        "".join([PATHIN, filename, "/", in_file, file_suffix]),
        HEADINGS,
    )
    catalogue.attrs['BoxSize'] = boxsize
    catalogue['Position'] = catalogue['x'][:, None] * [1, 0, 0] \
        + catalogue['y'][:, None] * [0, 1, 0] \
        + catalogue['z'][:, None] * [0, 0, 1]

    mesh = catalogue.to_mesh(
        Nmesh=meshcal,
        resampler='tsc',
        compensated=True,
        interlaced=True,
    )
    cartesian_power = FFTPower(mesh, mode='1d', kmax=kmax).power

    spherical_map = SphericalMap(disc, catalogue)
    spherical_power = spherical_map.spherical_power()

    measurements['k'].append([cartesian_power['k']])
    measurements['Nk'].append([cartesian_power['modes']])
    measurements['Pk'].append([cartesian_power['power'].real])
    measurements['Pshot'].append([cartesian_power.attrs['shotnoise']])
    measurements['Pln'].append([np.concatenate(spherical_power)[order]])


# == FINALISATION =============================================================

base_path = f"{PATHOUT}{filename}"
root_name = f"{get_filename(in_file)}{prog_tag}"
assert confirm_dir(base_path)

output = {var: np.concatenate(vals) for var, vals in measurements.items()}
output.update({'ln': [modes], 'kln': [waves]})
np.save("".join([base_path, "/", root_name, ".npy"]), output)

results = {
    'Nk': np.sum(output['Nk'], axis=0),
    'k': np.average(output['k'], axis=0),
    'Pk': np.average(output['Pk'], axis=0),
    'Pshot': np.average(output['Pshot']),
    'ln': output['ln'],
    'kln': output['kln'],
    'Pln': np.average(output['Pln'], axis=0),
}
try:
    plt.style.use(harmony)
    plt.close('sall')

    cartesian_result = plt.errorbar(
        results['k'],
        results['Pk'],
        yerr=results['Pk']/np.sqrt(results['Nk']),
        color='#C40233',
        label='Cartesian',
    )
    spherical_result = plt.loglog(
        results['kln'],
        results['Pln'],
        color='#0087BD',
        label='spherical',
        )

    POLE_NUM = 2  # number of poles
    POLE_COLOUR = ['#000000', '#FFD300',]  # colour of poles
    for ell in range(POLE_NUM):
        idx_sel = (results['ln'][:, 0] == ell)
        plt.scatter(
            results['kln'][idx_sel],
            results['Pln'][idx_sel],
            color=POLE_COLOUR[ell],
            label=r'$\ell={:d}$'.format(ell),
        )

    plt.xlim(left=0.99*results['kln'].min(), right=1.01*results['kln'].max())
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
    plt.savefig("".join([base_path, "/", root_name, ".pdf"]))
except Exception as e:
    print(e)
