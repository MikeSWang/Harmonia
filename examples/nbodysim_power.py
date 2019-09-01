"""Recover real-space power spectrum from fixed-epoch N-body simulations.

"""
import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import CSVCatalog, FFTPower

from powerrc import PATHIN, PATHOUT, fdir, params, get_filename, confirm_dir
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import harmony
from harmonia.mapper import SphericalMap


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

file = params.infile
kmax = params.kmax
L = params.boxside
meshcal = params.meshcal

# -- Runtime constants --------------------------------------------------------

HEADINGS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass',]


# == PROCESSING ===============================================================

# Set up discretisation.
disc = DiscreteSpectrum(L/2, 'Dirichlet', kmax)
order = np.concatenate(disc.wavenumbers).argsort()
modes = np.concatenate(disc.waveindices)[order]
waves = np.concatenate(disc.wavenumbers)[order]

# Build catalogue (note unit conversion!).
clog = CSVCatalog(f"{PATHIN}{fdir}{file}", HEADINGS)

clog.attrs['BoxSize'] = L

clog['Position'] = clog['x'][:, None] * [1, 0, 0] \
    + clog['y'][:, None] * [0, 1, 0] \
    + clog['z'][:, None] * [0, 0, 1]

# Run Cartesian algorithm.
mesh = clog.to_mesh(
    resampler='tsc', Nmesh=meshcal, compensated=True, interlaced=True
    )
cpow = FFTPower(mesh, mode='1d', kmax=kmax).power

# Run spherical algorithm.
mapp = SphericalMap(disc, clog)
spow = mapp.spherical_power()


# == FINALISATION =============================================================

fpathful, fnameful = f"{PATHOUT}{fdir}", f"{get_filename(file)}"
confirm_dir(fpathful)

# -- Export -------------------------------------------------------------------

output = {
    'Nk': [cpow['modes']],
    'k': [cpow['k']],
    'Pk': [cpow['power'].real],
    'Pshot': [cpow.attrs['shotnoise']],
    'ln': [modes],
    'kln': [waves],
    'Pln': [np.concatenate(spow)[order]],
    }
np.save("".join([fpathful, fnameful, ".npy"]), output)

# -- Visualise ----------------------------------------------------------------

data = {var: val[0] for var, val in output.items()}
try:
    plt.style.use(harmony)
    plt.close('sall')

    c = plt.errorbar(
            data['k'], data['Pk'], yerr=data['Pk']/np.sqrt(data['Nk']),
            elinewidth=.8, color='#C40233', label='Cartesian'
            )
    s = plt.loglog(
            data['kln'], data['Pln'], color='#0087BD', label='spherical'
            )

    POL_NUM = 2  # number of poles
    POL_COLOUR = ['#000000', '#FFD300',]  # colour of poles
    for ell in range(POL_NUM):
        ellsel = (data['ln'][:, 0] == ell)
        plt.scatter(
            data['kln'][ellsel], data['Pln'][ellsel],
            color=POL_COLOUR[ell], label=r'$\ell={:d}$'.format(ell)
            )

    for idx, ind_lab in enumerate(data['ln']):
        if ind_lab[0] == 0 and False:
            plt.annotate(
                r'$({:d},{:d})$'.format(ind_lab[0], ind_lab[1]),
                xy=(data['kln'][idx], data['Pln'][idx]),
                verticalalignment='bottom', fontsize=6
                )

    plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
    plt.savefig("".join([fpathful, fnameful, ".pdf"]))
except Exception as e:
    print(e)

"""Use the following to combine results from two catalogues:

dataL = np.load(f"{fpathful}{}L.npy").item()
dataR = np.load(f"{fpathful}{}R.npy").item()

data = {
    'Nk': (dataL['Nk'] + dataR['Nk']),
    'k': (dataL['k'] + dataR['k'])/2,
    'Pk': (dataL['Pk'] + dataR['Pk'])/2,
    'Pshot': (dataL['Pshot'] + dataR['Pshot'])/2,
    'ln': (dataL['ln'] + dataR['ln'])/2,
    'kln': (dataL['kln'] + dataR['kln'])/2,
    'Pln': (dataL['Pln'] + dataR['Pln'])/2,
    }

LS = ['--', ':']
for flag, data in enumerate([dataL, dataR]):
    plt.loglog(
        data['kln'], data['Pln'], ls=LS[flag], color='#009F6B', alpha=.5
        )

"""
