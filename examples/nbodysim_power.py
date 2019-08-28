"""Recover real-space power spectrum from fixed-epoch N-body simulations.

"""
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import CSVCatalog, FFTPower

from power_rc import PATHIN, PATHOUT, fdir, get_filename
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import harmony
from harmonia.mapper import SphericalMap


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

parser = ArgumentParser(description="Power spectrum recovery set-up. ")

parser.add_argument('--infile', type=str, default="halos-(NG=0.,z=1.)-0L.txt")
parser.add_argument('--kmax', type=float, default=2.)
parser.add_argument('--boxside', type=float, default=1000.)
parser.add_argument('--dk', type=float, default=1e-2)

params = parser.parse_args()

file = params.infile
kmax = params.kmax
L = params.boxside
dk = params.dk


# -- Runtime constants --------------------------------------------------------

HEADINGS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass',]
NMESH = 512


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
    resampler='tsc', Nmesh=NMESH, compensated=True, interlaced=True
    )
cpow = FFTPower(mesh, mode='1d', dk=dk, kmax=kmax).power

# Run spherical algorithm.
mapp = SphericalMap(disc, clog)
spow = mapp.spherical_power()


# == FINALISATION =============================================================

# -- Export -------------------------------------------------------------------

data = {
    'Nk': cpow['modes'],
    'k': cpow['k'],
    'Pk': cpow['power'].real,
    'Pshot': cpow.attrs['shotnoise'],
    'ln': modes,
    'kln': waves,
    'Pln': np.concatenate(spow)[order],
    }
np.save(f"{PATHOUT}{fdir}{get_filename(file)}.npy", data)


# -- Visualise ----------------------------------------------------------------

try:
    plt.style.use(harmony)
    plt.close('all')
    plt.figure('N-body simulation power recovery')

    c = plt.errorbar(
        data['k'], data['Pk'], yerr=np.sqrt(2*data['Pk']**2/data['Nk']),
        color='#C40233', elinewidth=.8, label='Cartesian'
        )
    s = plt.loglog(
        data['kln'], data['Pln'], color='#0087BD', label='spherical'
        )

    POL_NUM = 2  # number of poles
    POL_COLOUR = ['#000000', '#FFD300']  # colour of poles
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
                verticalalignment='bottom', fontsize=7
                )

    plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
    plt.savefig(f"{PATHOUT}{fdir}{get_filename(file)}.pdf")
except:
    pass

"""Use the following to combine results from two catalogues:

dataL = np.load(f"{PATHOUT}{fdir}{}-L.npy").item()
dataR = np.load(f"{PATHOUT}{fdir}{}-R.npy").item()

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
for l, data in enumerate([dataL, dataR]):
    plt.loglog(data['kln'], data['Pln'], ls=LS[l], color='#009F6B', alpha=.5)

"""
