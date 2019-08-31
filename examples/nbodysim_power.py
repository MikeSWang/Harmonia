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
    'Nk': cpow['modes']/2,
    'k': cpow['k'],
    'Pk': cpow['power'].real,
    'Pshot': cpow.attrs['shotnoise'],
    'ln': modes,
    'kln': waves,
    'Pln': np.concatenate(spow)[order],
    }
np.save("".join([fpathful, fnameful, ".npy"]), output)

# -- Visualise ----------------------------------------------------------------

try:
    plt.style.use(harmony)
    plt.close('all')

    c = plt.errorbar(
            output['k'], output['Pk'],
            yerr=np.sqrt(2*output['Pk']**2/output['Nk']),
            elinewidth=.8, color='#C40233', label='Cartesian'
            )
    s = plt.loglog(
            output['kln'], output['Pln'], color='#0087BD', label='spherical'
            )

    POL_NUM = 2  # number of poles
    POL_COLOUR = ['#000000', '#FFD300',]  # colour of poles
    for ell in range(POL_NUM):
        ellsel = (output['ln'][:, 0] == ell)
        plt.scatter(
            output['kln'][ellsel], output['Pln'][ellsel],
            color=POL_COLOUR[ell], label=r'$\ell={:d}$'.format(ell)
            )

    for idx, ind_lab in enumerate(output['ln']):
        if ind_lab[0] == 0 and False:
            plt.annotate(
                r'$({:d},{:d})$'.format(ind_lab[0], ind_lab[1]),
                xy=(output['kln'][idx], output['Pln'][idx]),
                verticalalignment='bottom', fontsize=6
                )

    plt.xlim(left=0.99*output['kln'].min(), right=1.01*output['kln'].max())
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
    plt.savefig("".join([fpathful, fnameful, ".pdf"]))
except Exception as e:
    print(e)

"""Use the following to combine results from two catalogues:

outputL = np.load(f"{PATHOUT}{fdir}{}L.npy").item()
outputR = np.load(f"{PATHOUT}{fdir}{}R.npy").item()

output = {
    'Nk': (outputL['Nk'] + outputR['Nk']),
    'k': (outputL['k'] + outputR['k'])/2,
    'Pk': (outputL['Pk'] + outputR['Pk'])/2,
    'Pshot': (outputL['Pshot'] + outputR['Pshot'])/2,
    'ln': (outputL['ln'] + outputR['ln'])/2,
    'kln': (outputL['kln'] + outputR['kln'])/2,
    'Pln': (outputL['Pln'] + outputR['Pln'])/2,
    }

LS = ['--', ':']
for flag, output in enumerate([outputL, outputR]):
    plt.loglog(
        output['kln'], output['Pln'], ls=LS[flag], color='#009F6B', alpha=.5
        )

"""
