"""Recover constant power spectrum from unclustered catalogues.

"""
import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import FFTPower

from powerrc import PATHOUT, argv, fdir, fname, save_data
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import harmony, format_float as ff
from harmonia.mapper import SphericalMap, RandomCatalogue


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

sarg = iter(argv[1:])
try:
    nbar, rmax, niter = float(next(sarg)), float(next(sarg)), int(next(sarg))
    try:
        progid = "-[{}]".format(next(sarg))
    except StopIteration:
        progid = ""
except StopIteration:
    nbar, rmax, niter, progid = 1e-3, 150, 25, ""

KMAX = 0.125
MESHCAL = 256


# -- Program identifier -------------------------------------------------------

ftag = "-(nbar={},rmax={},niter={}){}".format(
    ff(nbar, 'sci'), ff(rmax, 'intdot'), niter, progid
    )


# == PROCESSING ===============================================================

print(ftag)

# Set up discretisation.
disc = DiscreteSpectrum(rmax, 'Dirichlet', KMAX)
order = np.concatenate(disc.wavenumbers).argsort()
modes = np.concatenate(disc.waveindices)[order]
waves = np.concatenate(disc.wavenumbers)[order]

k_, Nk_, Pk_, Pshot_, Pln_ = [], [], [], [], []
for run in range(niter):
    # Generate data catalogue.
    data = RandomCatalogue(nbar, boxsize=2*rmax)

    # Run Cartesian algorithm.
    mesh = data.to_mesh(Nmesh=MESHCAL, resampler='tsc', compensated=True)
    cpow = FFTPower(mesh, mode='1d', kmax=KMAX).power

    # Run spherical algorithm.
    mapp = SphericalMap(disc, data, nmean_data=nbar)
    spow = mapp.spherical_power()

    # Append reordered results.
    k_.append([cpow['k']])
    Nk_.append([cpow['modes']])
    Pk_.append([cpow['power'].real])
    Pshot_.append([cpow.attrs['shotnoise']])
    Pln_.append([np.concatenate(spow)[order]])


# == FINALISATION =============================================================

# -- Gather -------------------------------------------------------------------

k_all = np.concatenate(k_)
Nk_all = np.concatenate(Nk_)
Pk_all = np.concatenate(Pk_)
Pshot_all = np.concatenate(Pshot_)
Pln_all = np.concatenate(Pln_)


# -- Export -------------------------------------------------------------------

output = {
    'Nk': Nk_all, 'k': k_all, 'Pk': Pk_all, 'Pshot': Pshot_all,
    'ln': modes, 'kln': waves, 'Pln': Pln_all,
    }
save_data(f"{PATHOUT}{fdir}", f"{fname}{ftag}.npy", output)


# -- Visualise ----------------------------------------------------------------

results = {
    'Nk': np.sum(output['Nk'], axis=0)/2,
    'k': np.average(output['k'], axis=0),
    'Pk': np.average(output['Pk'], axis=0),
    'Pshot': np.average(output['Pshot']),
    'ln': output['ln'],
    'kln': output['kln'],
    'Pln': np.average(output['Pln'], axis=0),
    }
results.update({
    'dk': np.std(output['k'], axis=0, ddof=1),
    'dPk': np.std(output['Pk'], axis=0, ddof=1),
    'dPln': np.std(output['Pln'], axis=0, ddof=1),
    'dof1': np.size(output['k'], axis=0) - 1,
    'dof2': np.size(output['Pln'], axis=0) - 1,
    })

try:
    plt.style.use(harmony)
    plt.close('all')
    plt.figure('No-clustering power recovery')

    c = plt.errorbar(
        results['k'], results['Pk'],
        xerr=results['dk']/np.sqrt(results['dof1']),
        yerr=results['dPk']/np.sqrt(results['dof2']),
        elinewidth=.8, color='#0087BD', label='Cartesian'
        )

    s = plt.loglog(
        results['kln'], results['Pln'], color='#C40233', label='spherical'
        )
    plt.fill_between(
        results['kln'],
        results['Pln']-results['dPln']/np.sqrt(results['dof2']),
        results['Pln']+results['dPln']/np.sqrt(results['dof2']),
        color=s[0].get_color(), alpha=1/4
        )
    plt.fill_between(
        results['kln'],
        results['Pln']-2*results['dPln']/np.sqrt(results['dof2']),
        results['Pln']+2*results['dPln']/np.sqrt(results['dof2']),
        color=s[0].get_color(), alpha=1/20
        )
    for idx, ind_lab in enumerate(results['ln']):
        if ind_lab[0] == 0:
            plt.annotate(
                r'$({:d},{:d})$'.format(ind_lab[0], ind_lab[1]),
                xy=(results['kln'][idx], results['Pln'][idx]),
                verticalalignment='bottom', fontsize=6
                )

    plt.axhline(y=results['Pshot'], ls='--', c=c[0].get_color(), alpha=.5)

    plt.xlim(left=0.99*results['kln'].min(), right=1.01*results['kln'].max())
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
    plt.savefig(f"{PATHOUT}{fdir}{fname}{ftag}.pdf")
except:
    pass
