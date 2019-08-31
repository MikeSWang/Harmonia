"""Recover real-space power spectrum from fixed-epoch log-normal catalogues.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower, ConvolvedFFTPower, FKPCatalog

from powerrc import PATHOUT, fdir, fname, params, confirm_dir
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import harmony, format_float as ff
from harmonia.mapper import SphericalMap, LognormalCatalogue, RandomCatalogue


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

nbar = params.nbar
contrast = params.contrast
bias = params.bias
redshift = params.redshift
zmax = params.zmax
kmax = params.kmax
dk = params.dk
expand = params.expand
meshgen = params.meshgen
meshcal = params.meshcal
niter = params.niter
progid = params.progid

# -- Cosmology ----------------------------------------------------------------

cosmo = cosmology.Planck15
rmax = cosmo.comoving_distance(zmax)
Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')


# -- Program identifier -------------------------------------------------------

try:
    ratio_tag, is_case_mock = ff(float(contrast), 'decdot'), True
except (TypeError, ValueError):
    ratio_tag, is_case_mock = 'simu', False

if meshgen == meshcal:
    mesh_tag = f"cp{meshgen}"
else:
    mesh_tag = f"c{meshgen},p{meshcal}"

ftag = (
    f"-("
    f"nbar={ff(nbar, 'sci')},ratio={ratio_tag},rmax={ff(rmax, 'intdot')},"
    f"xpd={ff(expand, 'decdot')},nmesh=[{mesh_tag}],niter={niter}"
    f"){progid}"
    )


# == PROCESSING ===============================================================

print(ftag)

# Set up discretisation.
disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
order = np.concatenate(disc.wavenumbers).argsort()
modes = np.concatenate(disc.waveindices)[order]
waves = np.concatenate(disc.wavenumbers)[order]

suite = defaultdict(list)
for run in range(niter):
    # Generate data and random catalogues.
    data = LognormalCatalogue(
        Plin, nbar, bias=bias, boxsize=2*expand*rmax, nmesh=meshgen
        )
    if is_case_mock:
        nrand = contrast*nbar
        rand = RandomCatalogue(nrand, boxsize=2*expand*rmax)
        data['NZ'] = nbar * data['Weight']
        rand['NZ'] = nbar * rand['Weight']
        fkplog = FKPCatalog(data, rand)
    else:
        rand, nrand = None, None

    # Run Cartesian algorithm.
    if is_case_mock:
        mesh = fkplog.to_mesh(Nmesh=meshcal, resampler='tsc', compensated=True)
        cpow = ConvolvedFFTPower(
            mesh, poles=[0], dk=dk, kmax=waves.max()+dk
            ).poles
    else:
        mesh = data.to_mesh(Nmesh=meshcal, resampler='tsc', compensated=True)
        cpow = FFTPower(mesh, mode='1d', dk=dk, kmax=waves.max()+dk).power

    # Run spherical algorithm.
    mapp = SphericalMap(
        disc, data, rand=rand, nmean_data=nbar, nmean_rand=nrand
        )
    spow = mapp.spherical_power()

    # Append reordered results.
    suite['k'].append([cpow['k']])
    suite['Nk'].append([cpow['modes']]/2)
    suite['Pshot'].append([cpow.attrs['shotnoise']])
    suite['Pln'].append([np.concatenate(spow)[order]])
    if is_case_mock:
        suite['Pk'].append([cpow['power_0'].real])
    else:
        suite['Pk'].append([cpow['power'].real])


# == FINALISATION =============================================================

fpathful, fnameful = f"{PATHOUT}{fdir}", f"{fname}{ftag}"
confirm_dir(fpathful)

# -- Export -------------------------------------------------------------------

output = {var: np.concatenate(val_list) for var, val_list in suite.iteritems()}
output.update({'ln': modes, 'kln': waves})

np.save("".join([fpathful, fnameful, ".npy"]), output)

# -- Visualise ----------------------------------------------------------------

results = {
    'Nk': np.sum(output['Nk'], axis=0),
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

    plt.xlim(left=0.99*results['kln'].min(), right=1.01*results['kln'].max())
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
    plt.savefig("".join([fpathful, fnameful, ".pdf"]))
except Exception as e:
    print(e)
