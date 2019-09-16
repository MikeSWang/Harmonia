from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower

from fieldrc import PATHOUT, params, confirm_dir

from catalogue import GaussianCatalogue, LogNormalCatalogue
from harmonia.algorithms.fields import (
    generate_gaussian_random_field,
    generate_lognormal_random_field,
    poisson_sample,
    _cal_isotropic_power_spectrum as cal_power
    )
from harmonia.collections import harmony, format_float as ff

MECHANISM = {
    'gaussian': generate_gaussian_random_field,
    'lognormal': generate_lognormal_random_field
    }
CATALOGUE = {
    'gaussian': GaussianCatalogue,
    'lognormal': LogNormalCatalogue
    }


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

stat = params.stat
cat = not params.noncat
samp = not params.nosmp
nbar = params.nbar
b = params.bias
z = params.redshift
kmax = params.kmax
L = params.boxside
meshg = params.meshgen
meshc = params.meshcal
niter = params.niter
progid = params.progid


# -- Cosmology ----------------------------------------------------------------

Plin = cosmology.LinearPower(cosmology.Planck15, redshift=z, transfer='CLASS')


# -- Program identifier -------------------------------------------------------

if samp:
    fname = "catalogue"
else:
    fname = "realisation"
if stat.lower().startswith('g'):
    fname += "-gaussian"
elif stat.lower().startswith('l'):
    fname += "-lognormal"

if meshg == meshc:
    mesh_tag = f"cp{meshg}"
else:
    mesh_tag = f"c{meshg},p{meshc}"

ftag = (
    f"-("
    f"nbar={ff(nbar, 'sci')},b={ff(b, 'decdot')},size={ff(L, 'intdot')},"
    f"kmax={ff(kmax, 'sci')},nmesh=[{mesh_tag}],niter={niter}"
    f")-[{progid}]"
    )


# == PROCESSING ===============================================================

print(fname, ftag.lstrip("-"))

suite = defaultdict(list)
if cat:
    for run in range(niter):
        clog = CATALOGUE[stat](Plin, nbar, bias=b, BoxSize=L, Nmesh=meshg)
        mesh = clog.to_mesh(Nmesh=meshc, resampler='tsc', compensated=True)
        cpow = FFTPower(mesh, mode='1d', kmax=kmax).power

        suite['k'].append([cpow['k']])
        suite['Nk'].append([cpow['modes']])
        suite['Pshot'].append([cpow.attrs['shotnoise']])
        suite['Pk'].append([cpow['power'].real])
else:
    for run in range(niter):
        field = MECHANISM[stat](L, meshg, Plin, bias=b)
        if samp: field = poisson_sample(field, nbar, L)
        k, Pk, Nk = cal_power(field, L, kmax=kmax)

        suite['k'].append([k])
        suite['Nk'].append([Nk])
        suite['Pshot'].append([0.])
        suite['Pk'].append([Pk])


# == FINALISATION =============================================================

fpathful, fnameful = PATHOUT + fname + "/", fname + ftag
assert confirm_dir(fpathful)


# -- Export -------------------------------------------------------------------

output = {var: np.concatenate(val_list) for var, val_list in suite.items()}

np.save("".join([fpathful, fnameful, ".npy"]), output)


# -- Visualise ----------------------------------------------------------------

results = {
    'Nk': np.sum(output['Nk'], axis=0),
    'k': np.average(output['k'], axis=0),
    'Pk': np.average(output['Pk'], axis=0),
    'Pshot': np.average(output['Pshot']),
    }
results.update({
    'Pkmod': b**2 * Plin(results['k']) + results['Pshot'],
    'dk': np.std(output['k'], axis=0, ddof=1),
    'dPk': np.std(output['Pk'], axis=0, ddof=1),
    'dof1': np.size(output['k'], axis=0) - 1,
    'dof2': np.size(output['Pk'], axis=0) - 1,
    })

try:
    plt.style.use(harmony)
    plt.close('all')

    plt.loglog(results['k'], results['Pkmod'], label='model')
    plt.errorbar(
        results['k'], results['Pk'],
        xerr=results['dk']/np.sqrt(results['dof1']),
        yerr=results['dPk']/np.sqrt(results['dof2']),
        elinewidth=.8, label='catalogue'
        )

    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
    plt.savefig("".join([fpathful, fnameful, ".pdf"]))
except Exception as e:
    print(e)
