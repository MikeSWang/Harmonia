from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower

from fieldrc import PATHOUT, params, fname, confirm_dir

from catalogue_of_field import GaussianCatalogue, LogNormalCatalogue
from harmonia.collections import harmony, format_float as ff


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

stat = params.stat
nbar = params.nbar
bias = params.bias
redshift = params.redshift
rmax = params.boxside
kmax = params.kmax
meshgen = params.meshgen
meshcal = params.meshcal
niter = params.niter
progid = params.progid


# -- Cosmology ----------------------------------------------------------------

cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')
catalogue = {
    'gaussian': GaussianCatalogue,
    'lognormal': LogNormalCatalogue
    }


# -- Program identifier -------------------------------------------------------

fname = fname.split("_")[0]
if stat.lower().startswith('g'):
    fname += "-gaussian"
elif stat.lower().startswith('l'):
    fname += "-lognormal"

if meshgen == meshcal:
    mesh_tag = f"cp{meshgen}"
else:
    mesh_tag = f"c{meshgen},p{meshcal}"

ftag = (
    f"-("
    f"nbar={ff(nbar, 'sci')},b={ff(bias, 'decdot')},rmax={ff(rmax, 'intdot')},"
    f"kmax={ff(kmax, 'sci')},nmesh=[{mesh_tag}],niter={niter}"
    f")-[{progid}]"
    )


# == PROCESSING ===============================================================

print(fname, ftag)

suite = defaultdict(list)
for run in range(niter):
    clog = catalogue[stat](
        Plin, nbar, bias=bias, BoxSize=2*rmax, Nmesh=meshgen
        )
    mesh = clog.to_mesh(Nmesh=meshcal, resampler='tsc', compensated=True)
    cpow = FFTPower(mesh, mode='1d', kmax=kmax).power

    suite['k'].append([cpow['k']])
    suite['Nk'].append([cpow['modes']])
    suite['Pshot'].append([cpow.attrs['shotnoise']])
    suite['Pk'].append([cpow['power'].real])


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
    'Pkmod': bias**2 * Plin(results['k']) + results['Pshot'],
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
