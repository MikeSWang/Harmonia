from os import path

import numpy as np
from matplotlib import pyplot as plt
from nbodykit import cosmology

from fieldrc import PATHOUT, confirm_dir
from harmonia.collections import collate, harmony


def aggregate(result):
    return {
        'Nk': np.sum(result['Nk'], axis=0),
        'k': np.average(result['k'], axis=0),
        'Pk': np.average(result['Pk'], axis=0),
        'Pshot': np.average(result['Pshot']),
        'dPk': np.std(result['Pk'], axis=0, ddof=1),
        'dof': np.size(result['k'], axis=0) - 1,
        }


# == CONFIGURATION ============================================================

PREFIX = "ensemble-"
STAT = "lognormal"
TAG = "-(nbar=0.0001,b=2.,rmax=1000.,kmax=0.1,nmesh=[cp128],niter=1000)"

COLLATE = True
LOAD = False
SAVE = True
SAVEFIG = False

NBAR = 1e-3
BIAS = 2.

prefix = PREFIX + STAT
savepath = f"{PATHOUT}{prefix}/"
savename = f"{prefix}{TAG}.npy"
collpath = savepath + "collated/"


# == OPERATION ================================================================

if COLLATE:
    output, count, _ = collate(f"{savepath}{prefix}*.npy", 'npy')
    if SAVE:
        assert confirm_dir(collpath)
        if path.exists(collpath + savename):
            raise FileExistsError(
                "Saving would overwrite existing file at destination. "
                "Please change file saving path. "
                )
        np.save(collpath + savename, output)
    results = aggregate(output)

if LOAD:
    output = np.load(collpath + savename).item()
    results = aggregate(output)


# == PRODUCTION ===============================================================

sel = slice(1, None)
ddof = results['dof']
k = results['k'][sel]
Pk = results['Pk'][sel]
dPk = results['dPk'][sel] / np.sqrt(ddof)

Plin = cosmology.LinearPower(cosmology.Planck15, redshift=0, transfer='CLASS')
Pmod = BIAS**2 * Plin(k) + 1 / NBAR

ratio = Pk / Pmod
epatch_ht = 0.05

plt.style.use(harmony)
plt.close('all')

fig = plt.figure("Catalogue fidelity")

mainax = plt.subplot2grid((5,6), (0,0), rowspan=4, colspan=6)

plt.loglog(k, Pmod, '--', label='model')
plt.errorbar(k, Pk, dPk, elinewidth=.8, label=f'{STAT} catalogue')

plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
plt.legend()

subax = plt.subplot2grid((5,6), (4,0), rowspan=1, colspan=6, sharex=mainax)
xlim = mainax.get_xlim()

plt.plot(k, ratio-1, '--')
plt.axhline(y=0, lw=1, ls='--')
plt.fill_between(xlim, [epatch_ht,]*2, [-epatch_ht]*2, alpha=0.2)

plt.xlim(xlim)
plt.ylim(-0.2, 0.2)
plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
plt.ylabel(r'$\hat{P} \big/ P_\mathrm{model}(k) - 1$')

plt.setp(mainax.get_xticklabels(), visible=False)
fig.subplots_adjust(hspace=0)
fig.suptitle(
    'Suite of size 1000, boxsize 1000, mesh number 128, '
    r'density $\bar{n} = 10^{-3}$',
    fontsize=12
    )

if SAVEFIG:
    plt.savefig(f"{PATHOUT}{prefix}{TAG}.pdf")
