import numpy as np
from matplotlib import pyplot as plt
from nbodykit import cosmology

from fieldrc import PATHOUT, harmony, collate_and_save


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

COLLATE = False
LOAD = True
SAVEFIG = False

NBAR = 0.0005
NMESH = 256

BIAS = 2.
LSIDE = 1000.
NITER = 1000

PREFIX = "catalogue-lognormal"
TAG = "-(nbar=0.0005,b=2.,size=1000.,kmax=0.1,nmesh=[c256,p128],niter=1000)"
if str(NMESH) not in TAG or str(NBAR) not in TAG or str(NITER) not in TAG:
    cont = input("Mesh number/number density mismatch! Continue? [y/n] ")
    if cont.lower().startswith('n'):
        raise InterruptedError("User stopped execution. ")


# == OPERATION ================================================================

savepath = PATHOUT + PREFIX + "/"
savename = PREFIX + TAG + ".npy"

if COLLATE:
    output, count = collate_and_save(
        pattern=savepath+PREFIX+f"*{NBAR}*{NMESH}*.npy", extension='npy',
        opath=savepath+"collated/", oname=savename
        )
    results = aggregate(output)

if LOAD:
    output = np.load(savepath+"collated/"+savename).item()
    results = aggregate(output)


# == PRODUCTION ===============================================================

sel, dof = slice(1, None), results['dof']
k = results['k'][sel]
Nk = results['Nk'][sel]
Pk = results['Pk'][sel]  # - results['Pshot']
dPk = results['dPk'][sel] / np.sqrt(dof)

Plin = cosmology.LinearPower(cosmology.Planck15, redshift=0., transfer='CLASS')
Pshot = 1 / NBAR
Pmod = BIAS**2 * Plin(k) + Pshot
Perr = Pmod / np.sqrt(Nk)

deviat = Pk / Pmod - 1
if np.abs(np.average(deviat)) > 0.05:
    missing = input("Did you forget to add/subtract shot noise? [y/n] ")
    if missing.lower().startswith('y'):
        action = input("Do you want to add/subtract shot noise? [a/s] ")
        if action.lower().startswith('a'):
            Pmod += Pshot
        elif action.lower().startswith('s'):
            Pmod -= Pshot
        deviat = Pk / Pmod - 1

xlim = (0.95*min(k), 1.05*max(k))
YLIM = (22000, 120000)
EPANEL_CL = 0.05
EPATCH_HT = 0.01

plt.style.use(harmony)
plt.close('all')

fig = plt.figure("Catalogue fidelity")

mainax = plt.subplot2grid((5,6), (0,0), rowspan=4, colspan=6)

plt.errorbar(k, Pmod, Perr, elinewidth=.8, ls='--', label='model')
plt.errorbar(k, Pk, dPk, label=' '.join(list(reversed(PREFIX.split("-")))))

plt.tick_params(axis='x', which='both', labelbottom=False)
plt.xscale('log')
plt.yscale('log')
plt.ylim(YLIM)
plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
plt.legend()

subax = plt.subplot2grid((5,6), (4,0), rowspan=1, colspan=6, sharex=mainax)

plt.plot(k, deviat, ls='--')   # c='#C40233', c='#0087BD'
plt.axhline(y=0., lw=1., ls='--')
plt.fill_between(xlim, [EPATCH_HT,]*2, [-EPATCH_HT]*2, alpha=0.2)

plt.xlim(xlim)
plt.ylim(-EPANEL_CL, EPANEL_CL)
plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
plt.ylabel(r'$\hat{P} \big/ P_\mathrm{model}(k) - 1$')

fig.subplots_adjust(hspace=0)
fig.suptitle(
    f"Suite of size {NITER}, boxsize {LSIDE}, mesh number {NMESH}, "
    r"density $\bar{{n}} = {} \times 10^{{-{}}}$".format(
        "{:.1e}".format(NBAR).split("e")[0],
        "{:.1e}".format(NBAR).split("e")[1].lstrip("-").lstrip("0")
        ),
    fontsize=12
    )

if SAVEFIG:
    plt.savefig(f"{PATHOUT}{PREFIX}{TAG}.pdf")
