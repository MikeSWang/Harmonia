import sys, os
import warnings
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower

sys.path.insert(0, "../")

from catalogue import GaussianCatalogue, LogNormalCatalogue
from harmonia.collections import harmony, format_float as ff


def get_filename():
    return os.path.splitext(os.path.basename(sys.argv[0]))[0]


def confirm_dir(dirpath):
    if not dirpath.endswith("/"):
        dirpath += "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def clean_warnings(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def _setup_params(parser):

    parser.add_argument('--stat', required=True)
    parser.add_argument('--nbar', type=float, default=1e-3)
    parser.add_argument('--bias', type=float, default=2.)
    parser.add_argument('--redshift', type=float, default=0.)
    parser.add_argument('--zmax', type=float, default=0.05)
    parser.add_argument('--kmax', type=float, default=0.1)
    parser.add_argument('--meshgen', type=int, default=256)
    parser.add_argument('--meshcal', type=int, default=256)
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--progid', default="")

    return parser.parse_args()


# == INITIALISATION ===========================================================

PATHOUT = "./data/output/"

warnings.formatwarning = clean_warnings

# -- Runtime parameters -------------------------------------------------------

params = _setup_params(ArgumentParser(
    description="Simulate a suite of log-normal catalogues."
    ))

stat = params.stat
nbar = params.nbar
bias = params.bias
redshift = params.redshift
zmax = params.zmax
kmax = params.kmax
meshgen = params.meshgen
meshcal = params.meshcal
niter = params.niter
progid = params.progid

# -- Cosmology ----------------------------------------------------------------

cosmo = cosmology.Planck15
rmax = cosmo.comoving_distance(zmax)
Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')

# -- Program identifier -------------------------------------------------------

if stat.upper().startswith('G'):
    fname = "gaussian"
elif stat.upper().startswith('L'):
    fname = "lognormal"

if meshgen == meshcal:
    mesh_tag = f"cp{meshgen}"
else:
    mesh_tag = f"c{meshgen},p{meshcal}"

ftag = (
    f"-{fname}"
    f"-("
    f"nbar={ff(nbar, 'sci')},b={ff(bias, 'decdot')},rmax={ff(rmax, 'intdot')},"
    f"kmax={ff(kmax, 'sci')},nmesh=[{mesh_tag}],niter={niter}"
    f")-[{progid}]"
    )


# == PROCESSING ===============================================================

print(ftag)

suite = defaultdict(list)
count_invalid = 0
for run in range(niter):
    if stat.upper().startswith('G'):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('ignore')
                warnings.filterwarnings(
                    'always', message=".*clipped.*", category=RuntimeWarning
                    )
                clog = GaussianCatalogue(
                    Plin, nbar, bias=bias, BoxSize=2*rmax, Nmesh=meshgen
                    )
                count_invalid += len(w)
        except RuntimeError as e:
            print(e)
            pass
    elif stat.upper().startswith('L'):
        clog = LogNormalCatalogue(
            Plin, nbar, bias=bias, BoxSize=2*rmax, Nmesh=meshgen
            )
    mesh = clog.to_mesh(Nmesh=meshcal, resampler='tsc', compensated=True)
    cpow = FFTPower(mesh, mode='1d', kmax=kmax).power

    # Append reordered results.
    suite['k'].append([cpow['k']])
    suite['Nk'].append([cpow['modes']])
    suite['Pshot'].append([cpow.attrs['shotnoise']])
    suite['Pk'].append([cpow['power'].real])

print("{:d} of {:d} runs discarded. ".format(count_invalid, niter))


# == FINALISATION =============================================================

fpathful, fnameful = f"{PATHOUT}", f"{get_filename()}{ftag}"
confirm_dir(fpathful)

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
