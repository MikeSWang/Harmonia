import numpy as np
from matplotlib import pyplot as plt

from powerrc import PATHOUT
from harmonia.collections import collate, harmony


def aggregate(result):
    return {
        'Nk': np.sum(result['Nk'], axis=0),
        'k': np.average(result['k'], axis=0),
        'Pk': np.average(result['Pk'], axis=0),
        'Pshot': np.average(result['Pshot']),
        'dPk': np.std(result['Pk'], axis=0, ddof=1),
        'dof': np.size(result['Pk'], axis=0) - 1,
        }


PREFIX = "suite"
TAG = "-(nbar=0.001,ratio=simu,rmax=148.,xpd=2.,nmesh=[cp256],niter=1000)"

COLLATE = True
LOAD = False
SAVE = True
SAVEFIG = False

if COLLATE:
    output, count, _ = collate(f"{PATHOUT}{PREFIX}/{PREFIX}-*.npy", 'npy')
    if SAVE:
        np.save(f"{PATHOUT}{PREFIX}/collated/{PREFIX}{TAG}.npy", output)
    results = aggregate(output)

if LOAD and (TAG is not None):
    output = np.load(f"{PATHOUT}{PREFIX}/collated/{PREFIX}{TAG}.npy").item()
    results = aggregate(output)

plt.style.use(harmony)
# ...

if SAVEFIG:
    plt.savefig(f"{PATHOUT}{PREFIX}/{PREFIX}{TAG}.pdf")