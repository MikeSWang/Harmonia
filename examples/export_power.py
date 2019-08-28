"""Export recovered power spectra.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from powerrc import PATHOUT
from harmonia.collections import collate, harmony


# == DEFINITION ===============================================================

# HINT: Implementation of old versions of codes denotes by OV.
def aggregate(result):

    if 'count' not in globals():
        REDUCE = False
    else:
        REDUCE = True

    if REDUCE:
        try:
            reduced_len = int(len(result['modes']) / count)
            result['modes'] = result['modes'][:reduced_len]
            result['k_ln'] = result['k_ln'][:reduced_len]
            if result['modes'].ndim == 3:  # HACK: OV
                result['modes'] = np.squeeze(result['modes'])
        except KeyError:
            reduced_len = int(len(result['ln']) / count)
            result['ln'] = result['ln'][:reduced_len]
            result['kln'] = result['kln'][:reduced_len]
            if result['ln'].ndim == 3:  # HACK: OV
                result['ln'] = np.squeeze(result['ln'])

    if False:  # HACK: OV
        return {
            'Nk': np.sum(result['count_in_bins'], axis=0)/2,  # !!!: OV ``/2``
            'k': np.average(result['k'], axis=0),
            'Pk': np.average(result['P'], axis=0),
            'Pshot': np.average(result['P_shot']),
            'ln': result['modes'][:reduced_len],
            'kln': np.average(result['k_ln'], axis=0),
            'Pln': np.average(result['P_ln'], axis=0),
            'dk': np.std(result['k'], axis=0, ddof=1),
            'dPk': np.std(result['P'], axis=0, ddof=1),
            'dPln': np.std(result['P_ln'], axis=0, ddof=1),
            'dof1': np.size(result['P'], axis=0) - 1,
            'dof2': np.size(result['P_ln'], axis=0) - 1,
            }
    # !!!: OV ``[2:]``
    return {
        'Nk': np.sum(result['Nk'], axis=0)[2:]/2,  # !!!: OV ``/2``
        'k': np.average(result['k'], axis=0)[2:],
        'Pk': np.average(result['Pk'], axis=0)[2:],
        'Pshot': np.average(result['Pshot']),
        'ln': np.atleast_2d(result['ln']),
        'kln': np.atleast_2d(result['kln'])[0],
        'Pln': np.average(result['Pln'], axis=0),
        'dk': np.std(result['k'], axis=0, ddof=1)[2:],
        'dPk': np.std(result['Pk'], axis=0, ddof=1)[2:],
        'dPln': np.std(result['Pln'], axis=0, ddof=1),
        'dof1': np.size(result['Pk'][2:], axis=0) - 1,
        'dof2': np.size(result['Pln'], axis=0) - 1,
        }


# == EXECUTION ================================================================

PREFIX = "realspace_power"  # also specifies `DIR`

TAG = "-(nbar=0.001,ratio=simu,rmax=148.,xpd=2.,nmesh=[cp256],niter=4000)"
TAG_ADD = None

COLLATE = True
LOAD = False
AGGREGATE = True   # `False` for ``nbodysim`` results, see ``nbodysim``

EXPORT = True
ERROROFF = False  # for ``nbodysim`` results, see ``nbodysim``

SAVE = False
SAVEFIG = False

if COLLATE:
    output, count, _ = collate(f"{PATHOUT}{PREFIX}/{PREFIX}-*.npy", 'npy')
    if SAVE:
        np.save(f"{PATHOUT}{PREFIX}/{PREFIX}{TAG}.npy", output)
    if AGGREGATE: results = aggregate(output)

if LOAD and (TAG is not None):
    output = np.load(f"{PATHOUT}{PREFIX}/collated/{PREFIX}{TAG}.npy").item()
    if AGGREGATE: results = aggregate(output)

if LOAD and (TAG_ADD is not None):
    output_add = np.load(f"{PATHOUT}{PREFIX}/collated/{PREFIX}{TAG_ADD}.npy"
                         ).item()
    if AGGREGATE: results_add = aggregate(output_add)

if EXPORT:
    from view_power import view_spectrum

    if ERROROFF:  # for ``nbodysim`` results, see ``nbodysim``
        for dataset in [results, results_add]:
            dataset['dof1'] = 1
            dataset['dof2'] = 1
            dataset['dk'] = np.zeros(np.shape(dataset['k']))
            dataset['dPk'] = np.zeros(np.shape(dataset['Pk']))
            dataset['dPln'] = np.zeros(np.shape(dataset['Pln']))

    plt.style.use(harmony)
    plt_config = {
        'title': None, #r'$\times2R$',
        'title_add': None, #r'$\times2R$'
        }

    view_spectrum(results, config=plt_config, data_add=None, case='error')
    if SAVEFIG: plt.savefig(f"{PATHOUT}{PREFIX}-{TAG}.pdf")
