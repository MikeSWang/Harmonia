"""Export recovered power spectra.

"""
import numpy as np
from matplotlib import pyplot as plt

from powerrc import PATHOUT, aggregate, confirm_dir
from view_power import view_spectrum
from harmonia.collections import collate


# == CONFIGURATION ============================================================

SCRIPT = "realspace_power"
TAG = "-(nbar=0.001,ratio=simu,rmax=148.,xpd=2.,nmesh=[cp256],niter=)"
TAG_ADD = None

COLLATE = True
LOAD = False
AGGREGATE = True   # `False` for ``nbodysim_power`` results

EXPORT = True
SAVE = False
SAVEFIG = False


# == OPERATION ================================================================

fpath = f"{PATHOUT}{SCRIPT}/"
confirm_dir(fpath)

if COLLATE:
    output, count, _ = collate(f"{fpath}{SCRIPT}*.npy", 'npy')
    if SAVE:
        np.save(f"{fpath}{SCRIPT}{TAG}.npy", output)
    if AGGREGATE:
        results = aggregate(output)

if LOAD and (TAG is not None):
    output = np.load(f"{fpath}collated/{SCRIPT}{TAG}.npy").item()
    if AGGREGATE:
        results = aggregate(output)

if LOAD and (TAG_ADD is not None):
    output_add = np.load(f"{fpath}collated/{SCRIPT}{TAG_ADD}.npy").item()
    if AGGREGATE:
        results_add = aggregate(output_add)

if EXPORT:
    view_spectrum(results, case='error')
    if SAVEFIG: plt.savefig(f"{fpath}-{TAG}.pdf")
