"""Export recovered power spectra.

"""
import numpy as np
from matplotlib import pyplot as plt

from powerrc import PATHOUT, aggregate, confirm_dir
from view_power import view_spectrum
from harmonia.collections import collate


# == CONFIGURATION ============================================================

SCRIPT = "nbodysim_power"
PREFIX = "halos-(NG=0.,z=1.)"
TAG = "-(nbar=2.49e-4,bias=2.3415,kmax=0.04,boxside=1000.,nmesh=256)"
TAG_ADD = None

COLLATE = True
LOAD = False
AGGREGATE = True

EXPORT = True
SAVE = False
SAVEFIG = False


# == OPERATION ================================================================

fpath = f"{PATHOUT}{SCRIPT}/"
confirm_dir(fpath)

if COLLATE:
    output, count, _ = collate(f"{fpath}{PREFIX}*.npy", 'npy')
    if SAVE:
        confirm_dir(fpath+"collated/")
        np.save(f"{fpath}collated/{PREFIX}{TAG}.npy", output)
    if AGGREGATE:
        results = aggregate(output)

if LOAD and (TAG is not None):
    output = np.load(f"{fpath}collated/{PREFIX}{TAG}.npy").item()
    if AGGREGATE:
        results = aggregate(output)

if LOAD and (TAG_ADD is not None):
    output_add = np.load(f"{fpath}collated/{PREFIX}{TAG_ADD}.npy").item()
    if AGGREGATE:
        results_add = aggregate(output_add)

if EXPORT:
    view_spectrum(results, case='error')
    if SAVEFIG: plt.savefig(f"{fpath}{PREFIX}{TAG}.pdf")
