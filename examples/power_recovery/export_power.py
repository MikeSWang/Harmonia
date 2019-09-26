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
TAG = "-(nbar=2.49e-4,bias=2.3415,kmax=0.04,boxsize=1000.,mesh=256)"

COLLATE = True
LOAD = False
AGGREGATE = True

EXPORT = True
SAVE = False
SAVEFIG = False


# == OPERATION ================================================================

outpath = f"{PATHOUT}{SCRIPT}/"
assert confirm_dir(outpath)

if COLLATE:
    output, count, _ = collate(f"{outpath}{PREFIX}*.npy", 'npy')
    if SAVE:
        assert confirm_dir(outpath + "collated/")
        np.save(f"{outpath}collated/{PREFIX}{TAG}.npy", output)
    if AGGREGATE:
        results = aggregate(output)

if LOAD and (TAG is not None):
    output = np.load(f"{outpath}collated/{PREFIX}{TAG}.npy").item()
    if AGGREGATE:
        results = aggregate(output)

if EXPORT:
    view_spectrum(results, case='error')
    if SAVEFIG: plt.savefig(f"{outpath}{PREFIX}{TAG}.pdf")
