"""Measure 2-point function values from a suite of realisations.

"""
from collections import defaultdict

import numpy as np
from nbodykit.lab import cosmology as cosmo

from twoptrc import PATHOUT, fdir, fname, params, confirm_dir
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import format_float as ff
from harmonia.mapper import LognormalCatalogue, SphericalMap


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

nbar = params.nbar
contrast = params.contrast
bias = params.bias
redshift = params.redshift
rsd = params.rsd
zmax = params.zmax
kmax = params.kmax
expand = params.expand
meshgen = params.meshgen
meshcal = params.meshcal
niter = params.niter
progid = params.progid

PIVOTS = ['lmn', 'lnm', 'k',]


# -- Cosmology ----------------------------------------------------------------

rmax = cosmo.Planck15.comoving_distance(zmax)
Plin = cosmo.LinearPower(cosmo.Planck15, redshift=redshift, transfer='CLASS')
beta = cosmo.Planck15.scale_independent_growth_rate(redshift) / bias


# -- Program identifier -------------------------------------------------------

try:
    ratio_tag = ff(float(contrast), 'decdot')
except (TypeError, ValueError):
    ratio_tag = 'simu'

if rsd:
    subdir, rsd_tag = "wrsd/", "{:.2f}".format(beta)
else:
    subdir, rsd_tag = "nrsd/", 'none'

ftag = (
    f"-("
    f"nbar={ff(nbar, 'sci')},ratio={ratio_tag},"
    f"b={ff(bias, 'decdot')},rsd={rsd_tag},"
    f"rmax={ff(rmax, 'intdot')},xpd={ff(expand, 'decdot')},"
    f"kmax={ff(kmax, 'sci')},nmesh=c{meshgen},niter={niter}"
    f")-[{progid}]"
    )


# == PROCESSING ===============================================================

print(ftag)

disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)

output = defaultdict(list)
for run in range(niter):
    clog = LognormalCatalogue(
        Plin, nbar, bias=bias, boxsize=expand*2*rmax, nmesh=meshgen,
        add_RSD=rsd
        )
    smap = SphericalMap(disc, clog, nmean_data=nbar)

    for key in PIVOTS:
        output[key].append(smap.twopoint(pivot=key))


# == FINALISATION =============================================================

fpathful, fnameful = f"{PATHOUT}{fdir}{subdir}", f"{fname}{ftag}"
confirm_dir(fpathful)
np.save("".join([fpathful, fnameful, ".npy"]), output)
