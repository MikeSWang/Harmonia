"""Measure 2-point function values from a suite of realisations.

"""
import numpy as np
from nbodykit.lab import cosmology as cosmo

from twoptrc import PATHOUT, argv, fdir, fname, save_data
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import format_float as ff
from harmonia.mapper import LognormalCatalogue, SphericalMap


# == INITIALISATION ===========================================================

# -- Runtime parameters -------------------------------------------------------

try:
    sarg = iter(argv[1:])
    nbar, contrast = float(next(sarg)), next(sarg)
    bias, rsd = float(next(sarg)), next(sarg)
    zmax, expand = float(next(sarg)), float(next(sarg))
    meshgen, meshcal, niter = int(next(sarg)), int(next(sarg)), int(next(sarg))
    try:
        progid = "-[{}]".format(next(sarg))
    except StopIteration:
        progid = ""
except:
    nbar, contrast = 1e-3, None
    bias, rsd = 2., 'true'
    zmax, expand = 0.05, 2.
    meshgen, meshcal, niter = 256, 256, 25
    progid = ""

if rsd.upper().startswith('T'):
    rsd = True
elif rsd.upper().startswith('F'):
    rsd = False

KMAX = 0.1
Z = 0.
PIVOTS = ['lmn', 'lnm', 'k',]


# -- Cosmology ----------------------------------------------------------------

rmax = cosmo.Planck15.comoving_distance(zmax)
Plin = cosmo.LinearPower(cosmo.Planck15, redshift=Z, transfer='CLASS')
beta = cosmo.Planck15.scale_independent_growth_rate(Z) / bias


# -- Program identifier -------------------------------------------------------

try:
    ratio_tag = ff(float(contrast), 'decdot')
except (TypeError, ValueError):
    ratio_tag = 'simu'

if rsd:
    subdir, rsd_tag = "wrsd/", "{:.2f}".format(beta)
else:
    subdir, rsd_tag = "nrsd/", 'none'

if meshgen == meshcal:
    mesh_tag = f"cp{meshgen}"
else:
    mesh_tag = f"c{meshgen},p{meshcal}"

ftag = (
    f"-(nbar={ff(nbar, 'sci')},ratio={ratio_tag},"
    f"b={ff(bias, 'decdot')},rsd={rsd_tag},"
    f"rmax={ff(rmax, 'intdot')},xpd={ff(expand, 'decdot')},"
    f"nmesh=[{mesh_tag}],niter={niter}){progid}"
    )


# == PROCESSING ===============================================================

print(ftag)

disc = DiscreteSpectrum(rmax, 'Dirichlet', KMAX)

twopt = {key: [] for key in PIVOTS}
for run in range(niter):
    clog = LognormalCatalogue(
        Plin, nbar, boxsize=expand*2*rmax, nmesh=meshgen, bias=bias,
        add_rsd=rsd
        )

    smap = SphericalMap(disc, clog, nmean_data=nbar)

    for key in twopt:
        twopt[key].append(smap.twopoint(pivot=key))


# == FINALISATION =============================================================

save_data(f"{PATHOUT}{fdir}{subdir}{fname}{ftag}.npy", twopt)
