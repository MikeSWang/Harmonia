"""Measure 2-point function values for a suite of realisations.

"""
import numpy as np
from nbodykit.lab import cosmology as cosmo

from twopointrc import PATHOUT, argv, fdir, fname
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
except:
    nbar, contrast = 1e-3, None
    bias, rsd = 2., 'true'
    zmax, expand = 0.05, 2.
    meshgen, meshcal, niter = 256, 256, 25

    argv.extend([
        str(nbar), str(contrast).lower(), str(bias), str(rsd), str(zmax),
        str(expand), str(meshgen), str(meshcal), str(niter)
        ])

nargs = 1 + 9

if rsd.upper().startswith('T'):
    rsd = True
elif rsd.upper().startswith('F'):
    rsd = False

if argv[nargs:]:
    progid = "-[{}]".format(argv[-1])
else:
    progid = ""

KMAX = 0.1
Z = 0.
PIVOTS = ['natural', 'lnm', 'k',]


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
    f"(nbar={ff(nbar, 'sci')},ratio={ratio_tag},"
    f"b={ff(bias, 'decdot')},rsd={rsd_tag},"
    f"rmax={ff(rmax, 'intdot')},xpd={ff(expand, 'decdot')},"
    f"nmesh=[{mesh_tag}],niter={niter}){progid}"
    )


# == PROCESSING ===============================================================

print(ftag)

disc = DiscreteSpectrum(rmax, 'fdirichlet', KMAX)

twopt_ = {key: [] for key in PIVOTS}
for run in range(niter):
    # Generate rsd catalogues.
    clog = LognormalCatalogue(
        Plin, nbar, boxsize=expand*2*rmax, nmesh=meshgen, bias=bias,
        add_rsd=rsd
        )

    # Run spherical transform.
    smap = SphericalMap(disc, clog, nmean_data=nbar)
    n_ellmn, nbar_ellmn = smap.transform()

    # Append computed quantities.
    for key in twopt_:
        twopt_[key].append(smap.twopoint(pivot=key))


# == FINALISATION =============================================================

output_2pt = twopt_
np.save(f"{PATHOUT}{fdir}2pt/{subdir}{fname}-2pt-{ftag}.npy", output_2pt)
