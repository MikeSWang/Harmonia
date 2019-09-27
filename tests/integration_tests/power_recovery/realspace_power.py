"""Recover real-space power spectrum from fixed-epoch log-normal catalogues.

"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower, ConvolvedFFTPower, FKPCatalog

from powerrc import PATHOUT, confirm_dir, filename, params, quick_plot
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import format_float, harmony
from harmonia.mapper import NBKCatalogue, RandomCatalogue, SphericalMap


# == INITIALISATION ===========================================================

nbar = params.nbar
contrast = params.contrast
bias = params.bias
redshift = params.redshift
zmax = params.zmax
kmax = params.kmax
dk = params.dk
expand = params.expand
meshgen = params.meshgen
meshcal = params.meshcal
niter = params.niter
progid = params.progid

cosmo = cosmology.Planck15
rmax = cosmo.comoving_distance(zmax)
Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')

boxsize = 2 * expand * rmax
to_mesh_params = dict(Nmesh=meshcal, resampler='tsc', compensated=True)

try:
    ratio_tag = format_float(float(contrast), 'decdot')
    is_case_mock = True
except (TypeError, ValueError):
    ratio_tag = "simu"
    is_case_mock = False

if meshgen == meshcal:
    mesh_tag = f"gc{meshgen}"
else:
    mesh_tag = f"g{meshgen},c{meshcal}"

prog_tag = "-(nbar={},contrast={},rmax={},kmax={},xpd={},mesh=[".format(
    format_float(nbar, 'sci'),
    ratio_tag,
    format_float(rmax, 'intdot'),
    format_float(kmax, 'sci'),
    format_float(expand, 'decdot'),
)
prog_tag += mesh_tag
prog_tag += "],iter={})-[".format(niter)
prog_tag += progid
prog_tag += "]"

print(prog_tag.lstrip("-"))


# == PROCESSING ===============================================================

disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
order = np.concatenate(disc.wavenumbers).argsort()
waves = np.concatenate(disc.wavenumbers)[order]
modes = np.concatenate(disc.dbl_indices)[order]

measurements = defaultdict(list)
for run in range(niter):
    data_catalogue = NBKCatalogue(
        Plin,
        nbar,
        bias=bias,
        BoxSize=boxsize,
        Nmesh=meshgen,
    )
    if is_case_mock:
        rand_catalogue = RandomCatalogue(contrast*nbar, boxsize)
        data_catalogue['NZ'] = nbar * data_catalogue['Weight']
        rand_catalogue['NZ'] = nbar * rand_catalogue['Weight']
        pair_catalogue = FKPCatalog(data_catalogue, rand_catalogue)
    else:
        rand_catalogue = None

    if is_case_mock:
        mesh = pair_catalogue.to_mesh(**to_mesh_params)
        cartesian_power = ConvolvedFFTPower(
            mesh,
            poles=[0],
            dk=dk,
            kmax=waves.max()+dk,
        ).poles
    else:
        mesh = data_catalogue.to_mesh(**to_mesh_params)
        cartesian_power = FFTPower(
            mesh,
            mode='1d',
            dk=dk,
            kmax=waves.max()+dk,
        ).power

    spherical_map = SphericalMap(
        disc,
        data_catalogue,
        rand=rand_catalogue,
        mean_density_data=nbar,
        mean_density_rand=contrast*nbar,
    )
    spherical_power = spherical_map.spherical_power()

    measurements['k'].append([cartesian_power['k']])
    measurements['Nk'].append([cartesian_power['modes']])
    measurements['Pshot'].append([cartesian_power.attrs['shotnoise']])
    measurements['Pln'].append([np.concatenate(spherical_power)[order]])
    if is_case_mock:
        measurements['Pk'].append([cartesian_power['power_0'].real])
    else:
        measurements['Pk'].append([cartesian_power['power'].real])


# == FINALISATION =============================================================

base_path, root_name = f"{PATHOUT}{filename}", f"{filename}{prog_tag}"
assert confirm_dir(base_path)

output = {var: np.concatenate(vals) for var, vals in measurements.items()}
output.update({'ln': [modes], 'kln': [waves]})
np.save("".join([base_path, "/", root_name, ".npy"]), output)

try:
    plt.style.use(harmony)
    plt.close('all')
    quick_plot(output)
    plt.savefig("".join([base_path, "/", root_name, ".pdf"]))
except Exception as e:
    print(e)
