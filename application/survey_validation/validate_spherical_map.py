"""Validate spherical modelling.

"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    from application import data_dir
    from harmonia.algorithms import SphericalArray
    from harmonia.cosmology import BaseModel
    from harmonia.mapper import SphericalMap
    from harmonia.reader import Couplings, SphericalCorrelator
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import data_dir
    from harmonia.algorithms import SphericalArray
    from harmonia.cosmology import BaseModel
    from harmonia.mapper import SphericalMap
    from harmonia.reader import Couplings, SphericalCorrelator

# Set inputs.
NG = 0
MAP_SERIALS = range(1, 1 + 24)
MASK_TAG = "random0_BOSS_DR12v5_CMASS_North"
SELECTION_TAG = "[100.0,500.0]"
PIVOT = 'natural'

# Extract spherical powers from spherical maps.
spherical_power = []
for serial_num in MAP_SERIALS:
    spherical_data = SphericalArray.load(
        data_dir/"raw"/"catalpgue_maps"/"catalogue-map-({}).npz".format(
            ",".join([
                f"source=halo-(NG={NG}.,z=1.)-{serial_num}", "map=spherical",
                "scale=[None,0.04]", "orders=None", "rsd=False",
                f"mask={MASK_TAG}", f"selection={SELECTION_TAG}"
            ])
        )
    )

    disc = spherical_data.disc

    # pylint: disable=protected-access
    spherical_map = SphericalMap._from_state({
        'density_contrast': spherical_data.__getstate__(),
        'disc': disc.__getstate__(),
    })

    spherical_power.append(spherical_map.mode_power['mode_powers'])

# Make spherical model predictions.
couplings = Couplings.load(
    data_dir/"processed"/"survey_products"/
    f"couplings-(kmax=0.04,mask={MASK_TAG},selection={SELECTION_TAG}).npz"
)

spherical_model = SphericalCorrelator(
    disc, redshift=1., growth_rate=0., couplings=couplings,
    cosmo=BaseModel(data_dir/"external"/"cosmology"/"simulation.txt")
)

spherical_correlator = spherical_model.correlator_matrix(
    "spectral", b_1=2.35, f_nl=NG, nbar=2.5e-4, contrast=10., radialise=True
)

_, unique_ind = np.unique(
    np.diag(np.around(spherical_correlator, decimals=0)),
    return_index=True
)

sort_order = np.argsort(list(disc.wavenumbers.values()))
normalisations = np.array(list(disc.normalisations.values()))[sort_order]

spherical_spectrum = \
    normalisations * np.diag(spherical_correlator)[np.sort(unique_ind)]

# Compare map powers with model predictions.
wavenumbers = np.array(list(disc.wavenumbers.values()))[sort_order]

plt.loglog(wavenumbers, np.mean(spherical_power, axis=0), label='map power')
plt.loglog(wavenumbers, spherical_spectrum, ls='--', label='model spectrum')
plt.xlabel(r"$k\ \  [h/\mathrm{{Mpc}}]$")
plt.ylabel(r"$P(k)\ \ [(\mathrm{{Mpc}}/h)^3]$")
plt.legend()
