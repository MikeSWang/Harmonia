"""Validate spherical likelihood evaluations on a coarse grid.

"""
import os
import sys

import seaborn as sns

try:
    from application import data_dir
    from harmonia.algorithms import SphericalArray
    from harmonia.cosmology import BaseModel
    from harmonia.reader import Couplings, LogLikelihood, SphericalCorrelator
    from harmonia.surveyor.synthesis import generate_compression_matrix
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import data_dir
    from harmonia.algorithms import SphericalArray
    from harmonia.cosmology import BaseModel
    from harmonia.reader import Couplings, LogLikelihood, SphericalCorrelator
    from harmonia.surveyor.synthesis import generate_compression_matrix

# Set inputs.
NG = 0
MAP_SERIAL_NUM = 1
MASK_TAG = "random0_BOSS_DR12v5_CMASS_North"
SELECTION_TAG = "[100.0,500.0]"

PIVOT = 'natural'

# Load components.
couplings = Couplings.load(
    data_dir/"processed"/"survey_products"/
    f"couplings-(kmax=0.04,mask={MASK_TAG},selection={SELECTION_TAG}).npz"
)

spherical_data = SphericalArray.load(
    data_dir/"raw"/"catalpgue_maps"/"catalogue-map-({}).npz".format(
        ",".join([
            f"source=halo-(NG={NG}.,z=1.)-{MAP_SERIAL_NUM}",
            "map=spherical", "scale=[None,0.04]", "orders=None", "rsd=False",
            f"mask={MASK_TAG}", f"selection={SELECTION_TAG}"
        ])
    )
)

spherical_model = SphericalCorrelator(
    spherical_data.disc, redshift=1., growth_rate=0., couplings=couplings,
    cosmo=BaseModel(data_dir/"external"/"cosmology"/"simulation.txt")
)

# Set up likelihood from components.
likelihood_func = LogLikelihood(
    spherical_data=spherical_data, base_spherical_model=spherical_model,
    spherical_pivot=PIVOT, nbar=2.5e-4, contrast=10.
)

radialise = (MASK_TAG is None or MASK_TAG.startswith('1.')) \
    and (SELECTION_TAG is None or SELECTION_TAG.upper() == 'NONE')

if radialise:
    comp_mat = None
else:
    comp_mat = generate_compression_matrix(
        {
            'pivot': PIVOT, 'spherical_model': spherical_model,
            'b_1': 2.5, 'f_nl': NG
        },
        {
            'pivot': PIVOT, 'spherical_model': spherical_model,
            'b_1': 1., 'f_nl': NG - 100
        }
    )

# Evaluate likelihood on a grid.
bias = [2., 2.5, 3.]
png = [-100., 0., 100.]

likelihood = [
    likelihood_func.spherical_map_likelihood(
        b_1=b_1, f_nl=f_nl, radialise=radialise, compression_matrix=comp_mat
    ) for b_1 in bias for f_nl in png
]

sns.heatmap(likelihood, xticklabels=png, yticklabels=bias)
