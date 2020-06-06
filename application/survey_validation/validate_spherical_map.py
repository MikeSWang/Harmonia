"""Validate spherical map by comparison with the Cartesian power spectrum
in the case of radialisation and/or the spherical mode amplitudes.

"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    from application import data_dir
    from harmonia.algorithms import SphericalArray
    from harmonia.cosmology import BaseModel, modified_power_spectrum
    from harmonia.mapper import SphericalMap
    from harmonia.reader import Couplings, SphericalCorrelator
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import data_dir
    from harmonia.algorithms import SphericalArray
    from harmonia.cosmology import BaseModel, modified_power_spectrum
    from harmonia.mapper import SphericalMap
    from harmonia.reader import Couplings, SphericalCorrelator


def extract_map_power(map_serial_nums):
    """Extract mode power from spherical maps.

    Returns
    -------
    list, list, :class:`harmonia.algorithms.disscretisation.DiscreteSpectrum`
        Lists of mode power and spherical map coefficients from the serial
        maps and the discrete spectrum.

    """
    _mode_power, _mode_coeff = [], []
    for serial_num in map_serial_nums:
        spherical_data = SphericalArray.load(
            map_dir/map_file.format(TRACER, NG, SERIES, serial_num)
        )

        _disc = spherical_data.disc

        # pylint: disable=protected-access
        spherical_map = SphericalMap._from_state({
            'density_contrast': spherical_data.__getstate__(),
            'disc': _disc.__getstate__(),
        })

        _mode_power.append(spherical_map.mode_power['mode_powers'])
        _mode_coeff.append(spherical_data.array)

    return _mode_power, _mode_coeff, _disc


def predict_power_spectrum():
    """Predict the power spectrum from spherical modelling.

    """
    # pylint: disable=global-statement
    global spherical_model
    spherical_model = SphericalCorrelator(
        disc, redshift=1., growth_rate=0.,
        couplings=Couplings.load(product_dir/couplings_file),
        cosmo=cosmology
    )

    radialised_power = spherical_model.radialised_power(**model_params)

    return radialised_power['mode_powers']


def compare_power_spectrum():
    """Compare map mode power, spherical model mode power and the
    Cartesian power spectrum.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Comparison plot.

    """
    wavenumbers = np.sort(list(disc.wavenumbers.values()))

    cartesian_spectrum = modified_power_spectrum(
        cosmo=cosmology, redshift=1., **model_params
    )(wavenumbers)

    try:
        volume_factor = float(MASK_TAG)
    except (TypeError, ValueError):
        volume_factor = 1.
    finally:
        cartesian_model_str = \
            r'${:.2f}\times$ power spectrum'.format(volume_factor) \
            if volume_factor != 1. else 'power spectrum'

    fig = plt.figure()
    plt.loglog(wavenumbers, np.mean(map_power, axis=0), label='map power')
    plt.loglog(
        wavenumbers, spherical_spectrum, ls='--', label='spherical model'
    )
    plt.loglog(
        wavenumbers, volume_factor * cartesian_spectrum, ls=':',
        label=cartesian_model_str
    )
    plt.xlabel(r"$k\ \  [h/\mathrm{{Mpc}}]$")
    plt.ylabel(r"$P(k)\ \ [(\mathrm{{Mpc}}/h)^3]$")
    plt.legend()

    return fig


spherical_model = None

if __name__ == '__main__':

    TRACER = 'halos'
    SERIES = '-cut_2'
    NG = 0
    MAP_SERIALS = range(1, 1+24)
    BIAS = 3.55  # 3.50
    DENSITY = 4.75e-5  # 4.91e-5
    CONTRAST = 50.
    KCUT = 0.04

    MASK_TAG = "1.0"  # "random0_BOSS_DR12v5_CMASS_North"
    SELECTION_TAG = "None"  # "[100.0,500.0]"

    # Set I/O paths.
    cosmo_file = data_dir/"external"/"cosmology"/"simulation-GadgetAHF.txt"

    map_dir = data_dir/"raw"/"catalogue_maps"
    map_file = "catalogue-map-({}).npz".format(",".join([
        "source={}-(NG={}.,z=1.){}-{}", "map=spherical",
        "scale=[None,0.04]", "orders=None", "rsd=False",
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
    ]))

    product_dir = data_dir/"processed"/"survey_products"
    couplings_file = "couplings-({}).npz".format(",".join([
        "rmax=500.0", "kmax={}".format(KCUT),
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
    ]))

    output_dir = data_dir/"raw"/"survey_validation"
    output_filename = "spherical-map-validation-({})".format(",".join([
        "scale=[None,0.04]", "rsd=False",
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG)
    ]))

    # Validate maps.
    cosmology = BaseModel(cosmo_file)

    model_params = dict(b_1=BIAS, f_nl=NG, nbar=DENSITY, contrast=CONTRAST)

    map_power, _, disc = extract_map_power(MAP_SERIALS)
    spherical_spectrum = predict_power_spectrum()
    figure = compare_power_spectrum()
    # pylint: disable=using-constant-test
    if False:
        figure.save(output_dir/(output_filename + '.pdf'))
