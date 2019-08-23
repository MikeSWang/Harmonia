import sys
sys.path.insert(0, "../")

import numpy as np

from harmonia.algorithms.integration import (
    radial_spherical_int, angular_spherical_int,
    radial_sphint_besselj, angular_sphint_harmonic,
    )

TEST_PARAMS = dict(
    rmax = 100.,
    ell = 5,
    m = 2,
    k = 0.05,
)

TEST_FUNCS = dict(
    radial_func = lambda r: np.sin(r) / TEST_PARAMS['rmax'],
    angular_func = lambda theta, phi: np.sin(theta + phi)
)


def test_radial_spherical_int():
    # WolframAlpha: integrate r^2 * sin(r) / 100 from r = 0 to 100
    assert np.isclose(
        radial_spherical_int(TEST_FUNCS['radial_func'], TEST_PARAMS['rmax']),
        -87.2473721335422
        )


def test_angular_spherical_int():
    # WolframAlpha: integrate sin(theta) * sin(theta + phi)
    # from theta = 0 to pi and phi = 0 to 2*pi
    assert np.isclose(
        angular_spherical_int(TEST_FUNCS['angular_func']),
        0
        )


def test_radial_sphint_besselj():
    # WolframAlpha: integrate r^2 * SphericalBesselJ[5, 0.05*r] * sin(r)/100
    # from r = 0 to 100
    assert np.isclose(
        radial_sphint_besselj(
            TEST_FUNCS['radial_func'],
            TEST_PARAMS['ell'], TEST_PARAMS['k'], TEST_PARAMS['rmax']
            ),
        -9.45626
        )


def test_angular_sphint_harmonic():
    # WolframAlpha: integrate
    # sin(theta) * SphericalHarmonicY[5, 2, theta, phi] * sin(theta + phi)
    # from theta = 0 to pi and phi = 0 to 2*pi
    assert np.isclose(
        angular_sphint_harmonic(
            TEST_FUNCS['angular_func'], TEST_PARAMS['ell'], TEST_PARAMS['m']
            ),
        0
        )
