import sys
sys.path.insert(0, "../")

import numpy as np

from harmonia.algorithms.bases import (
    sph_besselj, sph_besselj_root, sph_harmonic
    )

TEST_PARAMS = dict(
    ell = 5,
    x = 10.,
    nmax = 3,
    m = 2,
    theta = np.pi/3,
    phi = np.pi/5,
)


def test_sph_besselj():
    # WolframAlpha: SphericalBesselJ[5, 10.]
    assert np.isclose(
        sph_besselj(TEST_PARAMS['ell'], TEST_PARAMS['x'], deriv=False),
        -0.05553451162145218090882828945258120943941079880731541183
        )
    # WolframAlpha: ReplaceAll[D[SphericalBesselJ[5, z], z], {z -> 10.}]
    assert np.isclose(
        sph_besselj(TEST_PARAMS['ell'], TEST_PARAMS['x'], deriv=True),
        -0.07226857814482036397320745610547477366425546136898368696
        )


def test_sph_besselj_root():
    # WolframAlpha: SphericalBesselJ[5, x] == 0
    assert np.isclose(
        sph_besselj_root(
            TEST_PARAMS['ell'], TEST_PARAMS['nmax'], only=True, deriv=False
            ),
        16.3547096393505
        )
    assert np.allclose(
        sph_besselj_root(
            TEST_PARAMS['ell'], TEST_PARAMS['nmax'], only=False, deriv=False
            ),
        [9.35581211104275, 12.9665301727743, 16.3547096393505,]
        )
    # WolframAlpha: D[SphericalBesselJ[5, x], x] == 0
    assert np.isclose(
        sph_besselj_root(
            TEST_PARAMS['ell'], TEST_PARAMS['nmax'], only=True, deriv=True
            ),
        14.5905521631334
        )
    assert np.allclose(
        sph_besselj_root(
            TEST_PARAMS['ell'], TEST_PARAMS['nmax'], only=False, deriv=True
            ),
        [6.75645633020413, 11.0702068737291, 14.5905521631334,]
        )


def test_sph_harmonic():
    # WolframAlpha: SphericalHarmonicY[5, 2, Pi/3, Pi/5]
    assert np.isclose(
        sph_harmonic(
            TEST_PARAMS['ell'], TEST_PARAMS['m'],
            TEST_PARAMS['theta'], TEST_PARAMS['phi']
            ),
        -0.0490981028629271532708573695235382621422006113471920923
        -0.151108422887768076838467536710630602471023507176469444*1j
        )
