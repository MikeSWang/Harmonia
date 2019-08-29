import numpy as np

from testrc import wolfram_alpha_query
from harmonia.algorithms.bases import (
    sph_besselj, sph_besselj_root, sph_harmonic
    )

TEST_PARAMS = dict(
    ell=5,
    x=10.,
    nmax=3,
    m=2,
    theta=np.pi/3,
    phi=np.pi/5,
)


def test_sph_besselj():
    wolfram_alpha_query(
        f"SphericalBesselJ[{TEST_PARAMS['ell']}, {TEST_PARAMS['x']}]"
        )
    assert np.isclose(
        sph_besselj(TEST_PARAMS['ell'], TEST_PARAMS['x'], deriv=False),
        -0.0555345116
        )
    wolfram_alpha_query(
        f"ReplaceAll[D[SphericalBesselJ[{TEST_PARAMS['ell']}, z], z], "
        f"{{z -> {TEST_PARAMS['x']}}}]"
        )
    assert np.isclose(
        sph_besselj(TEST_PARAMS['ell'], TEST_PARAMS['x'], deriv=True),
        -0.0722685781
        )


def test_sph_besselj_root():
    wolfram_alpha_query(
        f"SphericalBesselJ[{TEST_PARAMS['ell']}, x] == 0"
        )
    assert np.isclose(
        sph_besselj_root(
            TEST_PARAMS['ell'], TEST_PARAMS['nmax'], only=True, deriv=False
            ),
        16.3547096394
        )
    assert np.allclose(
        sph_besselj_root(
            TEST_PARAMS['ell'], TEST_PARAMS['nmax'], only=False, deriv=False
            ),
        [9.3558121110, 12.9665301728, 16.3547096394,]
        )
    wolfram_alpha_query(
        f"D[SphericalBesselJ[{TEST_PARAMS['ell']}, x], x] == 0"
        )
    assert np.isclose(
        sph_besselj_root(
            TEST_PARAMS['ell'], TEST_PARAMS['nmax'], only=True, deriv=True
            ),
        14.5905521631
        )
    assert np.allclose(
        sph_besselj_root(
            TEST_PARAMS['ell'], TEST_PARAMS['nmax'], only=False, deriv=True
            ),
        [6.7564563302, 11.0702068737, 14.5905521631,]
        )


def test_sph_harmonic():
    wolfram_alpha_query(
        f"SphericalHarmonicY[{TEST_PARAMS['ell']}, {TEST_PARAMS['m']}, "
        f"{TEST_PARAMS['theta']}, {TEST_PARAMS['phi']}]"
        )
    assert np.isclose(
        sph_harmonic(
            TEST_PARAMS['ell'], TEST_PARAMS['m'],
            TEST_PARAMS['theta'], TEST_PARAMS['phi']
            ),
        -0.0490981029 - 0.1511084229*1j
        )
