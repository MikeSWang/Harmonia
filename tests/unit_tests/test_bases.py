import numpy as np

from unit_tests_rc import wolfram_alpha_query as query
from harmonia.algorithms.bases import (
    spherical_besselj,
    spherical_besselj_root,
    spherical_harmonic,
)

TEST_PARAMS = dict(
    ell=5,
    x=10.,
    nmax=3,
    m=2,
    theta=np.pi/3,
    phi=np.pi/5,
)


def test_spherical_besselj():

    query(f"SphericalBesselJ[{TEST_PARAMS['ell']}, {TEST_PARAMS['x']}]")
    assert np.isclose(
        spherical_besselj(
            TEST_PARAMS['ell'],
            TEST_PARAMS['x'],
            derivative=False,
        ),
        -0.0555345116,
    )

    query(
        f"ReplaceAll[D[SphericalBesselJ[{TEST_PARAMS['ell']}, z], z], "
        f"{{z -> {TEST_PARAMS['x']}}}]"
    )
    assert np.isclose(
        spherical_besselj(
            TEST_PARAMS['ell'],
            TEST_PARAMS['x'],
            derivative=True,
        ),
        -0.0722685781,
    )


def test_spherical_besselj_root():

    query(f"SphericalBesselJ[{TEST_PARAMS['ell']}, x] == 0")
    assert np.isclose(
        spherical_besselj_root(
            TEST_PARAMS['ell'],
            TEST_PARAMS['nmax'],
            only=True,
            derivative=False,
        ),
        16.3547096394,
    )
    assert np.allclose(
        spherical_besselj_root(
            TEST_PARAMS['ell'],
            TEST_PARAMS['nmax'],
            only=False,
            derivative=False,
        ),
        [
            9.3558121110,
            12.9665301728,
            16.3547096394,
        ],
    )

    query(f"D[SphericalBesselJ[{TEST_PARAMS['ell']}, x], x] == 0")
    assert np.isclose(
        spherical_besselj_root(
            TEST_PARAMS['ell'],
            TEST_PARAMS['nmax'],
            only=True,
            derivative=True,
        ),
        14.5905521631,
    )
    assert np.allclose(
        spherical_besselj_root(
            TEST_PARAMS['ell'],
            TEST_PARAMS['nmax'],
            only=False,
            derivative=True,
        ),
        [
            6.7564563302,
            11.0702068737,
            14.5905521631,
        ],
    )


def test_spherical_harmonic():
    query(
        f"SphericalHarmonicY[{TEST_PARAMS['ell']}, {TEST_PARAMS['m']}, "
        f"{TEST_PARAMS['theta']}, {TEST_PARAMS['phi']}]"
    )
    assert np.isclose(
        spherical_harmonic(
            TEST_PARAMS['ell'],
            TEST_PARAMS['m'],
            TEST_PARAMS['theta'],
            TEST_PARAMS['phi'],
        ),
        -0.0490981029 - 0.1511084229*1j,
    )
