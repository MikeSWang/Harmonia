import numpy as np
import pytest

from harmonia.algorithms.bases import (
    spherical_besselj,
    spherical_besselj_root,
    spherical_harmonic
)

from . import display_mathematica_query as show_query

TEST_PARAMS = dict(
    ell=5,
    x=10.,
    nmax=3,
    m=2,
    theta=np.pi/3,
    phi=np.pi/5,
)


@pytest.mark.parametrize(
    "ell,x,deriv,value",
    [
        (15, 10., False, 0.00106354),
        (30, 20., True, 0.0000240929),
    ]
)
def test_spherical_besselj(ell, x, deriv, value):

    if deriv:
        show_query(
            f"ReplaceAll[D[SphericalBesselJ[{ell}, z], z], {{z -> {x}}}]"
        )
    else:
        show_query(f"SphericalBesselJ[{ell}, {x}]")

    assert np.isclose(spherical_besselj(ell, x, derivative=deriv), value), \
        "Incorrect evaluaion of spherical Bessel functions."


@pytest.mark.slow
@pytest.mark.parametrize(
    "ell,nmax,deriv,values",
    [
        (5, 4, False, [9.35581, 12.9665, 16.3547, 19.6532]),
        (15, 2, True, [17.4080, 22.6625]),
    ]
)
def test_spherical_besselj_root(ell, nmax, deriv, values):

    if deriv:
        show_query(f"D[SphericalBesselJ[{ell}, x], x] == 0")
    else:
        show_query(f"SphericalBesselJ[{ell}, x] == 0")

    assert np.allclose(
        spherical_besselj_root(ell, nmax, derivative=deriv, only=False),
        values
    ), "Root finding of spherical Bessel functions failed."


@pytest.mark.parametrize(
    "ell,m,theta,phi,value",
    [
        (5, 4, np.pi/3, np.pi/3, -0.206397 - 0.357491*1j),
        (15, -2, np.pi/5, np.pi/6, 0.200298 - 0.346926*1j),
    ]
)
def test_spherical_harmonic(ell, m, theta, phi, value):

    show_query(f"SphericalHarmonicY[{ell}, {m}, {theta}, {phi}]")
    assert np.isclose(spherical_harmonic(ell, m, theta, phi), value), \
        "Incorrect evaluaion of spherical harmonic functions."
