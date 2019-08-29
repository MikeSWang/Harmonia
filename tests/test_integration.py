import numpy as np

from testrc import wolfram_alpha_query, NamedFunction
from harmonia.algorithms.integration import (
    radial_spherical_int, angular_spherical_int,
    radial_sphint_besselj, angular_sphint_harmonic,
    )

TEST_PARAMS = dict(
    rmax=100.,
    ell=5,
    m=2,
    k=0.05,
    )

radial_func = NamedFunction(
    f"Sin[r]/{TEST_PARAMS['rmax']}",
    lambda r: np.sin(r) / TEST_PARAMS['rmax']
    )

angular_func = NamedFunction(
    f"Sin[theta] Sin[theta + phi]",
    lambda theta, phi: np.sin(theta + phi)
    )


def test_radial_spherical_int():
    wolfram_alpha_query(
        f"Integrate[r^2 {repr(radial_func)}, {{r, 0, {TEST_PARAMS['rmax']}}}]"
        )
    assert np.isclose(
        radial_spherical_int(radial_func, TEST_PARAMS['rmax']), -87.2473721335
        )


def test_angular_spherical_int():
    wolfram_alpha_query(
        f"Integrate[{repr(angular_func)}, {{theta, 0, Pi}}, {{phi, 0, 2 Pi}}]"
        )
    assert np.isclose(angular_spherical_int(angular_func), 0)


def test_radial_sphint_besselj():
    wolfram_alpha_query(
        f"Integrate[r^2 "
        f"SphericalBesselJ[{TEST_PARAMS['ell']}, {TEST_PARAMS['k']}*r] "
        f"{repr(radial_func)}, {{r, 0, {TEST_PARAMS['rmax']}}}]"
        )
    assert np.isclose(radial_sphint_besselj(
        radial_func, TEST_PARAMS['ell'], TEST_PARAMS['k'], TEST_PARAMS['rmax']
        ), -9.45626)


def test_angular_sphint_harmonic():
    wolfram_alpha_query(
        f"Integrate[Sin(theta) "
        f"SphericalHarmonicY[{TEST_PARAMS['ell']}, {TEST_PARAMS['m']}, "
        f"theta, phi] {repr(radial_func)}, "
        f"{{theta, 0, Pi}}, {{phi, 0, 2Pi}}]"
        )
    assert np.isclose(angular_sphint_harmonic(
        angular_func, TEST_PARAMS['ell'], TEST_PARAMS['m']
        ), 0)
