import numpy as np

import pytest

from . import NamedFunction, mathematica_query as query
from harmonia.algorithms.integration import (
    angular_harmonic_integral,
    angular_spherical_integral,
    radial_besselj_integral,
    radial_spherical_integral,
)

TEST_PARAMS = dict(rmax=100., ell=5, m=2, k=0.05)

radial_func = NamedFunction(
    f"Sin[r]/{TEST_PARAMS['rmax']}",
    lambda r: np.sin(r) / TEST_PARAMS['rmax'],
)

angular_func = NamedFunction(
    f"Sin[theta] Sin[theta + phi]",
    lambda theta, phi: np.sin(theta + phi),
)


def test_radial_spherical_integral():
    query(
        f"Integrate[r^2 {repr(radial_func)}, {{r, 0, {TEST_PARAMS['rmax']}}}]"
    )
    assert radial_spherical_integral(radial_func, TEST_PARAMS['rmax']) \
        == pytest.approx(-87.2473721335)


def test_angular_spherical_integral():
    query(
        f"Integrate[{repr(angular_func)}, {{theta, 0, Pi}}, {{phi, 0, 2 Pi}}]"
    )
    assert angular_spherical_integral(angular_func) == pytest.approx(0.)


def test_radial_besselj_integral():
    query(
        f"Integrate[r^2 "
        f"SphericalBesselJ[{TEST_PARAMS['ell']}, {TEST_PARAMS['k']}*r] "
        f"{repr(radial_func)}, {{r, 0, {TEST_PARAMS['rmax']}}}]"
    )
    assert np.isclose(
        radial_besselj_integral(
            radial_func,
            TEST_PARAMS['ell'],
            TEST_PARAMS['k'],
            TEST_PARAMS['rmax'],
        ),
        -9.45626,
    )


def test_angular_harmonic_integral():
    query(
        f"Integrate[Sin(theta) "
        f"SphericalHarmonicY[{TEST_PARAMS['ell']}, {TEST_PARAMS['m']}, "
        f"theta, phi] {repr(radial_func)}, "
        f"{{theta, 0, Pi}}, {{phi, 0, 2Pi}}]"
    )
    assert np.isclose(
        angular_harmonic_integral(
            angular_func,
            TEST_PARAMS['ell'],
            TEST_PARAMS['m'],
        ),
        0.,
    )
