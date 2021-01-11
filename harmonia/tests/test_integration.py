import numpy as np
import pytest

from harmonia.algorithms.bases import spherical_besselj as sph_jn
from harmonia.algorithms.integration import (
    angular_integral,
    pixelated_angular_integral,
    radial_integral,
)

from . import NamedFunction, display_mathematica_query as show_query


@pytest.fixture(scope='module')
def radial_func():
    return NamedFunction(
        "SphericalBesselJ[5, r] * D[SphericalBesselJ[10, r], r]",
        lambda r: sph_jn(5, r) * sph_jn(10, r, derivative=True)
    )


@pytest.fixture(scope='module')
def angular_func():
    return NamedFunction(
        "Sin[theta + phi]",
        lambda theta, phi: np.sin(theta + phi)
    )


@pytest.mark.parametrize(
    "rmax,value",
    [
        (10, 0.211562),
    ]
)
def test_radial_integral(radial_func, rmax, value):

    show_query(f"Integrate[r^2 {str(radial_func)}, {{r, 0, {rmax}}}]")
    assert radial_integral(radial_func, rmax) == pytest.approx(value), \
        "Incorrect radial integration results for the test function."


def test_angular_integral(angular_func):

    show_query(
        f"Integrate[{str(angular_func)}, {{theta, 0, Pi}}, {{phi, 0, 2 Pi}}]"
    )
    assert angular_integral(angular_func) == pytest.approx(0.), \
        "Incorrect angular integration results for the test function."


@pytest.mark.parametrize(
    "nside",
    [64, 128]
)
def test_pixelated_angular_integral(angular_func, nside):

    show_query(
        f"Integrate[{str(angular_func)}, {{theta, 0, Pi}}, {{phi, 0, 2 Pi}}]"
    )
    assert pixelated_angular_integral(angular_func, nside) \
        == pytest.approx(0.), \
        "Incorrect angular integration results for the test function."
