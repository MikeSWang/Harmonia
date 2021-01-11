import pytest

from harmonia.mapper.catalogue_maker import spherical_indicator


@pytest.mark.parametrize(
    "vec,r,value",
    [
        ([1, 0, -1], 1, False),
    ]
)
def test_spherical_indicator(vec, r, value):
    assert (spherical_indicator(vec, r) == value), \
        "Incorrect indication for spherical domain."
