import numpy as np
import pytest

from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.algorithms.arrays import CartesianArray, SphericalArray

TEST_PARAMS = dict(
    radius=100.,
    condition='Dirichlet',
    cutoff=0.1,
    cuton=0.,
    maxdeg=None,
    mindeg=0,
)

TEST_DATA_ARRAY = [
    np.array([[3.14, 6.28, 9.42]] * 1),
    np.array([[4.49, 7.73]] * 3),
    np.array([[5.76, 9.10]] * 5),
    np.array([[6.99]] * 7),
    np.array([[8.18]] * 9),
    np.array([[9.36]] * 11),
]

INVALID_TEST_DATA_ARRAY = [
    np.array([[3.14, 6.28, 9.42]] * 1),
    np.array([[4.49, 7.73, 10.00]] * 3),
    np.array([[5.76, 9.10]] * 4),
    np.array([[6.99]] * 7),
    np.array([[8.18]] * 9),
    np.array([[9.36]] * 11),
]


@pytest.fixture(scope='module')
def discrete_spectrum():
    return DiscreteSpectrum(**TEST_PARAMS)


@pytest.fixture(scope='module')
def spherical_array(discrete_spectrum):
    return SphericalArray(
        discrete_spectrum.degrees,
        discrete_spectrum.depths,
        filling=TEST_DATA_ARRAY
    )


def test_init(discrete_spectrum, spherical_array):

    with pytest.raises(ValueError):
        SphericalArray(
            discrete_spectrum.degrees,
            discrete_spectrum.depths,
            filling=INVALID_TEST_DATA_ARRAY
        )

    assert all(
        [
            np.all(block == test_block)
            for block, test_block in zip(
                spherical_array.data_array,
                TEST_DATA_ARRAY
            )
        ]
    )


def test_build(discrete_spectrum, spherical_array):

    alternative_spherical_array = SphericalArray.build(
        disc=discrete_spectrum,
        filling=TEST_DATA_ARRAY,
    )
    assert all(
        [
            np.all(block == alternative_block)
            for block, alternative_block in zip(
                spherical_array.data_array,
                alternative_spherical_array.data_array,
            )
        ]
    )
    assert all(
        [
            np.all(block == alternative_block)
            for block, alternative_block in zip(
                spherical_array.index_array,
                alternative_spherical_array.index_array,
            )
        ]
    )


MAX_ENTRY = 9


@pytest.mark.parametrize(
    "axis_order,flat_array,flat_indices",
    [
         (
             'natural',
             [3.14, 6.28, 9.42, 4.49, 7.73, 4.49, 7.73, 4.49, 7.73],
             [
                 (0, 0, 1), (0, 0, 2), (0, 0, 3), (1, -1, 1), (1, -1, 2),
                 (1, 0, 1), (1, 0, 2), (1, 1, 1), (1, 1, 2),
             ],
         ),
         (
             'transposed',
             [3.14, 6.28, 9.42, 4.49, 4.49, 4.49, 7.73, 7.73, 7.73],
             [
                 (0, 0, 1), (0, 0, 2), (0, 0, 3), (1, -1, 1), (1, 0, 1),
                 (1, 1, 1), (1, -1, 2), (1, 0, 2), (1, 1, 2),
             ],
         ),
         (
             'root',
             [3.14, 6.28, 9.42, 4.49, 7.73, 5.76, 9.1, 6.99, 8.18],
             [
                 (0, None, 1), (0, None, 2), (0, None, 3), (1, None, 1),
                 (1, None, 2), (2, None, 1), (2, None, 2), (3, None, 1),
                 (4, None, 1),
             ],
         ),
         (
             'spectral',
             [3.14, 4.49, 4.49, 4.49, 5.76, 5.76, 5.76, 5.76, 5.76],
             [
                 (0, 0, 1), (1, -1, 1), (1, 0, 1), (1, 1, 1), (2, -2, 1),
                 (2, -1, 1), (2, 0, 1), (2, 2, 1), (2, 1, 1),
             ],
         ),
         (
             'scale',
             [3.14, 4.49, 5.76, 6.28, 6.99, 7.73, 8.18, 9.1, 9.36],
             [
                 (0, None, 1), (1, None, 1), (2, None, 1), (0, None, 2),
                 (3, None, 1), (1, None, 2), (4, None, 1), (2, None, 2),
                 (5, None, 1),
             ],
         ),
    ]
)
def test_unfold(spherical_array, axis_order, flat_array, flat_indices):

    unfolded_array, unfolded_indices = \
        spherical_array.unfold(axis_order=axis_order)
    assert unfolded_array[:MAX_ENTRY] == pytest.approx(flat_array)
    assert unfolded_indices[:MAX_ENTRY] == flat_indices


@pytest.mark.parametrize(
    "in_structure",
    ['natural', 'transposed', 'root', 'spectral', 'scale'],
)
def test_fold(spherical_array, in_structure):
    assert all(
        [
            np.allclose(init_subarr, folded_subarr)
             for init_subarr, folded_subarr in zip(
                 spherical_array.data_array,
                 spherical_array.fold(
                     spherical_array.unfold(axis_order=in_structure)[0],
                     in_structure,
                     'data',
                 )
             )
        ]
    )
    assert all(
        [
            init_subarr == folded_subarr
            for init_subarr, folded_subarr in zip(
                 spherical_array.index_array,
                 spherical_array.fold(
                     spherical_array.unfold(axis_order=in_structure)[1],
                     in_structure,
                     'index',
                 )
             )
         ]
    )


@pytest.mark.parametrize("in_structure,out_structure", [('lnm', 'u')])
def test_morph(spherical_array, in_structure, out_structure):

    in_array, in_indices = spherical_array.unfold(axis_order=in_structure)
    out_array, out_indices = spherical_array.unfold(axis_order=out_structure)

    assert all(
        [
            out_subarr == morphed_subarr
            for out_subarr, morphed_subarr in zip(
                out_array,
                spherical_array.morph(
                    in_array, in_structure, out_structure, 'data'
                )
            )
        ]
    )
    assert all(
        [
            out_subarr == morphed_subarr
            for out_subarr, morphed_subarr in zip(
                out_indices,
                spherical_array.morph(
                    in_indices, in_structure, out_structure, 'index'
                )
            )
        ]
    )


TEST_DATA_DICT = {
    'x': [0.0, 0.5, 1.0],
    'y_0': [-1.5, -1., -0.5],
    'y_1': [-0.25, 0., 0.25],
    'y_2': [0.5, 1., 1.5],
}

@pytest.mark.parametrize(
    'pivot,results',
    [
        (
            'coord',
            (
                [-1.5, -0.25, 0.5, -1., 0., 1., -0.5, 0.25, 1.5],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
            )
        ),
        (
            'variable',
            (
                [-1.5, -1., -0.5, -0.25, 0., 0.25, 0.5, 1., 1.5],
                [0.0, 0.5, 1.0] * 3
            )
        ),
    ]
)
def test_cartesianarray_unfold(pivot, results):
    cartesian_array = CartesianArray(TEST_DATA_DICT, 'x', 'y_')
    assert np.array_equal(cartesian_array.unfold(pivot)[-1], results[-1])
    assert np.array_equal(cartesian_array.unfold(pivot)[0], results[0])
