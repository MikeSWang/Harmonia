import os

import numpy as np
import pytest

from harmonia.collections import utils


def test_confirm_directory_path():
    assert os.path.abspath("./")


def test_get_filename():
    assert utils.get_filename(os.path.abspath(__file__)) == 'test_utils'


TMP_DATA = [
    [[1, 2], [2, 4], [3, 6]],
    [[1, -2], [2, -4], [3, -6]],
]
TMP_COL = {
    'x': [[1, 2, 3], [1, 2, 3]],
    'y': [[2, 4, 6], [-2, -4, -6]],
}


def test_collate_data_files(tmpdir):

    tmp_dir = tmpdir.strpath
    for fidx, tmp_dat in enumerate(TMP_DATA):
        np.savetxt("{}/tmp_dat_{}.txt".format(tmp_dir, fidx), tmp_dat)

    collated_data, collation_count, last_collated_file = \
        utils.collate_data_files(
            "{}/tmp_dat_*.txt".format(tmp_dir),
            'txt',
            headings=['x', 'y'],
            columns=[0, 1]
        )
    assert np.allclose(collated_data['x'], TMP_COL['x'])
    assert np.allclose(collated_data['y'], TMP_COL['y'])
    assert collation_count == 2
    assert last_collated_file.startswith('tmp_dat_')


@pytest.mark.parametrize("ntasks,nproc", [(10, 3), (100, 12), (1000, 72)])
def test_allocate_tasks(ntasks, nproc):

    tasks = utils.allocate_tasks(ntasks, nproc)

    assert np.sum(tasks) == ntasks
    assert np.max(np.abs(np.diff(tasks))) <= 1


TEST_CASE_INTERVAL = 20
TEST_CASE_PROCS = 5


@pytest.mark.parametrize(
    "tasks,ntask,nproc",
    [
        (
            [TEST_CASE_INTERVAL] * (TEST_CASE_PROCS - 1) \
                + [TEST_CASE_INTERVAL + 1],
            TEST_CASE_INTERVAL * TEST_CASE_PROCS + 1,
            TEST_CASE_PROCS,
        ),
    ]
)
def test_allocate_segments(tasks, ntask, nproc):

    assert utils.allocate_segments(tasks=tasks) == \
        [
            slice(TEST_CASE_INTERVAL * n, TEST_CASE_INTERVAL * (n + 1))
            for n in range(nproc-1)
        ] + \
        [
            slice(
                TEST_CASE_INTERVAL * (nproc - 1),
                TEST_CASE_INTERVAL * nproc + 1,
            )
        ]

    assert utils.allocate_segments(total_task=ntask, total_proc=nproc) == \
        [
            slice(TEST_CASE_INTERVAL*n, TEST_CASE_INTERVAL*(n+1))
            for n in range(nproc-1)
        ] \
        + [slice(TEST_CASE_INTERVAL*(nproc - 1), TEST_CASE_INTERVAL*nproc + 1)]

    with pytest.raises(TypeError):
        utils.allocate_segments(total_task=ntask)


@pytest.mark.parametrize(
    "x,case,float_str",
    [
        (0.000001, 'latex', r'1.0 \times 10^{-6}'),
        (0.000001, 'sci', '1.e-6'),
        (1, 'intdot', '1.'),
        (1.02, 'decdot', '1.'),
    ]
)
def test_format_float(x, case, float_str):

    assert utils.format_float(x, case) == float_str

    with pytest.raises(ValueError):
        utils.format_float(x, 'invalid_case')


@pytest.mark.parametrize(
    "keys,arrays,sorted_arrays",
    [
        ([1, 3, 2], [[1], [2, 3], [4, 5, 6]], [[1], [4, 5, 6], [2, 3]]),
    ]
)
def test_sort_dict_to_list(keys, arrays, sorted_arrays):

    dict_data = dict(zip(keys, arrays))

    assert utils.sort_dict_to_list(dict_data) == sorted_arrays


@pytest.mark.parametrize(
    "keys,arrays,sorted_keys,sorted_arrays",
    [
        (
            [1, 3, 2],
            [[1], [2, 3], [4, 5, 6]],
            [1, 2, 3],
            [[1], [4, 5, 6], [2, 3]],
        ),
    ]
)
def test_sort_list_to_dict(keys, arrays, sorted_keys, sorted_arrays):
    assert utils.sort_list_to_dict(sorted_arrays, sorted_keys) \
        == dict(zip(keys, arrays))


def test_zero_const():
    assert utils.zero_const() == pytest.approx(0.)


def test_unit_const():
    assert utils.unit_const() == pytest.approx(1.)


@pytest.mark.parametrize("const,x", [(1j, 5), (100, 1j)])
def test_const_function(const, x):
    assert utils.const_function(const)(x) == pytest.approx(const)


@pytest.mark.parametrize(
    "matrix,logdet",
    [(10*np.eye(499), 1148.9899614040)]
)
def test_mat_logdet(matrix, logdet):
    assert utils.mat_logdet(matrix) == pytest.approx(logdet)


@pytest.mark.parametrize("ndim,nsize", [(3, 10), (5, 8)])
def test_covar_to_corr(ndim, nsize):

    randvec = np.random.random(size=(ndim, nsize))
    covar = np.cov(randvec)
    corr = np.corrcoef(randvec)

    assert np.allclose(utils.covar_to_corr(covar), corr)


@pytest.mark.parametrize(
    "func,a,b,maxnum,roots",
    [
        (np.sin, -0.01, 3.15, 1, np.array([0])),
        (np.cos, -0.01, 3.15, 10, np.array([np.pi/2])),
    ]
)
def test_binary_search(func, a, b, maxnum, roots):
    assert np.allclose(utils.binary_search(func, a, b, maxnum=maxnum), roots)


@pytest.mark.parametrize("vec", [[[1, 3, -5], [0.2, -0.88, -10]]])
def test_normalise_vector(vec):
    assert np.allclose(
        np.array(vec) / np.linalg.norm(vec, axis=-1, keepdims=True),
        utils.normalise_vector(vec)
    )


@pytest.mark.parametrize("vec,r", [([1, 0, -1], 1)])
def test_spherical_indicator(vec, r):
    assert not utils.spherical_indicator(vec, r)


@pytest.mark.parametrize("vec", [[[1, 3, -5], [0.2, -0.88, -10]]])
def test_cartesian_to_spherical(vec):
    assert np.allclose(
        utils.cartesian_to_spherical(vec),
        [
            [5.916079783, 2.577650012, 1.249045772],
            [10.04063743, 3.051592333, 4.935866174],
        ]
    )


@pytest.mark.parametrize(
    "vec",
    [
        [
            [5.916079783, 2.577650012, 1.249045772],
            [10.04063743, 3.051592333, 4.935866174],
        ],
    ]
)
def test_spherical_to_cartesian(vec):
    assert np.allclose(
        utils.spherical_to_cartesian(vec),
        [[1, 3, -5], [0.2, -0.88, -10]]
    )


def test_bin_edges_from_centres():

    assert utils.bin_edges_from_centres([1, 4.5, 8.5], [0, 11], align='low') \
        == pytest.approx([0, 2, 7, 11])
    assert utils.bin_edges_from_centres([1, 4.5, 8.5], [0, 11], align='high') \
        == pytest.approx([0, 3, 6, 11])


TMP_COARSE_DATA = {
    'x': np.array([0.4, 1.4, 2.6, 3.6, 4.3, 5.3, 6.7, 7.7, 8.5, 9.5]),
    'y': np.array([0.8, 1.2, 1.7, 2.3, 2.95, 3.05, 3.98, 4.02, 4.99, 5.01]),
    'dx': np.array([0.05] * 10),
    'dy': np.array([0.1] * 10),
}
TMP_SMOOTH_DATA = {
    'x': np.array([0.9, 3.1, 4.8, 7.2, 9.]),
    'y': np.array([1., 2., 3., 4., 5.]),
    'dx': np.array([0.0707106781]*5),
    'dy': np.array([2*0.0707106781]*5),
}

TEST_BIN_EDEGS = [0, 2, 4, 6, 8, 10]
TEST_BIN_COUNTS = [2, 2, 2, 2, 2]


def test_smooth_by_bin_average():

    with pytest.raises(NotImplementedError):
        utils.smooth_by_bin_average(
            [1, 2, 3], [0, 0.005, 0.05, 0.1], 'kln', 'Pln'
        )

    smoothed_data, count_in_bins = utils.smooth_by_bin_average(
        TMP_COARSE_DATA, TEST_BIN_EDEGS, 'x', 'y',
        dx_coarse_key='dx',
        dy_coarse_key='dy'
    )
    for key in smoothed_data:
        assert smoothed_data[key] == pytest.approx(TMP_SMOOTH_DATA[key])
    assert count_in_bins == pytest.approx(TEST_BIN_COUNTS)
