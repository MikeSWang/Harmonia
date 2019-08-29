import os
import math

import numpy as np
import pytest

import testrc
import harmonia.collections.utils as utils


def test_filename():
    assert utils.filename(os.path.abspath(__file__)) == 'test_utils'


TMP_DATA = [
    [[1, 2], [2, 4], [3, 6]],
    [[1, -2], [2, -4], [3, -6]],
    ]
TMP_COL = {
    'x': [[1, 2, 3], [1, 2, 3]],
    'y': [[2, 4, 6], [-2, -4, -6]],
    }


def test_collate(tmpdir):
    with pytest.raises(NotImplementedError):
        utils.collate('some_pattern', 'invalid_ext')
    with pytest.raises(ValueError):
        utils.collate('some_pattern', 'txt', headings=['a', 'b', 'c'])
    with pytest.raises(ValueError):
        utils.collate('some_pattern', 'dat', columns=[1, 5, 4])
    with pytest.raises(ValueError):
        utils.collate(
            'pattern', 'txt', headings=['a', 'b', 'c'], columns=[1, 2]
            )

    tmp_dir = tmpdir.strpath
    for fidx, tmp_dat in enumerate(TMP_DATA):
        np.savetxt("{}/tmp_dat_{}.txt".format(tmp_dir, fidx), tmp_dat)

    collated_data, count, last_file = utils.collate(
        "{}/tmp_dat_*.txt".format(tmp_dir), 'txt',
        headings=['x', 'y'], columns=[0, 1]
        )
    assert np.allclose(collated_data['x'], TMP_COL['x'])
    assert np.allclose(collated_data['y'], TMP_COL['y'])
    assert count == 2
    assert last_file.startswith('tmp_dat_')


@pytest.mark.parametrize('ntasks,nproc', [(10, 3), (100, 12), (1000, 72)])
def test_allocate_tasks(ntasks, nproc):
    tasks = utils.allocate_tasks(ntasks, nproc)

    assert np.sum(tasks) == ntasks
    assert np.max(np.abs(np.diff(tasks))) <= 1


TEST_CASE_INTERVAL = 20
TEST_CASE_PROCS = 5


@pytest.mark.parametrize(
    'tasks,ntask,nproc',
    [([TEST_CASE_INTERVAL]*(TEST_CASE_PROCS-1) + [TEST_CASE_INTERVAL+1],
      TEST_CASE_INTERVAL*TEST_CASE_PROCS+1, TEST_CASE_PROCS)]
    )
def test_allocate_segments(tasks, ntask, nproc):
    assert (
        utils.allocate_segments(tasks=tasks)
        == [slice(TEST_CASE_INTERVAL*n, TEST_CASE_INTERVAL*(n+1))
            for n in range(nproc-1)]
        + [slice(TEST_CASE_INTERVAL*(nproc-1), TEST_CASE_INTERVAL*nproc+1)]
        )
    assert (
        utils.allocate_segments(ntask=ntask, nproc=nproc)
        == [slice(TEST_CASE_INTERVAL*n, TEST_CASE_INTERVAL*(n+1))
            for n in range(nproc-1)]
        + [slice(TEST_CASE_INTERVAL*(nproc-1), TEST_CASE_INTERVAL*nproc+1)]
        )
    with pytest.raises(ValueError):
        utils.allocate_segments(ntask=ntask)


@pytest.mark.parametrize(
    'x,case,ff_str',
    [(0.000001, 'latex', r'1 \times 10^{-6}'),
     (0.000001, 'sci', '1e-6'),
     (1, 'intdot', '1.'),
     (1.02, 'decdot', '1.')]
    )
def test_format_float(x, case, ff_str):
    assert utils.format_float(x, case) == ff_str
    with pytest.raises(ValueError):
        utils.format_float(x, 'invalid_case')


def test_zeroconst():
    assert utils.zeroconst() == pytest.approx(0)


def test_unitconst():
    assert utils.unitconst() == pytest.approx(1)


@pytest.mark.parametrize('ndim,nsize', [(3, 10), (5, 8)])
def test_covar_to_corr(ndim, nsize):
    randvec = np.random.random(size=(ndim, nsize))
    covar = np.cov(randvec)
    corr = np.corrcoef(randvec)

    assert np.allclose(utils.covar_to_corr(covar), corr)


@pytest.mark.parametrize(
    'func,a,b,maxnum,roots',
    [(np.sin, -0.01, 3.15, 1, np.array([0])),
     (np.cos, -0.01, 3.15, 10, np.array([np.pi/2]))]
    )
def test_bisect_roots(func, a, b, maxnum, roots):
    assert np.allclose(utils.bisect_roots(func, a, b, maxnum=maxnum), roots)


@pytest.mark.parametrize('vec', [[[1, 3, -5], [0.2, -0.88, -10]]])
def test_normalise_vector(vec):
    assert np.allclose(
        np.array(vec)/np.linalg.norm(vec, axis=-1, keepdims=True),
        utils.normalise_vector(vec)
        )


@pytest.mark.parametrize('vec', [[[1, 3, -5], [0.2, -0.88, -10]]])
def test_cartesian_to_spherical(vec):
    assert np.allclose(
        utils.cartesian_to_spherical(vec),
        [[5.91608, np.pi-math.radians(32.3115), math.radians(71.5651)],
         [10.0406, np.pi-math.radians(5.15664), math.radians(-77.1957)]]
        )


@pytest.mark.parametrize(
    'vec',
    [[[5.91607978, 2.57765001, 1.24904577],
      [10.04063743, 3.05159233, -1.34731973]]]
    )
def test_spherical_to_cartesian(vec):
    assert np.allclose(
        utils.spherical_to_cartesian(vec), [[1, 3, -5], [0.2, -0.88, -10]]
        )


def test_bin_edges_from_centres():
    assert (
        utils.bin_edges_from_centres([1, 4.5, 8.5], [0, 11], align='low')
        == pytest.approx([0, 2, 7, 11])
        )
    assert (
        utils.bin_edges_from_centres([1, 4.5, 8.5], [0, 11], align='high')
        == pytest.approx([0, 3, 6, 11])
        )


TMP_COARSE_DATA = {
    'x': np.array([0.4, 1.4, 2.6, 3.6, 4.3, 5.3, 6.7, 7.7, 8.5, 9.5]),
    'y': np.array([0.8, 1.2, 1.7, 2.3, 2.95, 3.05, 3.98, 4.02, 4.99, 5.01]),
    'dx': np.array([0.05] * 10),
    'dy': np.array([0.1] * 10)
    }
TMP_SMOOTH_DATA = {
    'x': np.array([0.9, 3.1, 4.8, 7.2, 9.]),
    'y': np.array([1., 2., 3., 4., 5.]),
    'dx': np.array([0.07071067811865477]*5),
    'dy': np.array([2*0.07071067811865477]*5)
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
        dx_coarse='dx', dy_coarse='dy'
        )
    for key in smoothed_data:
        assert smoothed_data[key] == pytest.approx(TMP_SMOOTH_DATA[key])
    assert count_in_bins == pytest.approx(TEST_BIN_COUNTS)
