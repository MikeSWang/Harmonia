import os
import math

import numpy as np
import pytest

import testrc
import harmonia.collections.utils as utils


def test_filename():
    assert utils.filename(os.path.abspath(__file__)) == 'test_utils'


# NOTE: Tests with files not performed.
def test_collate():
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
        utils.allocate_segments(tasks=tasks) ==
        [slice(TEST_CASE_INTERVAL*n, TEST_CASE_INTERVAL*(n+1))
         for n in range(nproc-1)]
        + [slice(TEST_CASE_INTERVAL*(nproc-1), TEST_CASE_INTERVAL*nproc+1)]
        )
    assert (
        utils.allocate_segments(ntask=ntask, nproc=nproc) ==
        [slice(TEST_CASE_INTERVAL*n, TEST_CASE_INTERVAL*(n+1))
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
    assert utils.zeroconst() == 0


def test_unitconst():
    assert utils.unitconst() == 1


@pytest.mark.parametrize('ndim,nsize', [(3, 10), (5, 8)])
def test_covar_to_corr(ndim, nsize):
    randvec = np.random.random(size=(ndim, nsize))
    covar = np.cov(randvec)
    corr = np.corrcoef(randvec)
    assert np.allclose(
        utils.covar_to_corr(covar),
        corr
        )


@pytest.mark.parametrize(
    'func,a,b,maxnum,roots',
    [(lambda x : np.sin(x), -0.01, 3.15, 1, np.array([0])),
     (lambda x : np.cos(x), -0.01, 3.15, 10, np.array([np.pi/2]))]
    )
def test_bisect_roots(func, a, b, maxnum, roots):
    assert np.allclose(
        utils.bisect_roots(func, a, b, maxnum=maxnum),
        roots
        )


@pytest.mark.parametrize('vec', [([[1, 3, -5], [0.2, -0.88, -10]])])
def test_normalise_vector(vec):
    assert np.allclose(
        np.array(vec)/np.linalg.norm(vec, axis=-1, keepdims=True),
        utils.normalise_vector(vec)
        )


@pytest.mark.parametrize('vec', [([[1, 3, -5], [0.2, -0.88, -10]])])
def test_cartesian_to_spherical(vec):
    assert np.allclose(
        utils.cartesian_to_spherical(vec),
        [[5.91608, np.pi-math.radians(32.3115), math.radians(71.5651)],
         [10.0406, np.pi-math.radians(5.15664), math.radians(-77.1957)]]
        )


@pytest.mark.parametrize(
    'vec',
    [([[5.91607978, 2.57765001, 1.24904577],
       [10.04063743, 3.05159233, -1.34731973]])]
    )
def test_spherical_to_cartesian(vec):
    assert np.allclose(
        utils.spherical_to_cartesian(vec), [[1, 3, -5], [0.2, -0.88, -10]]
        )


def test_bin_edges_from_centres():
    assert np.allclose(
        utils.bin_edges_from_centres([1, 4.5, 8.5], [0, 11], align='low'),
        [0, 2, 7, 11]
        )
    assert np.allclose(
        utils.bin_edges_from_centres([1, 4.5, 8.5], [0, 11], align='high'),
        [0, 3, 6, 11]
        )


# NOTE: Tests with data not performed.
def test_smooth_by_bin_average():
    with pytest.raises(NotImplementedError):
        utils.smooth_by_bin_average(
            [1, 2, 3], [0, 0.005, 0.05, 0.1], 'kln', 'Pln'
            )
