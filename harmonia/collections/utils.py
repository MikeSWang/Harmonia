"""
Utilities (:mod:`~harmonia.collections.utils`)
===============================================================================

Common system and computational utilities.

System utilities
-------------------------------------------------------------------------------

**File handling**

.. autosummary::

    filename
    collate

**Multi-processing**

.. autosummary::

    allocate_tasks
    allocate_segments

**Formatting**

.. autosummary::

    format_float

Computational utilities
-------------------------------------------------------------------------------

**Algebraic algorithms**

.. autosummary::

    zeroconst
    unitconst
    covar_to_corr
    bisect_roots

**Geometrical algorithms**

.. autosummary::

    normalise_vector
    cartesian_to_spherical
    spherical_to_cartesian

**Statistical algorithms**

.. autosummary::

    bin_edges_from_centres
    smooth_by_bin_average

|

"""
import warnings
from collections import defaultdict
from glob import glob
from os.path import basename, splitext

import numpy as np
from numpy import arccos, arctan2
from numpy.linalg import norm


# SYSTEM UTILITIES
# -----------------------------------------------------------------------------

def filename(filepath):
    """Return file name without directory path or file extension.

    Parameters
    ----------
    filepath : str
        Full or partial file path.

    Returns
    -------
    str
        File name only.

    """
    return splitext(basename(filepath))[0]


def collate(filename_pattern, file_extension, headings=None, columns=None):
    """Collate data files.

    Parameters
    ----------
    filename_pattern : str
        Common root string of the file directory and name.
    file_extension : {'npy', 'txt', 'dat'}
        Data file extension.
    headings : list of str or None, optional
        Column headings to be used as dictionary keys (default is `None`).
    columns : list of int or None, optional
        Column indices (zero-indexed) to be used as dictionary keys
        corresponding to headings (default is `None`).

    Returns
    -------
    collated_data : dict
        Collated data.
    count : int
        Number of data files collated.
    last_file : str
        Last collated file name.

    Raises
    ------
    NotImplementedError
        If `file_extension` is not currently supported.
    ValueError
        If `file_extension` is ``'txt'`` or ``'dat'``, but either `headings` or
        `columns` is `None`.
    ValueError
        If `headings` and `columns` are not in correpondence.

    Notes
    -----
    For text files, the data is assumed to be stored as a column-major 2-d
    array with each column in correspondence with a key in the returned
    :obj:`dict`.

    """
    if file_extension.lower().endswith('npy', -3):
        # Append all data to a list.
        all_data = []
        for file in glob(filename_pattern):
            # Use `allow_pickle` for old numpy versions.
            all_data.append(np.load(file, allow_pickle=True).item())

        # Get aggregate data files information.
        count = len(all_data)
        last_file = filename(glob(filename_pattern).pop())

        # Initialise collated data using keys from the first data file.
        collated_data = dict.fromkeys(all_data[0].keys())
        for key in collated_data:
            collated_data[key] = np.concatenate(
                [np.atleast_1d(data[key]) for data in all_data], axis=0
                )

        return collated_data, count, last_file

    if (file_extension.lower().endswith('txt', -3)
            or file_extension.lower().endswith('dat', -3)):
        # Heading columns consistency check.
        if headings is None or columns is None:
            raise ValueError(
                "`headings` or `columns` cannot be None "
                "when reading from text files. "
                )
        if len(headings) != len(columns):
            raise ValueError(
                "Lengths of `headings` and `columns` must agree. "
                )

        # Aggregate data.
        collated_data = defaultdict(list)
        count = 0
        for file in glob(filename_pattern):
            data = np.loadtxt(file, usecols=columns)
            for keyidx, key in enumerate(headings):
                collated_data[key].append(np.atleast_2d(data[:, keyidx]))
            count += 1
        last_file = filename(glob(filename_pattern).pop())

        for key in headings:
            collated_data[key] = np.concatenate(collated_data[key], axis=0)

        return collated_data, count, last_file

    raise NotImplementedError("File extension currently unsupported. ")


def allocate_tasks(ntask, nproc):
    r"""Allocate tasks to processes for parallel computation.

    If `nproc` processes share `ntask` tasks, then :func:`allocate_task`
    decides the numbers of tasks, :const:`tasks`, different processes receive:
    the rank-:math:`i` process receives ``tasks[i]`` many tasks.

    Parameters
    ----------
    ntask : int
        Number of tasks.
    nproc : int
        Number of processes.

    Returns
    -------
    tasks : list of int
        Number of tasks for each process.

    """
    ntask_remain, nproc_remain, tasks = ntask, nproc, []

    while ntask_remain > 0:
        ntask_assign = ntask_remain // nproc_remain
        tasks.append(ntask_assign)
        ntask_remain -= ntask_assign
        nproc_remain -= 1

    return tasks


def allocate_segments(tasks=None, ntask=None, nproc=None):
    r"""Allocate segments of tasks to each process by the number of tasks it
    receives and its rank.

    For instance, if the rank-:math:`i` process receives ``tasks[i]`` tasks
    (e.g. assigned by :func:`allocate_task`), then this function assigns a
    slice of the indexed tasks it should receive, with the indices ordered in
    ascending correspondence with the ranks of the processes.

    Parameters
    ----------
    tasks : list of int or None, optional
        The number of tasks each process receives.  This cannot be `None` if
        `ntask` or `nproc` is `None`.  If this is not `None`, `ntask` and
        `nproc` values are both ignored.
    ntask : int or None
        Number of tasks.  This is ignored if `tasks` is not `None`, otherwise
        this cannot be `None`.
    nproc : int or None
        Number of processes.  This is ignored if `tasks` is not `None`,
        otherwise this cannot be `None`.

    Returns
    -------
    segments : list of slice
        The index slice of the segment of tasks that each process should
        receive.

    Raises
    ------
    ValueError
        If either `ntask` or `nproc` is `None` while `tasks` is also `None`.

    """
    if tasks is None:
        if ntask is None or nproc is None:
            raise ValueError(
                "`ntask` and `nproc` cannot be None "
                "while `tasks` is None. "
                )
        tasks = allocate_tasks(ntask, nproc)
    if nproc is None:
        nproc = len(tasks)

    breakpoints = np.insert(np.cumsum(tasks), 0, 0)
    segments = [
        slice(breakpoints[rank], breakpoints[rank+1]) for rank in range(nproc)
        ]

    return segments


def format_float(x, case):
    r"""Format float as a string.

    Parameters
    ----------
    x : float
        Number to be formatted.
    case : {'latex', 'sci', 'intdot', 'decdot'}
        Format case, one of LaTeX (``'latex'``), scientific (``'sci'``),
        rounded integer ending with a decimal dot (``'intdot'``), or
        a float whose first decimal place is 0 represented as a rounded integer
        ending with a decimal dot (``'decdot'``).

    Returns
    -------
    x_str : str
        Formatted string.

    Raises
    ------
    ValueError
        If `case` is not one of the supported formats.

    """
    if not isinstance(x, float):
        x = float(x)

    if case.lower() == 'latex':
        x_str = "{:g}".format(x)
        if "e" in x_str:
            base, exponent = x_str.split("e")
            x_str = r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    elif case.lower() == 'sci':
        x_str = "{:g}".format(x)
        if "e" in x_str:
            x_str = x_str.replace("e+0", "e+").replace("e-0", "e-")
    elif case.lower() == 'intdot':
        x_str = "{}".format(np.around(x)).strip("0")
    elif case.lower() == 'decdot':
        x_to1dp = "{:.1f}".format(x)
        if x_to1dp[-1] == '0':
            x_str = x_to1dp.strip("0")
        else:
            x_str = x_to1dp
    else:
        raise ValueError(
            "Supported formats are 'latex', 'sci', 'intdot', 'decdot' only. "
            )

    return x_str


# COMPUTATIONAL UTILITIES
# -----------------------------------------------------------------------------

def zeroconst(*args):
    """Return constant 0.

    Parameters
    ----------
    *args
        Arbitrary parameters.

    Returns
    -------
    0.

    """
    return 0.


def unitconst(*args):
    """Return constant 1.

    Parameters
    ----------
    *args
        Arbitrary parameters.

    Returns
    -------
    1.

    """
    return 1.


def covar_to_corr(cov):
    """Convert a real-valued covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : float, array_like
        Covariance matrix.

    Returns
    -------
    corr : float, array_like
        Correlation matrix.


    .. todo::

        Implement other types of conversion for more genertic covariance
        matrices.

    """
    inv_diag = np.diag(1 / np.sqrt(np.diag(cov)))
    corr = inv_diag @ np.array(cov) @ inv_diag

    return corr


def bisect_roots(func, a, b, maxnum=np.iinfo(np.int64).max, precision=1.e-5):
    """Binary seach for all roots of a function in an interval.

    Parameters
    ----------
    func : callable
        Function whose zeros are to be found.
    a, b : float
        Interval end points.
    maxnum : int, optional
        Maximum number of roots needed from below (default is
        :const:`numpy.iinfo(np.int64).max`).
    precision : float
        Precision required (default is 1.0e-5).

    Returns
    -------
    roots : float, array_like or None
        Possible roots.

    """

    def _scan_interval(func, a, b, dx):
        """Scan interval from lower end to detect sign change.

        Parameters
        ----------
        func : callable
            Function whose sign change interval is to be found.
        a, b: float
            Starting interval end points.
        dx : float
            Increment from lower end point.

        Returns
        -------
        x0, x1 : float or None
            End points for an interval with sign change (`None` if the result
            is null).

        """
        x0, x1 = a, a + dx
        f0, f1 = func(x0), func(x1)
        while f0 * f1 >= 0:
            if x0 >= b:  # terminate when interval exhausted
                return None, None
            x0, x1 = x1, x1 + dx
            f0, f1 = f1, func(x1)

        return x0, x1

    def _find_root(func, x0, x1, convergence=1.e-9):
        """Bisection method for root finding.

        Parameters
        ----------
        func : callable
            Function whose zero bracket is to be found.
        x0, x1: float
            Starting interval end points.
        convergence : float, optional
            Precision control for convergence through maximum iteration
            (default is 1.0e-9).

        Returns
        -------
        float or None
            A single possible root.

        """
        # Simple checks.
        f0, f1 = func(x0), func(x1)
        if f0 == 0:
            return x0
        if f1 == 0:
            return x1
        if f0 * f1 > 0:
            warnings.warn("Root is not bracketed. ", RuntimeWarning)
            return None

        # Determine maximum iteration given convergence precision.
        niter = int(np.ceil(
            np.log(np.abs(x1 - x0)/convergence) / np.log(2.0)
            ))

        for _ in range(niter):
            x2 = (x0 + x1) / 2
            f2 = func(x2)
            if f2 == 0:  # root found
                return x2
            if f1 * f2 < 0:  # sign change, move lower end point
                x0 = x2
                f0 = f2
            else:  # no sign change, move upper end point
                x1 = x2
                f1 = f2

        return (x0 + x1) / 2

    roots = []
    while len(roots) < maxnum:
        x0, x1 = _scan_interval(func, a, b, precision)
        if x0 is not None:  # valid sign change interval
            root = _find_root(func, x0, x1)
            if root is not None:
                roots.append(round(root, -int(np.log10(precision))))
            a = x1  # reset interval for next root
        else:  # no more sign change interval, terminate
            return np.asarray(roots)

    return np.asarray(roots, dtype=float)


def normalise_vector(vec, axis=-1):
    """Normalise vector arrays to unit vectors.

    Parameters
    ----------
    vec : float, array_like
        Vector to be normalised.
    axis : int
        Axis over which the Euclidean 2-norm is taken (default is -1, i.e.
        `vec` is assumed to be an array of row vectors).

    Returns
    -------
    float, array_like
        Unit vector.

    """
    return vec / norm(vec, axis=axis, keepdims=True)


def cartesian_to_spherical(cartesian_coords):
    r"""Convert 3-d Cartesian coordinate arrays to spherical coordinate arrays.

    The coordinate transformation is given by

    .. math::

        r = \sqrt{x^2 + y^2 + z^2} \,, \quad
        \theta = \arccos(z/r) \,, \quad
        \phi = \arctan(y/x) \,,

    where the image :math:`\arccos` is :math:`[0, \pi]`, and :math:`\arctan`
    has an extended image set :math:`[0, 2\pi)`.

    Parameters
    ----------
    cartesian_coords : float, array_like
        Cartesian coordinates.

    Returns
    -------
    spherical_coords : float, array_like
        Spherical coordinates.

    Raises
    ------
    ValueError
        If `cartesian_coords` is not of the correct dimensions.

    """
    c_coords = np.atleast_2d(cartesian_coords)
    if np.size(c_coords, axis=-1) != 3:
        raise ValueError(
            "`cartesian_coords` is not of the correct dimensions. "
            )

    spherical_coords = np.empty(c_coords.shape)
    spherical_coords[:, 0] = norm(c_coords, axis=1)
    spherical_coords[:, 1] = arccos(c_coords[:, 2] / spherical_coords[:, 0])
    spherical_coords[:, 2] = arctan2(c_coords[:, 1], c_coords[:, 0])

    return spherical_coords


def spherical_to_cartesian(spherical_coords):
    r"""Convert 3-d spherical coordinate arrays to Cartesian coordinate arrays.

    The coordinate transformation is given by

    .. math::

        x = r \sin\theta \cos\phi \,, \quad
        y = r \sin\theta \sin\phi \,, \quad
        z = r \cos\theta \,.

    Parameters
    ----------
    spherical_coords : float, array_like
        Spherical coordinates.

    Returns
    -------
    cartesian_coords : float, array_like
        Cartesian coordinates.

    Raises
    ------
    ValueError
        If `spherical_coords` is not of the correct dimensions.

    """
    s_coords = np.atleast_2d(spherical_coords)
    if np.size(s_coords, axis=-1) != 3:
        raise ValueError(
            "`spherical_coords` is not of the correct dimensions. "
            )

    cartesian_coords = np.empty(s_coords.shape)
    cartesian_coords[:, 0] = np.sin(s_coords[:, 1]) * np.cos(s_coords[:, 2])
    cartesian_coords[:, 1] = np.sin(s_coords[:, 1]) * np.sin(s_coords[:, 2])
    cartesian_coords[:, 2] = np.cos(s_coords[:, 1])

    cartesian_coords *= (s_coords[:, 0])[:, None]

    return cartesian_coords


def bin_edges_from_centres(centres, extremes, align='low'):
    """Convert bin centres to bin edges given extremities of the binning range,
    assumed to be positive.

    Parameters
    ----------
    centres : float, array_like
        Bin centers (``centres > 0``).
    extremes : tuple or list (of length 2)
        Binning range extremities (``extremes >= 0``).
    align : {'high', 'low'}, optional
        Conversion from high end of the binning range for ``'high'``, or from
        low end for ``'low'`` (default).

    Returns
    -------
    edges : float, array_like
        Bin edges.

    """
    if np.array(centres).ndim == 1:
        centres = np.squeeze(centres)
    if np.allclose(centres[0], 0):
        centres = np.delete(centres, 0)

    nbins = len(centres)
    edges = np.concatenate(
        ([np.min(extremes)], np.zeros(nbins-1), [np.max(extremes)])
        )
    if align.lower().startswith('l'):
        for bin_idx in range(nbins-1):
            edges[bin_idx+1] = 2*centres[bin_idx] - edges[bin_idx]
    elif align.lower().startswith('h'):
        for bin_idx in range(nbins-1):
            edges[-bin_idx-2] = 2*centres[-bin_idx-1] - edges[-bin_idx-1]

    return edges


def smooth_by_bin_average(data, bin_edges, x_coarse, y_coarse, dx_coarse=None,
                          dy_coarse=None):
    """Smooth data points by simple average binning in coordinates.

    Parameters
    ----------
    data : dict
        Data to be smoothed with binning.
    bin_edges : float, array_like
        Bin edges.
    x_coarse, y_coarse : str
        Dictionary key holding unsmoothed data coordinates or data points.
    dx_coarse, dy_coarse : str or None, optional
        Dictionary key holding data coordinate or data point uncertainties to
        be added in quadrature in bin without averaging.

    Returns
    -------
    smoothed_data : dict
        Smoothed quantities correspond to dictionary keys `x_coarse`,
        `y_coarse`, `dx_coarse` and `dy_coarse` if the keys are not `None`.
    count_in_bins : int, array_like
        Number of data points in each bin.

    Raises
    ------
    NotImplementedError
        If `data` is not of supported data types.

    """
    if not isinstance(data, dict):
        raise NotImplementedError(
            "Data types other than `dict` are currently unsupported. "
            )
    else:
        nbins = len(bin_edges) - 1
        x_coarse_lab, y_coarse_lab = x_coarse, y_coarse

        # Sort modes by scales.
        order = np.argsort(data[x_coarse])
        x_coarse = data[x_coarse][order]
        y_coarse = data[y_coarse][order]

        # Decide on which bin.
        which_bins = np.empty(x_coarse.shape)
        for idx, val in enumerate(x_coarse):
            which_bins[idx] = np.sum(val > bin_edges) - 1  # 0-indexed bins

        # Average in bins and count.
        x_smooth, y_smooth, count_in_bins = np.empty((3, nbins))
        for bin_idx in range(nbins):
            x_smooth[bin_idx] = np.average(x_coarse[which_bins == bin_idx])
            y_smooth[bin_idx] = np.average(y_coarse[which_bins == bin_idx])
            count_in_bins[bin_idx] = np.sum(which_bins == bin_idx)

        # Add uncertainties in quadrature in each bin if requested.
        smoothed_data = {
            x_coarse_lab: x_smooth,
            y_coarse_lab: y_smooth,
            }
        for key in [dx_coarse, dy_coarse]:
            if key is not None:
                coarse = data[key][order]
                smooth = np.empty(nbins)
                for bin_idx in range(nbins):
                    smooth[bin_idx] = np.sqrt(np.sum(
                        coarse[which_bins == bin_idx]**2
                        ))
                smoothed_data.update({key: smooth})

        return smoothed_data, count_in_bins
