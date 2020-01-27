"""
Utility tools (:mod:`~harmonia.collections.utils`)
===========================================================================

Provide utilities for input/output handling, processing, formatting and
data type manipulation, and common algebraic, geometric and statistical
algorithms.


System utilities
---------------------------------------------------------------------------

**File handling**

.. autosummary::

    confirm_directory_path
    get_filename
    collate_data_files
    overwrite_protection

**Multi-processing**

.. autosummary::

    allocate_tasks
    allocate_segments
    mpi_compute
    progress_status

**Formatting**

.. autosummary::

    clean_warning_format
    format_float

**Data type conversion**

.. autosummary::

    sort_dict_to_list
    sort_list_to_dict


Computational utilities
---------------------------------------------------------------------------

**Algebraic utilities**

.. autosummary::

    zero_const
    unit_const
    const_function
    mat_logdet
    covar_to_corr
    binary_search

**Geometrical utilities**

.. autosummary::

    normalise_vector
    spherical_indicator
    cartesian_to_spherical
    spherical_to_cartesian

**Statistical utilities**

.. autosummary::

    bin_edges_from_centres
    smooth_by_bin_average

|

"""
from __future__ import division

import os
import warnings
from collections import defaultdict
from pathlib import Path
from glob import glob

import numpy as np

__all__ = [
    'confirm_directory_path',
    'get_filename',
    'collate_data_files',
    'overwrite_protection',
    'allocate_tasks',
    'allocate_segments',
    'mpi_compute',
    'progress_status',
    'clean_warning_format',
    'format_float',
    'sort_dict_to_list',
    'sort_list_to_dict',
    'zero_const',
    'unit_const',
    'const_function',
    'mat_logdet',
    'covar_to_corr',
    'binary_search',
    'normalise_vector',
    'spherical_indicator',
    'cartesian_to_spherical',
    'spherical_to_cartesian',
    'bin_edges_from_centres',
    'smooth_by_bin_average',
]


# SYSTEM UTILITIES
# -----------------------------------------------------------------------------

def confirm_directory_path(dir_path):
    """Ensure a given directoy path exists.

    Parameters
    ----------
    dir_path : str or :class:`pathlib.Path`
        Directory path.

    Returns
    -------
    bool
        `True` if `dir_path` exists or has been created.

    """
    dir_path = Path(dir_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return os.path.exists(dir_path)


def get_filename(file_path):
    """Return file name without the directory path or file extension.

    Parameters
    ----------
    file_path : str or :class:`pathlib.Path`
        Full or partial file path.

    Returns
    -------
    str
        File name without extension.

    """
    return Path(file_path).stem


def collate_data_files(file_path_pattern, file_extension, headings=None,
                       columns=None):
    """Collate data files.

    For text files, the data is assumed to be stored as a column-major 2-d
    array with each column in correspondence with a key in the returned
    dictionary.

    Parameters
    ----------
    file_path_pattern : str
        Common substring of the data file paths.
    file_extension : {'npy', 'txt', 'dat'}
        Data file extension.
    headings : list of str or None, optional
        Data column headings to be used as dictionary keys (default is
        `None`).
    columns : list of int or None, optional
        Data column indices (zero-indexed) corresponding to headings
        (default is `None`).

    Returns
    -------
    collated_data : dict
        Collated data.
    collation_count : int
        Number of data files collated.
    last_file : str
        Last collated file name.

    Raises
    ------
    NotImplementedError
        If `file_extension` is not supported.
    ValueError
        If `file_extension` is ``'txt'`` or ``'dat'``, but either
        `headings` or `columns` is `None`.
    ValueError
        If `headings` and `columns` are not in correpondence.

    """
    all_files = glob(file_path_pattern)
    last_file = get_filename(all_files[-1])

    if file_extension.lower().endswith('npy'):
        all_data = [
            np.load(file, allow_pickle=True).item()
            for file in all_files
        ]

        collated_data = dict.fromkeys(all_data[-1].keys())
        for key in collated_data:
            to_concat = [np.atleast_1d(data[key]) for data in all_data]
            # to_concat = []
            # for data in all_data:
            #     block_entry = [
            #         line_entry for line_entry in data[key]
            #         if len(line_entry) == 6
            #     ]
            #     to_concat.append(np.atleast_1d(block_entry))
            collated_data[key] = np.concatenate(to_concat, axis=0)

        collation_count = max(map(len, collated_data.values()))

        return collated_data, collation_count, last_file

    if file_extension.lower().endswith(('txt', 'dat')):
        if headings is None or columns is None:
            raise ValueError(
                "`headings` or `columns` cannot be None "
                "when reading from text files. "
            )
        if len(headings) != len(columns):
            raise ValueError(
                "Lengths of `headings` and `columns` must agree. "
            )

        collated_data = defaultdict(list)
        for key_idx, key in enumerate(headings):
            to_concat = [
                np.atleast_2d(np.loadtxt(file, usecols=columns)[:, key_idx])
                for file in all_files
            ]
            collated_data[key] = np.concatenate(to_concat, axis=0)

        collation_count = max(map(len, collated_data.values()))

        return collated_data, collation_count, last_file

    raise NotImplementedError(
        f"Unsupported file extension: {file_extension}. "
    )


def overwrite_protection(outpath, outname):
    """Inspect and modify overwrite permission.

    The function may ask for permission from the user to overwrite the file
    path at runtime.

    Parameters
    ----------
    outpath : str or :class:`pathlib.Path`
        Write-out directory path.
    outname : str
        Write-out filename.

    Returns
    -------
    bool
        Write permission.

    """
    outpath = Path(outpath)

    if not os.path.exists(outpath):
        warnings.warn(f"{outpath} does not exist. ", RuntimeWarning)
        return True

    if not os.path.exists(outpath/outname):
        return True

    grant_permission = input(
        "Saving would overwrite existing file at destination. "
        "Do you want to continue? [y/n] "
    )

    if grant_permission.lower().startswith('y'):
        return True

    warnings.warn("Overwrite permission denied. File not saved. ")
    return False


def allocate_tasks(total_task, total_proc):
    """Allocate tasks to processes for parallel computation.

    If `total_proc` processes share `total_task` tasks, then this decides
    the numbers of tasks, `tasks`, different processes receive: the
    rank-``i`` process receives ``tasks[i]`` many tasks.

    Parameters
    ----------
    total_task : int
        Total number of tasks.
    total_proc : int
        Total number of processes.

    Returns
    -------
    tasks : list of int
        Number of tasks for each process.

    Raises
    ------
    TypeError
        If `total_task` or `total_proc` is not an integer.

    """
    try:
        total_task, total_proc = map(int, (total_task, total_proc))
    except TypeError:
        raise TypeError(
            "`total_task` and `total_proc` must have integer values. "
        )

    num_task_remaining, num_proc_remaining, tasks = total_task, total_proc, []

    while num_task_remaining > 0:
        num_task_assigned = num_task_remaining // num_proc_remaining
        tasks.append(num_task_assigned)
        num_task_remaining -= num_task_assigned
        num_proc_remaining -= 1

    return tasks


def allocate_segments(tasks=None, total_task=None, total_proc=None):
    """Allocate segments of tasks to each process by the number of tasks it
    receives and its rank.

    For instance, if the rank-``i`` process receives ``tasks[i]`` tasks
    (e.g. assigned by :func:`allocate_tasks`), then this function assigns a
    slice of the indexed tasks it should receive, with the indices in
    ascending order in correspondence with ranks of the processes.

    Parameters
    ----------
    tasks : list of int or None, optional
        Number of tasks each process receives.  Cannot be `None` if either
        `total_task` or `total_proc` is `None`.  If not `None`,
        `total_task` and `total_proc` are both ignored.
    total_task : int or None, optional
        Total number of tasks.  Ignored if `tasks` is not `None`, otherwise
        cannot be `None`.
    total_proc : int or None, optional
        Total number of processes.  Ignored if `tasks` is not `None`,
        otherwise cannot be `None`.

    Returns
    -------
    segments : list of slice
        Index slice of the segment of tasks that each process should
        receive.

    """
    if tasks is None:
        tasks = allocate_tasks(total_task, total_proc)
    if total_proc is None:
        total_proc = len(tasks)

    breakpoints = np.insert(np.cumsum(tasks), 0, values=0)
    segments = [
        slice(breakpoints[rank], breakpoints[rank+1])
        for rank in range(total_proc)
    ]

    return segments


def mpi_compute(data_array, mapping, comm, root=0, logger=None,
                process_name=None):
    """Multiprocess mapping of data.

    For each map to be applied, the input data array is scattered over the
    first axis for computation on difference process, and the computed
    results are gathered in the exact structure of the input data array on
    the root process.

    Parameters
    ----------
    data_array : array_like
        Data array.
    mapping : callable
        Mapping to be applied.
    comm : :class:`mpi4py.MPI.Comm`
        MPI communicator.
    root : int, optional
        Rank of the process taken as the root process (default is 0).
    logger : :class:`logging.Logger` or None, optional
        Logger (default is `None`).
    process_name : str or None
        If not `None` (default), this is the process name to be logged.

    Returns
    -------
    output_array : array_like or None
        Output data processed from `mapping`.  `None` for process ranks
        other than `root`.

    """
    if root + 1 > comm.size:
        root = 0
        warnings.warn(
            "Input `root` set to 0 as it exceeds the number of processes. "
        )

    segments = allocate_segments(
        total_task=len(data_array), total_proc=comm.size
    )
    data_chunk = data_array[segments[comm.rank]]

    output = []
    chunk_length = len(data_chunk)
    for piece_idx, data_piece in enumerate(data_chunk):
        output.append(mapping(data_piece))
        progress_status(
            piece_idx, chunk_length, logger,
            process_name=process_name, comm=comm, root=root,
        )

    comm.Barrier()

    output = comm.gather(output, root=root)

    output_array = None
    if comm.rank == root:
        output_array = np.concatenate(output, axis=0)

    output_array = comm.bcast(output_array, root=root)

    return output_array


def progress_status(current_idx, task_length, logger, process_name=None,
                    comm=None, root=0):
    """Log progress status.

    Parameters
    ----------
    current_idx : int
        Current index in the number of tasks.
    task_length : int
        Total number of tasks.
    logger : :class:`logging.Logger`
        Logger.
    process_name : str or None
        If not `None` (default), this is the process name to be logged.
    comm : :class:`mpi4py.MPI.Comm` or None, optional
        MPI communicator (default is `None`).
    root : int, optional
        Root process number (default is 0).

    """
    if process_name is not None:
        proc_name = "'{}' ".format(process_name)
    else:
        proc_name = ""

    if comm is None:
        logged_process = 'single'
    else:
        if comm.rank == root:
            logged_process = "first"
        elif comm.rank == comm.size - 1:
            logged_process = "last"
        else:
            logged_process = None

    block_length = max(task_length // 4, 1)
    progress_length = current_idx + 1
    progress_percentage = 100 * progress_length / task_length

    if logger and logged_process:
        if progress_length % block_length == 0 \
                or progress_length == task_length:
            logger.info(
                "Progress for the %s %sprocess: %d%% computed. ",
                logged_process, proc_name, progress_percentage
            )


def clean_warning_format(message, category, filename, lineno, line=None):
    """Clean warning message format.

    Parameters
    ----------
    message, category, filename, lineno : str
        Warning message, warning catagory, origin file name, line number.
    line : str or None, optional
        Source code line to be included in the warning message (default is
        `None`).

    Returns
    -------
    str
        Warning message format.

    """
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def format_float(x, case, use_sci_dp=3):
    r"""Format a float number as a string.

    Parameters
    ----------
    x : float
        Number to be formatted.
    case : {'latex', 'sci', 'intdot', 'decdot'}
        Format case, one of LaTeX (``'latex'``), scientific (``'sci'``),
        rounded integer ending with a decimal dot (``'intdot'``), or a
        float whose first decimal place is 0 represented as a rounded
        integer ending with a decimal dot (``'decdot'``).
    use_sci_dp : int, optional
        Number of decimal places beyond which the scientific notation
        is used instead of the decimal notation when `case` is ``'sci'``.
        Default is 3.

    Returns
    -------
    x_str : str
        Formatted string.

    Raises
    ------
    ValueError
        If `case` is not one of the supported formats.

    """
    x = float(x)

    def _decimal_to_sci_switch(x, use_sci_dp):
        """Switch between the decimal and scientific representation of a
        float number.

        Parameters
        ----------
        x : float
            Float number to be recast.
        use_sci_dp : int
            Number of decimal places beyond which the scientific notation
            is used instead of the decimal representation.

        Returns
        -------
        x_str : str
            String representation of the float number.

        """
        if x == 0.:
            exponent, base = 0., 0.
        else:
            exponent = int(np.floor(np.log10(np.abs(x))))
            base = x/10**exponent

        if 0 <= exponent <= use_sci_dp:
            x_str = "{:.1f}".format(x).rstrip("0")
        elif - use_sci_dp <= exponent < 0:
            base = np.around(base, decimals=1)
            x_str = "{:f}".format(base*10**exponent).rstrip("0")
        else:
            base_str = "{:.2f}".format(base).rstrip("0")
            x_str = "{0}e{1}".format(base_str, exponent)

        return x_str

    if case.lower() == 'latex':
        x_str = _decimal_to_sci_switch(x, use_sci_dp)
        if "e" in x_str:
            base_str, exp_str = x_str.split("e")
            if base_str.endswith("."):
                base_str += "0"
            x_str = r"{0} \times 10^{{{1}}}".format(base_str, exp_str)
    elif case.lower() == 'sci':
        x_str = _decimal_to_sci_switch(x, use_sci_dp)
    elif case.lower() == 'intdot':
        x_str = "{}".format(np.around(x)).rstrip("0")
    elif case.lower() == 'decdot':
        x_str = "{:.1f}".format(x).rstrip("0")
    else:
        raise ValueError(f"Invalid `case` value: {case}. ")

    return x_str


def sort_dict_to_list(dict_data):
    """Sort dictionary values by key in ascending order to a list.

    Parameters
    ----------
    dict_data : dict
        Dictionary data to be sorted.

    Returns
    -------
    sorted_list : list of array_like
        `dict_data` values sorted by its keys.

    Raises
    ------
    TypeError
        If `dict_data` is not a dictionary.

    """
    try:
        sorted_list = [dict_data[key] for key in sorted(dict_data.keys())]
    except AttributeError:
        raise TypeError("`dict_data` must be a dictionary. ")

    return sorted_list


def sort_list_to_dict(list_data, int_keys):
    """Convert an ordered list to a dictionary with integer keys in
    correpondence with the list index.

    Parameters
    ----------
    list_data : list of array_like
        Ordered list-like data.
    int_keys : list of int
        Integer keys in correpondence with the list index.

    Returns
    -------
    sorted_dict : dict
        Dictionary with integer keys in correspondence with `list_data`
        indices.

    Raises
    ------
    ValueError
        If the lengths of `list_data` and `int_keys` do not match.

    """
    if len(list_data) != len(int_keys):
        raise ValueError("`list_data` and `int_keys` lengths do not match. ")

    sorted_dict = {
        int_keys[order_idx]: list_data[order_idx]
        for order_idx in np.argsort(int_keys)
    }

    return sorted_dict


# COMPUTATIONAL UTILITIES
# -----------------------------------------------------------------------------

def zero_const(*args, **kwargs):
    """Return constant 0 with arbitrary arguments.

    Parameters
    ----------
    *args, **kwargs
        Arbitrary arguments.

    Returns
    -------
    float
        0.

    """
    return 0.


def unit_const(*args, **kwargs):
    """Return constant 1 with arbitrary arguments.

    Parameters
    ----------
    *args, **kwargs
        Arbitrary arguments.

    Returns
    -------
    float
        1.

    """
    return 1.


def const_function(const):
    """Return a constant function with arbitrary arguments.

    Parameters
    ----------
    const : float
        Constant value.

    Returns
    -------
    callable
        Constant function.

    """
    return lambda *args, **kwargs: const


def mat_logdet(matrix, diag=False):
    """Calculate logarithm of the determinant of a positive-definite
    matrix.

    Parameters
    ----------
    matrix : float or complex, array_like
        Positive definite 2-d matrix.
    diag : bool, optional
        If `True` (default is `False`), the input matrix is assumed to be
        diagonal.

    Returns
    -------
    log_det : float
        Logarithm of the matrix determinant.

    Raises
    ------
    ValueError
        If `matrix` is not in an equivalent shape for a 2-d square matrix.
    ValueError
        If `matrix` is not positive definite.

    """
    if np.array(matrix).ndim != 2 or len(set(np.shape(matrix))) != 1:
        raise ValueError("`matrix` is not a 2-d square matrix. ")

    if diag:
        sign_det = np.prod(np.sign(np.diag(matrix)))
        log_det = np.sum(np.log(np.abs(np.diag(matrix))))
    else:
        sign_det, log_det = np.linalg.slogdet(matrix)

    if sign_det != 1.:
        raise ValueError("`matrix` is not positive definite. ")

    return log_det


def covar_to_corr(covar):
    """Convert a real-valued covariance matrix to a correlation matrix.

    Parameters
    ----------
    covar : float, array_like
        Covariance matrix.

    Returns
    -------
    corr : float, array_like
        Correlation matrix.

    Raises
    ------
    NotImplementedError
        If `covar` is complex.

    """
    if np.iscomplexobj(covar):
        raise NotImplementedError(
            "Complex covariance matrices are not supported. "
        )

    inv_diag = np.diag(np.power(np.diag(covar), -1/2))
    corr = inv_diag @ covar @ inv_diag

    return corr


def binary_search(func, a, b, maxnum=None, precision=1.e-5):
    """Binary seach for all zeros of a function in a real interval.

    Parameters
    ----------
    func : callable
        Function whose zeros are to be found.
    a, b : float
        Interval end points, ``a < b``.
    maxnum : int or None, optional
        Maximum number of zeros needed from below (default is `None`).
    precision : float, optional
        Desired precision of the zeros (default is 1.e-5).

    Returns
    -------
    roots : float :class:`numpy.ndarray` or None
        Possible roots.

    Raises
    ------
    ValueError
        If the initial interval covers only one point (``a == b``).

    """
    if maxnum is None:
        maxnum = np.iinfo(np.int64).max

    if a == b:
        raise ValueError(
            f"Initial interval covers only one point: [{a}, {b}]. "
        )
    if a > b:
        a, b = b, a
        warnings.warn(
            f"Initial interval [{a}, {b}] reordered to [{b}, {a}]. "
        )

    def _scan_interval(func, a, b, dx):
        """Scan an interval from the lower end to detect sign changes.

        Parameters
        ----------
        func : callable
            Function whose sign-change interval is to be found.
        a, b: float
            Initial interval end points, ``a < b``.
        dx : float
            Increment from the interval lower end, ``dx > 0``.

        Returns
        -------
        x_low, x_high : float or None
            End points for a sign-change interval, ``x_low < x_high``.
            `None` if the result is null.

        """
        x_low, x_high = a, a + dx

        f_low, f_high = func(x_low), func(x_high)
        while f_low * f_high >= 0:
            if x_low >= b:
                return None, None
            x_low, x_high = x_high, x_high + dx
            f_low, f_high = f_high, func(x_high)

        return x_low, x_high

    def _find_root(func, x_low, x_high, convergence=1.e-9):
        """Bisection method for root finding.

        Parameters
        ----------
        func : callable
            Function whose zero bracket is to be found.
        x_low, x_high: float
            Initial interval end points.
        convergence : float, optional
            Convergence precision for setting maximum iteration number
            (default is 1.e-9).

        Returns
        -------
        float or None
            Single possible root.

        """
        f_low, f_high = func(x_low), func(x_high)
        if f_low == 0:
            return x_low
        if f_high == 0:
            return x_high
        if f_low * f_high > 0:
            warnings.warn("Root is not bracketed. ", RuntimeWarning)
            return None

        maxiter = int(np.log((x_high - x_low)/convergence) / np.log(2) + 1)
        for _ in range(maxiter):
            x_middle = (x_low + x_high) / 2
            f_middle = func(x_middle)
            if f_middle == 0:
                return x_middle
            if f_high * f_middle < 0:
                x_low = x_middle
                f_low = f_middle
            else:
                x_high = x_middle
                f_high = f_middle

        return (x_low + x_high) / 2

    roots = []
    while len(roots) < maxnum:
        x_low, x_high = _scan_interval(func, a, b, precision)
        if x_low is not None:
            root = _find_root(func, x_low, x_high)
            if root is not None:
                roots.append(round(root, int(-np.log10(precision))))
            a = x_high
        else:
            break

    return np.array(roots, dtype=float)


def normalise_vector(vector, axis=-1):
    """Normalise vector arrays to unit vectors.

    Parameters
    ----------
    vector : float, array_like
        Vector to be normalised.
    axis : int
        Axis over which the Euclidean 2-norm is taken (default is -1).

    Returns
    -------
    unit_vector : float :class:`numpy.ndarray`
        Unit vector.

    """
    unit_vector = vector / np.linalg.norm(vector, axis=axis, keepdims=True)

    return unit_vector


def spherical_indicator(cartesian_position, bounding_radius):
    """Indicate whether an object lies within the bounding radius of a
    spherical domain.

    Parameters
    ----------
    cartesian_position : float, array_like
        Object position in Cartesian coordinates.
    bounding_radius : float
        Bounding radius of the spherical domain.

    Returns
    -------
    indication : bool :class:`numpy.ndarray`
        `True` if the object lies within the spherical domain.

    """
    indication = (
        np.linalg.norm(cartesian_position, axis=-1) <= bounding_radius
    )

    return indication


def cartesian_to_spherical(cartesian_coords):
    r"""Convert 3-d Cartesian coordinate arrays to spherical coordinate
    arrays.

    The coordinate transformation is given by

    .. math::

        r = \sqrt{x^2 + y^2 + z^2} \,, \quad
        \theta = \arccos(z/r) \,, \quad
        \phi = \arctan(y/x) \,,

    where the image of :math:`\arccos` is :math:`[0, \pi]`, and
    :math:`\arctan` has an extended image set :math:`[0, 2\pi]`.

    Parameters
    ----------
    cartesian_coords : float, array_like
        Cartesian coordinates.

    Returns
    -------
    spherical_coords : float :class:`numpy.ndarray`
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

    spherical_coords = np.zeros(c_coords.shape)
    spherical_coords[:, 0] = np.linalg.norm(c_coords, axis=1)
    spherical_coords[:, 1] = np.arccos(c_coords[:, 2] / spherical_coords[:, 0])
    spherical_coords[:, 2] = np.mod(
        np.arctan2(c_coords[:, 1], c_coords[:, 0]), 2*np.pi
    )

    return spherical_coords


def spherical_to_cartesian(spherical_coords):
    r"""Convert 3-d spherical coordinate arrays to Cartesian coordinate
    arrays.

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
    cartesian_coords : float :class:`numpy.ndarray`
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

    cartesian_coords = np.zeros(s_coords.shape)
    cartesian_coords[:, 0] = np.sin(s_coords[:, 1]) * np.cos(s_coords[:, 2])
    cartesian_coords[:, 1] = np.sin(s_coords[:, 1]) * np.sin(s_coords[:, 2])
    cartesian_coords[:, 2] = np.cos(s_coords[:, 1])

    cartesian_coords *= (s_coords[:, 0])[:, None]

    return cartesian_coords


def bin_edges_from_centres(centres, extremes, align='low'):
    """Convert bin centres to bin edges given extremities of the binning
    range, assumed to be positive.

    Parameters
    ----------
    centres : float, array_like
        Bin centers (``centres > 0``).
    extremes : tuple or list
        Binning range extremities (``extremes >= 0``).
    align : {'high', 'low'}, optional
        Conversion from high end of the binning range for ``'high'``, or
        from low end for ``'low'`` (default).

    Returns
    -------
    edges : float :class:`numpy.ndarray`
        Bin edges.

    """
    if np.array(centres).ndim == 1:
        centres = np.squeeze(centres)
    if np.allclose(centres[0], 0):
        centres = np.delete(centres, 0)

    num_bin = len(centres)
    edges = np.concatenate(
        ([np.min(extremes)], np.zeros(num_bin-1), [np.max(extremes)])
    )
    if align.lower().startswith('l'):
        for bin_idx in range(num_bin-1):
            edges[bin_idx+1] = 2*centres[bin_idx] - edges[bin_idx]
    elif align.lower().startswith('h'):
        for bin_idx in range(num_bin-1):
            edges[-bin_idx-2] = 2*centres[-bin_idx-1] - edges[-bin_idx-1]

    return edges


def smooth_by_bin_average(data, bin_edges, x_coarse_key, y_coarse_key,
                          dx_coarse_key=None, dy_coarse_key=None):
    """Smooth data points by simple average binning in coordinates, with
    optional binning of uncertainties in quadrature.

    Parameters
    ----------
    data : dict
        Data to be smoothed with binning.
    bin_edges : float, array_like
        Bin edges.
    x_coarse_key, y_coarse_key : str
        Dictionary key holding unsmoothed data points.
    dx_coarse_key, dy_coarse_key : str or None, optional
        Dictionary key holding data point uncertainties to be binned
        (default is `None`).

    Returns
    -------
    smoothed_data : dict
        Smoothed quantities correspond to dictionary keys `x_coarse_key`
        and `y_coarse_key`, and additionally `dx_coarse_key` and
        `dy_coarse_key` if the keys are not `None`.
    bin_count : int :class:`numpy.ndarray`
        Number of data points in each bin.

    Raises
    ------
    NotImplementedError
        If `data` does not have a supported data type.

    """
    if not isinstance(data, dict):
        raise NotImplementedError(
            "Data types other than `dict` are unsupported. "
        )
    else:
        order = np.argsort(data[x_coarse_key])
        x_coarse = data[x_coarse_key][order]
        y_coarse = data[y_coarse_key][order]

        which_bins = np.zeros(x_coarse.shape)
        for idx, val in enumerate(x_coarse):
            which_bins[idx] = np.sum(val > bin_edges) - 1  # zero-indexed bins

        num_bin = len(bin_edges) - 1
        x_smooth, y_smooth, bin_count = np.zeros((3, num_bin))
        for bin_idx in range(num_bin):
            x_smooth[bin_idx] = np.average(x_coarse[which_bins == bin_idx])
            y_smooth[bin_idx] = np.average(y_coarse[which_bins == bin_idx])
            bin_count[bin_idx] = np.sum(which_bins == bin_idx)

        smoothed_data = {x_coarse_key: x_smooth, y_coarse_key: y_smooth}
        for key in [dx_coarse_key, dy_coarse_key]:
            if key is not None:
                coarse = data[key][order]
                smooth = np.zeros(num_bin)
                for bin_idx in range(num_bin):
                    smooth[bin_idx] = np.sqrt(
                        np.sum(coarse[which_bins == bin_idx]**2)
                    )
                smoothed_data.update({key: smooth})

        return smoothed_data, bin_count
