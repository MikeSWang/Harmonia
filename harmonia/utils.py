"""
Utilities (:mod:`~harmonia.utils`)
===========================================================================

Provide utilities for processing and logging as well as common mathematical
function and algorithms.


**Processing and monitoring**

.. autosummary::

    Progress
    setup_logger
    clean_warning_format
    restore_warnings
    mpi_compute


**Mathematics**

.. autosummary::

    const_function
    binary_search
    covar_to_corr
    mat_logdet
    PositiveDefinitenessWarning
    is_positive_definite

|

"""
import logging
import sys
import time
import warnings

import numpy as np
from tqdm import tqdm

__all__ = [
    'Progress',
    'setup_logger',
    'clean_warning_format',
    'restore_warnings',
    'mpi_compute',
    'const_function',
    'binary_search',
    'covar_to_corr',
    'mat_logdet',
    'PositiveDefinitenessWarning',
    'is_positive_definite',
]


# Processing and monitoring utilities
# -----------------------------------------------------------------------------

class Progress:
    """Progress status of tasks.

    If multiple parallel processes exist, progress status is only reported
    for the first and last of them.

    Parameters
    ----------
    task_length : int
        Total number of tasks.
    num_checkpts : int, optional
        Number of checkpoints for reporting progress (default is 4).
    process_name : str or None, optional
        If not `None` (default), this is the process name to be logged.
    logger : :class:`logging.Logger` *or None, optional*
        Logger.  If `None` (default), a print statement is issued.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator (default is `None`).
    root : int, optional
        Root process number (default is 0).

    Attributes
    ----------
    process_name : str or None, optional
        If not `None` (default), this is the process name to be logged.
    task_length : int
        Total number of tasks.
    progress_checkpts : float
        Scheduled progress check points, ``0 < progress_checkpts <= 1``.
    last_checkpt : int
        Index of the last passed checkpoint,
        ``0 <= last_checkpt <= num_checkpts``.

    Examples
    --------
    >>> ntasks = 100
    >>> p = Progress(ntasks, process_name='null test')
    >>> for task_idx in range(ntasks):
    ...     p.report(task_idx)
    Progress for the single 'null test' process: 25% computed.
    Progress for the single 'null test' process: 50% computed.
    Progress for the single 'null test' process: 75% computed.
    Progress for the single 'null test' process: 100% computed.

    """

    def __init__(self, task_length, num_checkpts=4, process_name=None,
                 logger=None, comm=None, root=0):

        self.process_name = process_name
        self.task_length = task_length
        self.logger = logger

        if self.process_name is None:
            self._proc_name = ""
        else:
            self._proc_name = "'{}' ".format(process_name)

        if comm is None:
            self._which_proc = 'single'
        else:
            if comm.rank == root:
                self._which_proc = "first"
            elif comm.rank == comm.size - 1:
                self._which_proc = "last"
            elif comm.rank == comm.size // 2 + 1:
                self._which_proc = "middle"
            else:
                self._which_proc = None

        self.progress_checkpts = \
            np.linspace(1. / num_checkpts, 1., num=num_checkpts)
        self.last_checkpt = 0

        self._progressor = self._initialise()

    def report(self, current_position):
        """Report the current position in the tasks.

        Parameters
        ----------
        current_position : int
            Index of the current position in the tasks (starting from 0).

        """
        next(self._progressor)
        self._progressor.send(current_position)

    def _initialise(self):

        while True:
            current_idx = yield

            current_progress = (current_idx + 1) / self.task_length
            place_in_checkpts = np.searchsorted(
                self.progress_checkpts, current_progress, side='right'

            )

            if place_in_checkpts > self.last_checkpt \
                    and self._which_proc is not None:
                if self.logger is None:
                    print(
                        "Progress for the {} {}process: {:.0f}% computed."
                        .format(
                            self._which_proc, self._proc_name,
                            100 * current_progress
                        )
                    )
                else:
                    self.logger.info(
                        "Progress for the %s %sprocess: %.0f%% computed.",
                        self._which_proc, self._proc_name,
                        100 * current_progress
                    )
                self.last_checkpt = place_in_checkpts
            yield


class _LoggerFormatter(logging.Formatter):
    """Customised logging formatter.

    """

    _start_time = time.time()

    def format(self, record):
        """Modify the default logging record by adding elapsed time in
        hours, minutes and seconds.

        Parameters
        ----------
        record : :class:`Logging.LogRecord`
            Default logging record object.

        Returns
        -------
        str
            Modified record message with elapsed time.

        """
        elapsed_time = record.created - self._start_time
        h, remainder_time = divmod(elapsed_time, 3600)
        m, s = divmod(remainder_time, 60)

        record.elapsed = "(+{}:{:02d}:{:02d})".format(int(h), int(m), int(s))

        return logging.Formatter.format(self, record)


def setup_logger():
    """Return the root logger formatted with elapsed time and piped
    to ``stdout``.

    Returns
    -------
    logger : :class:`logging.Logger`
        Formatted root logger.

    """
    logger = logging.getLogger()
    logging_handler = logging.StreamHandler(sys.stdout)
    logging_formatter = _LoggerFormatter(
        fmt='[%(asctime)s %(elapsed)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_handler)

    return logger


# pylint: disable=unused-argument
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
    filename = filename if "harmonia" not in filename \
        else "".join(filename.partition("harmonia")[1:])

    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def restore_warnings(captured_warnings):
    """Emit captured warnings.

    Parameters
    ----------
    captured_warnings : *list of* :class:`warnings.WarningMessage`
        List of recorded warnings as returned by
        ``warnings.catch_warnings(record=True)``.

    """
    for record in captured_warnings:
        warnings.showwarning(
            record.message, record.category, record.filename, record.lineno,
            file=record.file, line=record.line
        )


def _allocate_tasks(total_task, total_proc):
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

    """
    try:
        total_task, total_proc = map(int, (total_task, total_proc))
    except TypeError as err:
        raise TypeError(
            "`total_task` and `total_proc` must have integer values."
        ) from err

    num_task_remaining, num_proc_remaining, tasks = total_task, total_proc, []

    while num_task_remaining > 0:
        num_task_assigned = num_task_remaining // num_proc_remaining
        tasks.append(num_task_assigned)
        num_task_remaining -= num_task_assigned
        num_proc_remaining -= 1

    return tasks


def _allocate_segments(tasks=None, total_task=None, total_proc=None):
    """Allocate segments of tasks to each process by the number of tasks it
    receives and its rank.

    For instance, if the rank-``i`` process receives ``tasks[i]`` tasks
    (e.g. assigned by :func:`_allocate_tasks`), then this function assigns
    a slice of the indexed tasks it should receive, with the indices in
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
        tasks = _allocate_tasks(total_task, total_proc)
    if total_proc is None:
        total_proc = len(tasks)

    breakpoints = np.insert(np.cumsum(tasks), 0, values=0)
    segments = [
        slice(breakpoints[rank], breakpoints[rank + 1])
        for rank in range(total_proc)
    ]

    return segments


def mpi_compute(data_array, mapping, comm=None, root=0, logger=None,
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
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator.  If `None`, no multiprocessing is performed.
    root : int, optional
        Rank of the process taken as the root process (default is 0).
    logger : :class:`logging.Logger` *or None, optional*
        Logger (default is `None`).
    process_name : str or None
        If not `None` (default), this is the process name to be logged.

    Returns
    -------
    output_array : array_like or None
        Output data processed from `mapping`.  Returns `None` for process
        ranks other than `root`.

    """
    if comm is None or comm.size == 1:
        if process_name is not None:
            process_name = process_name.capitalize()

        output_array = list(tqdm(
            map(mapping, data_array), total=len(data_array), mininterval=1,
            desc=process_name, file=sys.stdout
        ))

        return output_array

    if root + 1 > comm.size:
        root = 0
        warnings.warn(
            "Input `root` set to 0 as it exceeds the number of processes."
        )

    segments = _allocate_segments(
        total_task=len(data_array), total_proc=comm.size
    )

    data_chunk = data_array[segments[comm.rank]]

    progress = Progress(
        len(data_chunk), process_name=process_name,
        logger=logger, comm=comm, root=root
    )
    output = []
    for data_idx, data_unit in enumerate(data_chunk):
        output.append(mapping(data_unit))
        progress.report(data_idx)

    comm.Barrier()

    output = comm.gather(output, root=root)

    output_array = None
    if comm.rank == root:
        try:
            output_array = np.concatenate(output, axis=0)
        except ValueError:
            output_array = [
                array_in_block
                for output_block in output
                for array_in_block in output_block
            ]

    output_array = comm.bcast(output_array, root=root)

    return output_array


# Mathematical functions and algorithms
# -----------------------------------------------------------------------------

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
    # Incremental interval.
    x_low, x_high = a, a + dx
    f_low, f_high = func(x_low), func(x_high)
    # Continue interval increments unless sign changes.
    while f_low * f_high >= 0:
        # Terminate when incremental interval goes outside the
        # overall interval.
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

    # Trivial termination scenarios.
    if f_low == 0:
        return x_low
    if f_high == 0:
        return x_high
    if f_low * f_high > 0:
        warnings.warn("Root is not bracketed.", RuntimeWarning)
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


def binary_search(func, a, b, maxnum=None, precision=1.e-6):
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
        Desired absolute precision of the zeros (default is 1.e-8).

    Returns
    -------
    roots : float :class:`numpy.ndarray` or None
        Possible roots.

    Raises
    ------
    ValueError
        If the initial interval covers only one point (``a == b``).

    """
    # Maximum number of roots for definite termination.
    if maxnum is None:
        maxnum = np.iinfo(np.int32).max

    # Check overall interval is valid.
    if a == b:
        raise ValueError(
            f"Initial interval covers only one point: [{a}, {b}]."
        )
    if a > b:
        a, b = b, a
        warnings.warn(
            f"Initial interval [{a}, {b}] reordered to [{b}, {a}]."
        )

    roots = []
    while len(roots) < maxnum:
        x_low, x_high = _scan_interval(func, a, b, precision)
        if x_low is not None:
            root = _find_root(func, x_low, x_high)
            if root is not None:
                roots.append(round(root, int(- np.log10(precision))))
            a = x_high
        else:
            break

    return np.asarray(roots, dtype=float)


def covar_to_corr(covar):
    """Convert a covariance matrix to its correlation matrix.

    Parameters
    ----------
    covar : complex, array_like
        Covariance matrix.

    Returns
    -------
    corr : :class:`numpy.ndarray`
        Correlation matrix.

    """
    inv_diag = np.diag(np.power(np.abs(np.diag(covar)), -1/2))
    corr = inv_diag @ covar @ inv_diag

    return corr


def mat_logdet(matrix):
    """Calculate logarithm of the determinant of a matrix.

    Parameters
    ----------
    matrix : float or complex, array_like
        Matrix.

    Returns
    -------
    log_det : float
        Logarithm of the matrix determinant.

    """
    _, log_det = np.linalg.slogdet(matrix)

    return log_det


class PositiveDefinitenessWarning(UserWarning):
    """Emit a warning when a matrix is not positive definite.

    """


def is_positive_definite(matrix):
    """Check the positive definiteness of a square matrix by attempting a
    Cholesky decomposition.

    Parameters
    ----------
    matrix : float or complex, array_like
        Matrix.

    Returns
    -------
    bool
        Positive definiteness.

    """
    try:
        _ = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    else:
        return True
