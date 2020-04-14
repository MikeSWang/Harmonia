"""
Structured arrays (:mod:`~harmonia.algorithms.arrays`)
===========================================================================

Provide structured arrays for cosmological data.

.. autosummary::

    DataArray
    SphericalArray
    CartesianArray

|

"""
from collections.abc import Sequence
from itertools import product

import numpy as np

from .discretisation import DiscreteSpectrum

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class IndexingError(Exception):
    """Exception raised for unsupported slicing or indexing in
    `__getitem__` methods.

    """


class DataArray:
    """Abstract structured array with saving and loading methods.

    """

    def __setstate__(self, state):

        raise NotImplementedError

    def __getstate__(self):

        raise NotImplementedError

    def save(self, output_file, file_extension):
        """Save the structured array.

        Parameters
        ----------
        output_file : *str or* :class:`pathlib.Path`
            Output file path.
        extension : {'pkl', 'npz'}
            Output file extension.

        """
        if file_extension == 'pkl':
            with open(output_file, 'wb') as output_data:
                pickle.dump(self, output_data, protocol=-1)
        elif file_extension == 'npz':
            np.savez(output_file, **self.__getstate__())
        else:
            raise IOError(
                "Unwritable output file. "
                "The file extension must be either .npz or .pkl."
            )

    @classmethod
    def load(cls, input_file):
        """Load the structured array from a .npz or .pkl file.

        Parameters
        ----------
        input_file : *str or* :class:`pathlib.Path`
            Input file path.

        """
        try:
            extension = input_file.suffix
        except AttributeError:
            extension = input_file.rpartition(".")[-1]

        if extension.endswith('npz'):
            state_data = np.load(input_file)
            state = {}
            for attr in state_data.files:
                try:
                    state.update({attr: state_data[attr].item()})
                except ValueError:
                    state.update({attr: state_data[attr]})
            self = object.__new__(cls)
            self.__setstate__(state)
        elif extension.endswith('pkl'):
            with open(input_file, 'rb') as input_data:
                self = pickle.load(input_data)
        else:
            raise IOError(
                "Unreadable input file. "
                "The file extension must be either .npz or .pkl."
            )

        return self


class SphericalArray(DataArray):
    r"""Structured array for spherical decomposition of cosmological data.

    Array is initialised with a discrete spectrum of spherical modes
    and consists of three fields: the 'index' field of
    :math:`(\ell, m_\ell, n_\ell)` triplets, the 'wavenumber' field
    of discrete :math:`k_{\ell n}`, and the 'coefficient' field of
    spherically decomposed data.

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the structured array.

    Attributes
    ----------
    array : :class:`numpy.ndarray`
        Structured NumPy array.
    size : int
        Total number of elements in the array.  This should equal the sum
        of ``disc.mode_counts``.
    attrs : dict
        Attributes of the structured array inherited from `disc`.

    See Also
    --------
    :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`

    """

    # Class-wide ``numpy.dtype`` for the structured array and any
    # collapsed array.
    _dtype = np.dtype({
        'names': ['index', 'wavenumber', 'coefficient', '_position'],
        'formats': ['(3,)i4', 'f8', 'c16', 'i4'],
    })

    _dtype_collapsed = np.dtype({
        'names': ['index', 'wavenumber', 'coefficient', '_position'],
        'formats': ['(2,)i4', 'f8', 'c16', 'i4'],
    })

    def __init__(self, disc):

        self.disc = disc
        self.size = sum(disc.mode_counts)

        self.attrs = {
            attr: getattr(disc, attr)
            for attr in ['degrees', 'depths', 'wavenumbers', 'mode_counts']
        }

        self.array, self._directory = self._initialise_array()

    def __str__(self):

        return f"{self.__class__.__name__}({str(self.disc)})"

    def __setitem__(self, key, value):
        """Set the 'coefficient' field value(s).

        Parameters
        ----------
        key : int, tuple of int or slice
            Coefficient access key.
        value : complex
            Complex coefficient entry.

        See Also
        --------
        :meth:`__getitem__`

        """
        position = self._find_position(key)

        self.array['coefficient'][position] = value

    def __getitem__(self, key):
        """Access the 'coefficient' field.

        The access key can be an integer, a slice expression, a tuple of
        index triplet or a string, e.g. ``[-1]``, ``[:]``, ``[(0, 0, 1)]``
        or ``'degree_0'``.

        Parameters
        ----------
        key : int, slice, tuple(int, int, int) or str
            'coefficient' field access key.

        Returns
        -------
        complex
            'coefficient' field data entry.

        """
        position = self._find_position(key)

        return self.array['coefficient'][position]

    def __setstate__(self, state):

        for attr, value in state.items():
            if attr == 'disc':
                self.disc = DiscreteSpectrum._from_state(state['disc'])
            else:
                setattr(self, attr, value)

    def __getstate__(self):

        state = self.__dict__

        state.update({'disc': self.disc.__getstate__()})

        return state

    @classmethod
    def _from_state(cls, state):  # internal classmethod

        self = object.__new__(cls)

        self.__setstate__(state)

        return self

    def vectorise(self, pivot, collapse=None):
        r"""Returrn a data vector from the 'coefficient' field.

        Vectorisation is performed by *pivoting* in two orders of
        precedence---

            * 'natural': ordered by :math:`(\ell, m, n)`;
            * 'spectral': ordered by :math:`(k_{\ell n}, m)`.

        Subarrays of equivalent :math:`(\ell, n)` may be further collapsed
        over spherical order :math:`m` by averaging or averaging in
        quadrature.

        Parameters
        ----------
        pivot : {'natural', 'spectral'}
            Pivot order for vectorisation.
        collapse : {None, 'mean', 'qaudrature'}, optional
            If not `None` (default), subarrays are collapsed over
            equivalent spherical order :math:`m` by averaging ('mean') or
            averaging in quadrature ('qaudrature').

        Returns
        -------
        vectorised_data : :class:`numpy.ndarray`
            Vectorised coefficient data.

        """
        if collapse is None:
            array = self.array
        else:
            # Initialise the collapsed array.
            doublet_list = self._gen_index_list(
                self.disc.degrees, self.disc.depths, reduce=True
            )

            array = np.empty(len(doublet_list), dtype=self._dtype_collapsed)
            array['index'] = doublet_list

            for pos, ind in enumerate(array['index']):
                # Extract subarray matching the degree and depth of the
                # index doublet being considered.
                selector = np.all(
                    self.array['index'][:, [0, 2]] == ind, axis=-1
                )
                subarray = self.array['coefficient'][selector]

                # Collapse the extracted subarray.
                if collapse.lower() == 'mean':
                    collapsed_subarray = np.mean(subarray)
                elif collapse.lower() == 'qaudrature':
                    collapsed_subarray = np.mean(np.abs(subarray) ** 2)
                else:
                    raise ValueError(f"Unknown `collapse` option: {collapse}.")

                array['coefficient'][pos] = collapsed_subarray
                array['wavenumber'][pos] = \
                    self.disc.wavenumbers[tuple(ind)]

        # Sort array by the pivot order.  It is unsafe to sort by the
        # 'index' field which is multi-dimensional and becomes scrambled.
        # The private field '_position' is initialised for this purpose.
        if pivot == 'natural':
            sort_order = ['_position', 'wavenumber']
        elif pivot == 'spectral':
            sort_order = ['wavenumber', '_position']
        else:
            raise ValueError(f"Unknown `pivot` option: {pivot}.")

        vectorised_data = np.sort(array, order=sort_order)['coefficient']

        return vectorised_data

    def _initialise_array(self):

        # Initialise the structured array in the natural order.
        array = np.empty(self.size, dtype=self._dtype)

        # Generate the list of index triplets and the dictionary of index
        # directory.
        triplet_list = self._gen_index_list(
            self.disc.degrees, self.disc.depths
        )
        triplet_directory = self._gen_index_directory(triplet_list)

        # Initialise index triplets and corresponding wavenumbers.
        array['index'] = triplet_list
        array['wavenumber'] = [
            self.disc.wavenumbers[ell, n] for ell, _, n in triplet_list
        ]
        array['_position'] = list(range(self.size))

        return array, triplet_directory

    def _find_position(self, key):

        # If accessed by integer, reinterprete negative values as reverse
        # indexing.
        if isinstance(key, int):
            position = key if key >= 0 else key % self.size
            if position > self.size:
                raise IndexError(
                    f"Index {position} out of bound for key: {key}."
                )
            return position

        # Access by slice.
        if isinstance(key, slice):
            return key

        # If accessed by a sequence of (degree, order, depth); any list
        # is turned into a tuple.
        if isinstance(key, Sequence) and not isinstance(key, str):
            position = self._directory[tuple(key)]
            return position

        # Access by string of the form 'degree_<ell>'.
        if isinstance(key, str):
            degree = int(key.split('_')[-1])
            degree_idx = self.disc.degrees.index(degree)
            start = sum(self.disc.mode_counts[:degree_idx])
            stop = sum(self.disc.mode_counts[:(degree_idx + 1)])
            position = slice(start, stop)
            return position

        raise TypeError(f"Invalid type for key: {key}.")

    @staticmethod
    def _gen_index_list(degrees, depths, reduce=False):

        if reduce:
            index_list = [
                (ell, n + 1)
                for ell, nmax in zip(degrees, depths)
                for n in range(nmax)
            ]
        else:
            index_list = [
                (ell, m, n + 1)
                for ell, nmax in zip(degrees, depths)
                for m in range(- ell, ell+1)
                for n in range(nmax)
            ]

        return index_list

    @staticmethod
    def _gen_index_directory(index_list):

        # The directory is a mapping from an index tuple to an array position.
        index_directory = {
            tuple(tup): pos for pos, tup in enumerate(index_list)
        }

        return index_directory


class CartesianArray(DataArray):
    r"""Structured array for Cartesian decomposition of cosmological data.

    Array is initialised with three fields: the 'order' field of the
    Legendre multipole, the 'wavenumber' field of :math:`k`-bin centres
    and the 'power' field for power spectrum multipole measurements.

    Parameters
    ----------
    orders : list of tuple of int
        Orders of the power spectrum multipole.
    wavenumbers : float, array_like
        Wavenumbers of the multipole data.
    mode_counts : list of int or None, optional
        Mode counts in wavenumber bins (default is `None`).
    shot_noise : float or None, optional
        Shot noise level (default is `None`).

    Attributes
    ----------
    array : :class:`numpy.ndarray`
        Structured NumPy array.
    size : int
        Total number of elements in the array.  This should equal the
        product of the numbers of wavenumbers and multipoles.
    attrs : dict
        Initialisation parameters as attributes.

    """

    # Class-wide ``numpy.dtype`` for the structured array.
    _dtype = np.dtype({
        'names': ['order', 'wavenumber', 'power'],
        'formats': ['i4', 'f8', 'f8'],
    })

    def __init__(self, orders, wavenumbers, mode_counts=None, shot_noise=None):

        self.size = np.size(orders) * np.size(wavenumbers)

        orders = np.sort(orders).tolist()

        wavenumbers = np.array(wavenumbers)[np.argsort(wavenumbers)]

        if mode_counts is not None:
            mode_counts = np.array(mode_counts)[np.argsort(wavenumbers)]

        self.attrs = dict(
            orders=orders,
            wavenumbers=wavenumbers,
            mode_counts=mode_counts,
            shot_noise=shot_noise,
        )

        self.array, self._directory = self._initialise_array()

    def __setitem__(self, key, value):
        """Set the 'power' field value(s).

        Parameters
        ----------
        key : int, tuple(int, float) or slice
            'power' field access key.
        value : float
            'power' field data entry.

        See Also
        --------
        :meth:`__getitem__`

        """
        position = self._find_position(key)

        self.array['power'][position] = value

    def __getitem__(self, key):
        """Access the 'power' field.

        The access key can be an integer positional index, a slice, a tuple
        of (order, wavenumber) or a string, e.g. ``[-1]``, ``[:]``,
        ``[(0, 0.04)]`` or ``'power_0'``.

        Parameters
        ----------
        key : int, slice, tuple(int, float) or str
            'power' field access key.

        Returns
        -------
        float
            'power' field data entry.

        """
        position = self._find_position(key)

        return self.array['power'][position]

    def __setstate__(self, state):

        for attr, value in state.items():
            setattr(self, attr, value)

    def __getstate__(self):

        state = self.__dict__

        return state

    def vectorise(self, pivot):
        r"""Return a data vector from the 'power' field.

        Vectorisation is performed by *pivoting* in two orders of
        precedence---

            * 'order': ordered by multipole order :math:`\ell`;
            * 'wavenumber': ordered by wavenumber :math:`k`.

        Parameters
        ----------
        pivot : {'order', 'wavenumber'}
            Pivot order for vectorisation.

        Returns
        -------
        vectorised_data : :class:`numpy.ndarray`
            Vectorised power spectrum data.

        """
        if pivot == 'order':
            sort_order = ['order', 'wavenumber']
        elif pivot == 'wavenumber':
            sort_order = ['wavenumber', 'order']
        else:
            raise ValueError(f"Unknown `pivot` option: {pivot}.")

        return np.sort(self.array, order=sort_order)['power']

    def _initialise_array(self):

        # Initialise the structured array ordered by order.
        array = np.empty(self.size, dtype=self._dtype)

        array['order'] = np.repeat(
            self.attrs['orders'], len(self.attrs['wavenumbers']))
        array['wavenumber'] = np.tile(
            self.attrs['wavenumbers'], len(self.attrs['orders'])
        )

        # Generate the dictionary mapping (order, wavenumber) tuples to
        # array positions.
        directory = self._gen_directory(
            self.attrs['orders'], self.attrs['wavenumbers']
        )

        return array, directory

    def _find_position(self, key):

        # If accessed by integer, reinterprete negative values as reverse
        # indexing.
        if isinstance(key, int):
            position = key if key >= 0 else key % self.size
            if position > self.size:
                raise IndexError(
                    f"Index {position} out of bound for key: {key}."
                )
            return position

        # Access by slice.
        if isinstance(key, slice):
            return key

        # Access by sequence of (order, wavenumber).
        if isinstance(key, Sequence) and not isinstance(key, str):
            position = self._directory[tuple(key)]
            return position

        # Access by string of the form 'power_<ell>'.
        if isinstance(key, str):
            order = int(key.split('_')[-1])
            order_idx = self.attrs['orders'].index(order)
            length = len(self.attrs['wavenumbers'])
            position = slice(order_idx * length, (order_idx + 1) * length)
            return position

        raise TypeError(f"Invalid type for key: {key}.")

    @staticmethod
    def _gen_directory(orders, wavenumbers):

        directory = {
            tuple(key): pos
            for pos, key in enumerate(product(orders, wavenumbers))
        }

        return directory
