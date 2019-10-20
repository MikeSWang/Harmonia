"""
Array morphing (:mod:`~harmonia.algorithms.morph`)
===========================================================================

Manipulate cosmological field data arrays.

.. autosummary::

    SphericalArray

|

"""
import warnings

import numpy as np

from .bases import spherical_besselj_root
from harmonia.collections.utils import sort_dict_to_list


class SphericalArray:
    r"""Spherical arrays with morphable structures.

    The array is initialised in the natural structure together with an
    index array of the same structure.  A natural structure array is a
    length-:math:`\ell` sequence of
    :math:`(m_\ell \times n_\ell)`-rectangular arrays whose entries are
    indexed by a triplet :math:`(\ell, m_\ell, n_\ell)` corresponding to
    the spherical degree, order and depth respectively.

    Spherical degrees :math:`\ell` and orders :math:`m` are associated with
    the spherical Bessel and harmonic functions, and spherical depths
    :math:`n` are the number of allowed radial wave numbers for each degree
    of a discrete Fourier spectrum.

    Generally, the array can appear in the following structures:

        * 'natural', or equivalently :math:`(\ell, m, n)`;
        * 'transposed', or equivalently :math:`(\ell, n, m)`, where
          ordering by spherical depths takes precedence over that by
          spherical orders;
        * 'spectral', or equivalently :math:`k`, where a flattened array is
          sorted by the spectral wavenumbers in ascending order;
        * 'root', or equivalently :math:`(\ell, n)`, as above but subarrays
          of equivalent :math:`m` have been averaged/collapsed in the
          corresponding axis;
        * 'scale', or equivalently :math:`u`, similar to 'spectral' but
          subarrays of equivalent :math:`m` have been averaged/collapsed in
          the corresponding axis.

    Parameters
    ----------
    degrees : int, array_like
        Spherical degrees.
    depths : int, array_like
        Spherical depths for each degree.
    roots : *dict of* {*int*: :class:`numpy.ndarray`}, optional
        Roots corresponding to `degrees`, `depths` and wavenumbers of the
        discrete spherical spectrum.  If `None` (default), the roots are
        computed by calling
        :func:`~harmonia.algorithms.bases.spherical_besselj_root`.
    filling : float array_like or None, optional
        Data to be filled in the spherical array (default is `None`).

    Attributes
    ----------
    degrees : list of int
        Spherical degrees.
    depths : list of int
        Spherical depths associated with each spherical degree.
    roots : *list of float* :class:`numpy.ndarray`
        Roots of spherical Bessel functions or their derivatives
        corresponding to `degrees` and `depths`.
    index_array : list of (int, int, int), array_like
        Triplet indices stored in the natural structure.
    data_array : *list of* :class:`numpy.ndarray` *or None*
        Data stored in the natural structure.

    Raises
    ------
    ValueError
        If `filling` does not have the natural structure matching `degrees`
        and `depths`.

    See Also
    --------
    :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`

    """

    def __init__(self, degrees, depths, roots=None, filling=None):

        self.degrees, self.depths = degrees, depths

        if isinstance(roots, dict):
            roots = sort_dict_to_list(roots)
        if roots is None:
            roots = [
                spherical_besselj_root(ell, nmax, only=False)
                for ell, nmax in zip(degrees, depths)
            ]
        self.roots = roots

        index_array = []
        for ell, nmax in zip(degrees, depths):
            ell_block = []
            for m in range(-ell, ell+1):
                m_line = []
                for n_idx in range(nmax):
                    m_line.append((ell, m, n_idx + 1))
                ell_block.append(m_line)
            index_array.append(ell_block)
        self.index_array = index_array

        self.data_array = None
        if filling is not None:
            if len(filling) == len(degrees):
                for ell, nmax, fillblock in zip(degrees, depths, filling):
                    if np.shape(fillblock) != (2*ell+1, nmax):
                        raise ValueError(
                            f"Element of spherical degree {ell} in `filling` "
                            "is not a rectangular array whose shape "
                            "is consistent with `degrees` and `depths`. "
                        )
            else:
                raise ValueError(
                    "Length of `filling` is not consistent with `degrees`. "
                )
            self.data_array = [np.array(fillblock) for fillblock in filling]

    @classmethod
    def build(cls, filling=None, disc=None):
        """Build spherical array from a given discrete spherical spectrum.

        If `disc` is not provided, the natural structure is inferred from
        `filling`; if only `disc` is provided, a natural structue index
        array is returned with :attr:`data_array` being `None`.

        Parameters
        ----------
        filling : array_like or None, optional
            Data array in the natural structure for filling (default is
            `None`).
        disc : :class:`.DiscreteSpectrum` *or None, optional*
            Discretisation set-up (default is `None`).

        Raises
        ------
        ValueError
            `disc` and `filling` are both `None`.
        ValueError
            `filling` has the wrong shape or ordering.

        """
        if disc is not None:
            return cls(
                disc.degrees,
                disc.depths,
                roots=disc.roots,
                filling=filling
            )
        if filling is None:
            raise ValueError("`disc` and `filling` cannot both be None. ")

        if not all([len(fillblock) % 2 for fillblock in filling]):
            raise ValueError(
                "Subarrays of `filling` should have odd lengths. "
            )

        degrees = [len(fillblock) // 2 for fillblock in filling]

        if sorted(degrees) != degrees:
            raise ValueError(
                "`filling` subarrays should be in ascending order of lengths. "
            )

        depths = [np.size(fillblock, axis=-1) for fillblock in filling]

        return cls(degrees, depths, filling=filling)

    def unfold(self, axis_order, collapse=None, return_only=None):
        r"""Flatten data and index arrays in the specified axis order,
        which corresponds to an array structure.

        If the arrays are collapsed amongst equivalent spherical orders,
        each block element in :attr:`data_array` is first averaged over the
        rows and the triplet index tuples in :attr:`index_array` are
        stripped of their middle index before flattening.

        Parameters
        ----------
        axis_order : {'natural', 'transposed', 'spectral', 'root', 'scale'}
            Axis order for array flattening.
        collapse : {'mean', 'rms', None}, optional
            If ``'mean'`` or ``'rms'`` (default is `None`), the arrays are
            collapsed over spherical orders by averaging or averaging in
            square modulus before flattening.  If `None` but `axis_order`
            is ``'root'`` or ``'scale'``, this is overridden to `mean`.
        return_only : {'data', 'index', None}, optional
            Only return the 'data' or 'index' array (default is `None`).

        Returns
        -------
        data_flat : float array_like or None
            Flattend 1-d data array.  Returned only if `return_only` is
            `None` or ``'data'``.
        index_flat : list of tuple
            Flattend 1-d index array.  Returned only if `return_only` is
            `None` or ``'index'``.

        """
        dat_arr, idx_arr = self.data_array, self.index_array

        empty_flag = (dat_arr is None)
        if empty_flag:
            data_flat = None
            warnings.warn(
                "`data_array` is None and thus not flattened. ",
                RuntimeWarning
            )

        axis_order = self._alias(axis_order)
        if axis_order == 'ln':
            axis_order = 'lmn'
            if collapse is None:
                collapse = 'mean'
        if axis_order == 'u':
            axis_order = 'k'
            if collapse is None:
                collapse = 'mean'

        transpose = (axis_order == 'lnm')
        if collapse:
            square = (collapse == 'rms')
            if not empty_flag:
                dat_arr = \
                    self._collapse_subarray(dat_arr, 'data', square=square)
            idx_arr = self._collapse_subarray(idx_arr, 'index')

        index_flat =  \
            self._flatten(idx_arr, 'index', subarray_transpose=transpose)
        if not empty_flag:
            data_flat = np.array(
                self._flatten(dat_arr, 'data', subarray_transpose=transpose)
            )

        if axis_order == 'k':
            roots = self.roots
            if not collapse:
                roots = \
                    self._repeat_subarray(roots, 'data', degrees=self.degrees)

            flat_order = np.argsort(self._flatten(roots, 'data'))
            index_flat = [index_flat[order_idx] for order_idx in flat_order]
            if not empty_flag:
                data_flat = data_flat[flat_order]

        if return_only == 'data':
            return data_flat
        if return_only == 'index':
            return index_flat
        return data_flat, index_flat

    def refold(self, flat_array, structure, subarray_type):
        """Return a compatible external flat array to the natural
        structure.

        Parameters
        ----------
        flat_array : array_like
            External flat array to be refolded.
        structure : {'natural', 'transposed', 'spectral', 'root', 'scale'}
            Structure in which `flat_array` has been flattened.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'``
            for index arrays.

        Returns
        -------
        array : list of tuple or float, array_like
            External array in natural structure.

        """
        if subarray_type == 'index':
            return self.index_array

        ordered_index = self.unfold(structure, return_only='index')

        structure = self._alias(structure)
        if structure in ['lmn', 'lnm', 'k']:
            array = []
            for ell, nmax in zip(self.degrees, self.depths):
                ell_block = []
                for m in range(-ell, ell+1):
                    m_line = []
                    for n in range(nmax):
                        m_line.append(None)
                    ell_block.append(m_line)
                array.append(ell_block)
            for index, entry in zip(ordered_index, flat_array):
                ell_idx, m_idx, n_idx = \
                    index[0], index[1] + index[0], index[-1] - 1
                array[ell_idx][m_idx][n_idx] = entry
        elif structure in ['ln', 'u']:
            array = [
                [None for n in range(nmax)]
                for ell, nmax in zip(self.degrees, self.depths)
            ]
            for index, entry in zip(ordered_index, flat_array):
                ell_idx, n_idx = index[0], index[-1] - 1
                array[ell_idx][n_idx] = entry
            array = \
                self._repeat_subarray(array, 'data', degrees=self.degrees)

        return array

    def morph(self, flat_array, in_struct, out_struct, subarray_type):
        """Morph a compatible external flat array flattened in one
        structure to another structure by returning it to the natural
        structure first.

        Parameters
        ----------
        flat_array : array_like
            External flat array to be morphed.
        in_struct : {'natural', 'transposed', 'spectral', 'root', 'scale'}
            Input structure.
        out_struct : {'natural', 'transposed', 'spectral', 'root', 'scale'}
            Output structure.  If this is ``'spectral'`` or ``'scale'``,
            the morphed array is flattened; if this is ``'root'``, the
            morphed array is collapsed.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` arrays or ``'index'`` arrays.

        Returns
        -------
        morphed_array : array_like
            Morphed array.

        """
        in_struct = self._alias(in_struct)
        out_struct = self._alias(out_struct)

        nat_arr = self.refold(flat_array, in_struct, subarray_type)

        if out_struct == 'lmn':
            morphed_array = nat_arr
        if out_struct == 'lnm':
            morphed_array = self._transpose_subarray(
                flat_array,
                subarray_type=subarray_type
            )
        if out_struct == 'ln':
            morphed_array = self._collapse_subarray(
                flat_array,
                subarray_type=subarray_type
            )
        if out_struct == 'k':
            morphed_array = self._flatten(nat_arr, subarray_type)
            flat_order = np.argsort(
                self._flatten(
                    self._repeat_subarray(
                        self.roots,
                        'data',
                        degrees=self.degrees
                    ),
                    'data'
                )
            )
            if subarray_type == 'data':
                morphed_array = morphed_array[flat_order]
            elif subarray_type == 'index':
                morphed_array = [
                    morphed_array[order_idx] for order_idx in flat_order
                ]
        if out_struct == 'u':
            morphed_array = self._flatten(
                self._collapse_subarray(nat_arr, subarray_type),
                subarray_type
            )
            flat_order = np.argsort(self._flatten(self.roots, 'data'))
            if subarray_type == 'data':
                morphed_array = morphed_array[flat_order]
            elif subarray_type == 'index':
                morphed_array = [
                    morphed_array[order_idx] for order_idx in flat_order
                ]

        return morphed_array

    def _flatten(self, array, subarray_type, subarray_transpose=False):
        """Flatten a natural structure array.

        Parameters
        ----------
        array : list of float or tuple, array_like
            Natural structure array.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` arrays or ``'index'`` arrays.
        subarray_transpose : bool, optional
            If `True` (default is `False`), each subarray is flattened
            along the columns rather than rows by a transposition.

        Returns
        -------
        float or tuple, array_like
            Flat 1-d array.

        Raises
        ------
        ValueError
            If `subarray_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarray_transpose:
            array = self._transpose_subarray(array, subarray_type)

        if subarray_type == 'data':
            return np.concatenate(
                [np.array(array_block).flatten() for array_block in array]
            )
        if subarray_type == 'index':
            return [
                entry for block in array for line in block for entry in line
            ]
        raise ValueError(f"Invalid `subarray_type` value: {subarray_type}. ")

    @staticmethod
    def _alias(structure_name):
        """Replace aliases of stucture names by the default structure name.

        Parameters
        ----------
        structure_name : str
            Array structure name.

        Returns
        -------
        str
            Equivalent array structure name.

        Raises
        ------
        ValueError
            If `structure_name` is not a valid structure name.

        """
        if structure_name == 'natural':
            return 'lmn'
        if structure_name == 'transposed':
            return 'lnm'
        if structure_name == 'spectral':
            return 'k'
        if structure_name == 'root':
            return 'ln'
        if structure_name == 'scale':
            return 'u'
        if structure_name in ['lmn', 'lnm', 'ln', 'k', 'u']:
            return structure_name
        else:
            raise ValueError(
                f"Invalid `structure_name` value: {structure_name}. "
            )

    @staticmethod
    def _transpose_subarray(array, subarray_type):
        """Transpose array elements in a list.

        Parameters
        ----------
        array : list of float or tuple, array_like
            Array of uncollapsed subarrays.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` arrays or ``'index'`` arrays.

        Returns
        -------
        list of float or tuple, array_like
            Array of transposed subarrays.

        Raises
        ------
        ValueError
            If `subarray_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarray_type == 'data':
            return [np.array(ell_block).T for ell_block in array]
        if subarray_type == 'index':
            return [list(map(list, zip(*ell_block))) for ell_block in array]
        raise ValueError(f"Invalid `subarray_type` value: {subarray_type}. ")

    @staticmethod
    def _collapse_subarray(array, subarray_type, square=False):
        """Collapse a natural structure array over equivalent spherical
        orders while preserving array dimensions.

        Parameters
        ----------
        array : list of float or tuple, array_like
            Natural structure array.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` arrays or ``'index'`` arrays.
        square : bool, optional
            If `True` (default is `False`), the collapsed data array values
            are root mean square magnitudes.

        Returns
        -------
        list of float or tuple, array_like
            Array of collapsed subarrays.

        Raises
        ------
        ValueError
            If `subarray_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarray_type == 'data' and square:
            return [
                np.sqrt(
                    np.mean(
                        np.square(np.abs(ell_block)),
                        axis=0,
                        keepdims=True
                    )
                )
                for ell_block in array
            ]
        if subarray_type == 'data' and not square:
            return [
                np.mean(ell_block, axis=0, keepdims=True)
                for ell_block in array
            ]
        if subarray_type == 'index':
            return [
                [
                    list(
                        map(lambda index: (index[0], index[-1]), ell_block[0])
                    )
                ]
                for ell_block in array
            ]
        raise ValueError(f"Invalid `subarray_type` value: {subarray_type}. ")

    @staticmethod
    def _repeat_subarray(array, subarray_type, degrees=None):
        """Repeat an array collapsed over equivalent spherical orders to
        recover the natural structure array.

        Parameters
        ----------
        array : list of float or tuple, array_like
            Array of collapsed subarrays.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` arrays or ``'index'`` arrays.
        degrees : list of int array_like or None
            Spherical degrees correponding to the subarrays to be repeated
            (default is `None`).  If `None`, this is inferred from the
            `array` length assuming the lowest degree is 0.

        Returns
        -------
        list of float or tuple, array_like
            Natural structure array.

        Raises
        ------
        ValueError
            If `subarray_type` is neither ``'data'`` nor ``'index'``.
        ValueError
            If length of `array` disagrees with input `degrees`.

        """
        if degrees is None:
            degrees = np.arange(len(array))
        elif len(degrees) != len(array):
            raise ValueError(
                "`degrees` and `array` lengths differ: {} and {}. "
                .format(len(degrees), len(array))
            )

        if subarray_type == 'data':
            return [
                np.repeat(np.atleast_2d(line), 2*ell+1, axis=0)
                for line, ell in zip(array, degrees)
            ]
        if subarray_type == 'index':
            new_array = []
            for lineblock, ell in zip(array, degrees):
                block = [
                    list(
                        map(
                            lambda index: (index[0], m, index[-1]),
                            lineblock[0]
                        )
                    )
                    for m in range(-ell, ell+1)
                ]
                new_array.append(block)
            return new_array
        raise ValueError(f"Invalid `subarray_type` value: {subarray_type}. ")
