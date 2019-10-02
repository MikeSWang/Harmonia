"""
Array morphing (:mod:`~harmonia.algorithms.morph`)
===========================================================================

Manipulate cosmological data array structures.

.. autosummary::

    SphericalArray

|

"""
import warnings

import numpy as np

from .bases import spherical_besselj_root
from harmonia.collections.utils import sort_dict_to_list


class SphericalArray:
    r"""Morphable spherical arrays with specified spherical degrees, orders
    and depths.

    The array is initialised in the natural structure together with an
    index array of the same structure.  A natural structure array is a
    length-:math:`\ell` list of :math:`(m_\ell \times n_\ell)`-rectangular
    arrays whose entries are indexed by a triplet :math:`(\ell, m_\ell,
    n_\ell)`, which are respectively the spherical degree, order and depth.

    Spherical degrees :math:`\ell` and orders :math:`m` are associated with
    the spherical Bessel and harmonic functions, and spherical depths
    :math:`n` are the number of allowed radial wave numbers for each degree
    of a discrete Fourier spectrum, implemented by
    :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`.

    The supported array structures include the following:

        * 'natural' or equivalently :math:`(\ell, m, n)`;
        * :math:`(\ell, n, m)`, where ordering by spherical depths takes
          precedence over that by spherical orders;
        * :math:`(\ell, n)`, as above but values for equivalent :math:`m`
          have been averaged/collapsed in the corresponding axis;
        * :math:`k`, where a flattened array is sorted in ascending order
          by wave numbers;
        * 'scale', as above but values for equivalent :math:`m` have first
          been averaged/collapsed in the corresponding axis.

    Parameters
    ----------
    degrees : int, array_like
        Spherical degrees.
    depths : int, array_like
        Spherical depths for each degree.
    roots : :obj:`dict` of {int: :class:`numpy.ndarray`}, optional
        Roots of spherical Bessel functions or their derivatives
        corresponding to `degrees` and `depths`.  If this is `None`
        (default), the roots are computed by calling
        :func:`~harmonia.algorithms.bases.spherical_besselj_root`.
    filling : float array_like or None, optional
        Data array from which the spherical array is built (default is
        `None`).

    Attributes
    ----------
    degrees : int, array_like
        Spherical degrees.
    depths : int, array_like
        Spherical depths associated with each spherical degree.
    roots : list of float array_like
        Roots of spherical Bessel functions or their derivatives
        corresponding to `degrees` and `depths`.
    index_array : :obj:`list` of (int, int, int) array_like
        Triplet indices stored in the natural structure.
    data_array : :obj:`list` of float array_like or None
        Data stored in the natural structure.

    Raises
    ------
    ValueError
        If `filling` does not have the natural structure matching `degrees`
        and `depths`.

    """

    def __init__(self, degrees, depths, roots=None, filling=None):

        if roots is None:
            roots = [
                spherical_besselj_root(ell, nmax, only=False)
                for ell, nmax in zip(degrees, depths)
            ]
        if isinstance(roots, dict):
            roots = sort_dict_to_list(roots)
        self.degrees, self.depths, self.roots = degrees, depths, roots

        self.index_array = []
        for ell, nmax in zip(degrees, depths):
            ell_block = []
            for m in range(-ell, ell+1):
                m_line = []
                for n_idx in range(nmax):
                    m_line.append((ell, m, n_idx + 1))
                ell_block.append(m_line)
            self.index_array.append(ell_block)

        if filling is not None:
            if len(filling) == len(degrees):
                for ell, nmax, fillblock in zip(degrees, depths, filling):
                    if np.shape(fillblock) != (2*ell + 1, nmax):
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
        else:
            self.data_array = None

    def __repr__(self):
        return f"SphericalArray(degnum={len(self.degrees)},id={id(self)})"

    @classmethod
    def build(cls, filling=None, disc=None):
        """Alternative build from a natural structure array and/or a
        discrete spectrum.

        If `disc` is not provided, the natural structure is inferred from
        `filling`; if only `disc` is provided, a natural structue index
        array is returned with :attr:`data_array` being `None`.

        Parameters
        ----------
        filling : array_like or None, optional
            Natural structure data array for filling (default is `None`).
        disc : :class:`.DiscreteSpectrum` or None, optional
            Discretisation set-up (default is `None`).

        Raises
        ------
        ValueError
            `disc` and `filling` are both `None`.
        ValueError
            Spherical degrees inferred from `filling` do not respect the
            odd parity and ascending ordering.

        """
        if disc is not None:
            return cls(
                disc.degrees,
                disc.depths,
                roots=disc.roots,
                filling=filling,
            )
        if filling is None:
            raise ValueError("`disc` and `filling` cannot both be None. ")

        if not all([len(fillblock) % 2 for fillblock in filling]):
            raise ValueError(
                "At least one block element of `filling` "
                "is not of odd lengths. "
            )

        degrees = [len(fillblock) // 2 for fillblock in filling]

        if sorted(degrees) != degrees:
            raise ValueError(
                "Block elements of `filling` are not ordered by ascending "
                "block lengths. "
            )

        depths = [np.size(fillblock, axis=-1) for fillblock in filling]

        return cls(degrees, depths, filling=filling)

    def unfold(self, axis_order, collapse=False, return_only=None):
        r"""Flatten data and index arrays in the specified order.

        If the arrays are collapsed amongst equivalent spherical orders,
        each block element in :attr:`data_array` is first averaged over the
        rows and the triplet index tuple elements in :attr:`index_array`
        are stripped of their middle order-index before flattening.

        Parameters
        ----------
        axis_order : {'natural', 'scale', 'lmn', 'lnm', 'ln', 'k'}
            Axis order for array flattening.  If this is set to ``'ln'`` or
            ``'scale'``, `collapse` is overriden to `True`.
        collapse : bool, optional
            If `True` (default is `False`), the arrays are collapsed over
            spherical orders before flattening.  This is overriden to
            `True` if `axis_order` is ``'ln'`` or ``'k'``.
        return_only : {'data', 'index'} or None, optional
            Only return the 'data' or 'index' array (default is `None`).

        Returns
        -------
        data_flat : float array_like or None
            Flattend 1-d data array.
        index_flat : :obj:`list` of :obj:`tuple`
            Flattend 1-d index array.

        """
        data_arr, index_arr = self.data_array, self.index_array

        empty_flag = (data_arr is None)
        if empty_flag:
            data_flat = None
            warnings.warn(
                "`data_array` is None and thus not flattened. ",
                RuntimeWarning,
            )

        axis_order = self._alias(axis_order)
        if axis_order == 'ln':
            axis_order, collapse = 'lmn', True
        if axis_order == 'scale':
            axis_order, collapse = 'k', True

        if collapse:
            if not empty_flag:
                data_arr = self.collapse_subarray(data_arr, 'data')
            index_arr = self.collapse_subarray(index_arr, 'index')

        transpose = (axis_order == 'lnm')

        if not empty_flag:
            data_flat = np.array(
                self._flatten(data_arr, 'data', subarray_transpose=transpose),
            )
        index_flat = self._flatten(
            index_arr,
            'index',
            subarray_transpose=transpose,
        )

        if axis_order == 'k':
            roots = self.roots
            if not collapse:
                roots = self.repeat_subarray(
                    roots,
                    'data',
                    degrees=self.degrees,
                )

            order = np.argsort(self._flatten(roots, 'data'))
            if not empty_flag:
                data_flat = data_flat[order]
            index_flat = [index_flat[ord_idx] for ord_idx in order]

        if return_only == 'data':
            return data_flat
        if return_only == 'index':
            return index_flat
        return data_flat, index_flat

    def refold(self, flat_array, in_structure, subarray_type):
        """Return a flattened array to its natural structure which is taken
        to be the same as :attr:`data_array`.

        Parameters
        ----------
        flat_array : array_like
            Flat array to be refolded.
        in_structure : {'natural', 'lmn', 'lnm', 'ln', 'k', 'scale'}
            Input structure.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'``
            for index arrays.

        Returns
        -------
        :obj:`list` of :obj:`tuple` or float array_like
            Refolded natural array.

        Raises
        ------
        ValueError
            If `in_structure` is not a valid structure name.

        """
        if subarray_type == 'index':
            return self.index_array

        ordered_index = self.unfold(in_structure, return_only='index')

        if in_structure in ['natural', 'lmn', 'lnm', 'k']:
            return_arr = []
            for ell, nmax in zip(self.degrees, self.depths):
                ell_block = []
                for m in range(-ell, ell+1):
                    m_line = []
                    for n in range(nmax):
                        m_line.append(None)
                    ell_block.append(m_line)
                return_arr.append(ell_block)
            for index, entry in zip(ordered_index, flat_array):
                ell_idx, m_idx, n_idx = \
                    index[0], index[1] + index[0], index[-1] - 1
                return_arr[ell_idx][m_idx][n_idx] = entry
        elif in_structure in ['ln', 'scale']:
            return_arr = [
                [None for n in range(nmax)]
                for ell, nmax in zip(self.degrees, self.depths)
            ]
            for index, entry in zip(ordered_index, flat_array):
                ell_idx, n_idx = index[0], index[-1] - 1
                return_arr[ell_idx][n_idx] = entry
            return_arr = self.repeat_subarray(
                return_arr,
                'data',
                degrees=self.degrees
            )
        else:
            raise ValueError("`in_structure` is invalid. ")

        return return_arr

    def morph(self, array, in_structure, out_structure, subarray_type):
        """Morph an array from one structure to another structure.

        All morphings are performed by returning the array shape to the
        natural structure first.  The allowed morphings are (any
        composition of) the following (modulo equivalent structure names):

            * 'lmn' to/from 'lnm';
            * 'lmn' to/from 'k';
            * 'lmn' to 'scale' and the reverse with repetitions;
            * 'lmn' to 'ln' and the reverse with repetitions.

        Parameters
        ----------
        array : array_like
            Array to be morphed.
        in_structure : {'natural', 'lmn', 'lnm', 'ln', 'k', 'scale'}
            Input structure.
        out_structure : {'natural', 'lmn', 'lnm', 'ln', 'k', 'scale'}
            Output structure.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'``
            for index arrays.

        Returns
        -------
        morph_array : array_like
            Morphed array.

        """
        in_structure = self._alias(in_structure)
        out_structure = self._alias(out_structure)

        natarr = self.refold(array, in_structure, subarray_type)
        if out_structure == 'lmn':
            morph_array = natarr
        if out_structure == 'lnm':
            morph_array = self.transpose_subarray(
                array,
                subarray_type=subarray_type,
            )
        if out_structure == 'ln':
            morph_array = self.collapse_subarray(
                array,
                subarray_type=subarray_type,
            )
        if out_structure == 'k':
            morph_array = self._flatten(natarr, subarray_type)
            order = np.argsort(
                self._flatten(
                    self.repeat_subarray(
                        self.roots,
                        'data',
                        degrees=self.degrees,
                    ),
                    'data',
                )
            )
            if subarray_type == 'data':
                morph_array = morph_array[order]
            elif subarray_type == 'index':
                morph_array = [morph_array[ord_idx] for ord_idx in order]
        if out_structure == 'scale':
            morph_array = self._flatten(
                self.collapse_subarray(natarr, subarray_type),
                subarray_type,
            )
            order = np.argsort(self._flatten(self.roots, 'data'))
            if subarray_type == 'data':
                morph_array = morph_array[order]
            elif subarray_type == 'index':
                morph_array = [morph_array[ord_idx] for ord_idx in order]

        return morph_array

    @staticmethod
    def transpose_subarray(array, subarray_type):
        """Transpose array elements in a list.

        Parameters
        ----------
        array : list of float or tuple, array_like
            List of subarrays.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'``
            for index arrays.

        Returns
        -------
        :obj:`list` of float or :obj:`tuple`, array_like
            List of transposed subarrays.

        Raises
        ------
        ValueError
            If `subarray_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarray_type == 'data':
            return [np.array(ell_block).T for ell_block in array]
        if subarray_type == 'index':
            return [list(map(list, zip(*ell_block))) for ell_block in array]
        raise ValueError(f"Invalid `subarray_type`: {subarray_type}. ")

    @staticmethod
    def collapse_subarray(array, subarray_type):
        """Collapse a natural structure array over equivalent spherical
        orders while preserving array dimensions.

        Parameters
        ----------
        array : list of float or tuple, array_like
            Natural structure array.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'``
            for index arrays.

        Returns
        -------
        :obj:`list` of float or :obj:`tuple`, array_like
            List of collapsed subarrays along the spherical order axis.

        Raises
        ------
        ValueError
            If `subarray_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarray_type == 'data':
            return [
                np.mean(ell_block, axis=0, keepdims=True)
                for ell_block in array
            ]
        if subarray_type == 'index':
            return [
                [
                    list(map(lambda tup: (tup[0], tup[-1]), ell_block[0]))
                ]
                for ell_block in array
            ]
        raise ValueError(f"Invalid `subarray_type`: {subarray_type}. ")

    @staticmethod
    def repeat_subarray(array, subarray_type, degrees=None):
        """Repeat an array collapsed over equivalent spherical orders to
        recover the natural structure array.

        Parameters
        ----------
        array : list of float or tuple, array_like
            Array collapsed over spherical orders.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'``
            for index arrays.
        degrees : list of int array_like or None
            Spherical degrees for which equivalent spherical order arrays
            are repeated (default is `None`).  If it is `None`, the degrees
            are inferred from the length of `array` assuming the first
            degree is 0.

        Returns
        -------
        :obj:`list` of float or :obj:`tuple`, array_like
            Uncollapsed natural structure array.

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
                "Lengths of `degrees` and `array` disagree: "
                f"{len(degrees)} and {len(array)}. "
            )

        if subarray_type == 'data':
            return [
                np.repeat(np.atleast_2d(line), 2*ell+1, axis=0)
                for line, ell in zip(array, degrees)
            ]
        if subarray_type == 'index':
            return [
                [
                    list(map(lambda tup: (tup[0], m, tup[-1]), lineblock[0]))
                    for m in range(-ell, ell+1)
                ]
                for lineblock, ell in zip(array, degrees)
            ]
        raise ValueError(f"Invalid `subarray_type`: {subarray_type}. ")

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

        """
        if structure_name == 'natural':
            return 'lmn'
        return structure_name

    def _flatten(self, array, subarray_type, subarray_transpose=False):
        """Flatten a natural structure array.

        Parameters
        ----------
        array : list of float or tuple, array_like
            Natural structure array.
        subarray_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'``
            for index arrays.
        subarray_transpose : bool, optional
            If `True` (default is `False`), each subarray is flattened
            along the columns rather than rows by a transposition.

        Returns
        -------
        float or :obj:`tuple`, array_like
            Flat 1-d array.

        Raises
        ------
        ValueError
            If `subarray_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarray_transpose:
            array = self.transpose_subarray(array, subarray_type)

        if subarray_type == 'data':
            return np.concatenate(
                [np.array(array_block).flatten() for array_block in array]
            )
        if subarray_type == 'index':
            return [
                entry for block in array for line in block for entry in line
            ]
        raise ValueError(f"Invalid `subarray_type`: {subarray_type}. ")
