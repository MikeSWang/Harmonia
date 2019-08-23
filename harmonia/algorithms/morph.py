"""
Array morphing (:mod:`~harmonia.algorithms.morph`)
===============================================================================

Manipulate cosmological data array structure.

.. autosummary::

    SphericalArray

|

"""
import warnings

import numpy as np

from .bases import sph_besselj_root


class SphericalArray:
    r"""Morphable spherical arrays with specified spherical degrees, orders and
    depths.

    The array is initialised in the natural structure together with an index
    array of the same structure.  A natural structure array is a
    length-:math:`\ell` list of :math:`m_\ell \times n_\ell` rectangular arrays
    whose entries are indexed by a triplet :math:`\{(\ell, m_\ell, n_\ell)\}`,
    which are respectively the spherical degree, order and depth.

    Spherical degrees :math:`\ell` and orders :math:`m` are associated with the
    spherical Bessel and harmonic functions, and spherical depths :math:`n` are
    the number of allowed radial modes for each degree of a discrete Fourier
    spectrum, implemented in this program by
    :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`.

    The supported array structures include the following:
        * 'natural' or equivalently :math:`(\ell, m, n)`;
        * :math:`(\ell, n, m)`, where ordering by spherical depths takes
          precedence over that by spherical orders;
        * :math:`(\ell, n)`, as above but values for equivalent :math:`m` have
          been averaged/collapsed in the corresponding axis;
        * :math:`k`, where a flattened array is sorted in ascending order by
          wave numbers;
        * 'scale', as above but values for equivalent :math:`m` have first been
          averaged/collapsed in the corresponding axis.

    Attributes
    ----------
    degrees : int, array_like
        Spherical degrees.
    depths : int, array_like
        Spherical depths associated with each spherical degree.
    roots : float, array_like
        Roots of spherical Bessel functions or their derivatives corresponding
        to :attr:`degrees` and :attr:`depths`.
    init_indices : list of (int, int, int), array_like
        Triplet indices stored in the natural structure.
    init_array : list of float, array_like or None
        Data stored in the natural structure.

    """

    def __init__(self, degrees, depths, roots=None, filling=None):
        r"""
        Parameters
        ----------
        degrees : int, array_like
            Spherical degrees.
        depths : int, array_like
            Spherical depths for each degree.
        roots : float, array_like, optional
            Roots of spherical Bessel functions or their derivatives
            corresponding to `degrees` and `depths`.  If this is `None`
            (default), the roots are computed by calling
            :func:`~harmonia.toolbox.basis.sph_besselj_root`.
        filling : float, array_like or None, optional
            Data array from which the spherical array is built (default is
            `None`).

        Raises
        ------
        ValueError
            If `filling` does not have the natural structure matching `degrees`
            and `depths`.

        """
        if roots is None:
            roots = [
                sph_besselj_root(ell, nmax, only=False)
                for ell, nmax in zip(degrees, depths)
                ]
        self.degrees, self.depths, self.roots = degrees, depths, roots

        self._waveindices = [
            [[(ell, nidx+1) for nidx in range(nmax)]]
            for ell, nmax in zip(degrees, depths)
            ]

        self.init_indices = [
            [[(ell, m, nidx+1)
              for nidx in range(nmax)]
             for midx, m in enumerate(range(-ell, ell+1))]
            for ell, nmax in zip(degrees, depths)
            ]

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
            self.init_array = [np.array(fillblock) for fillblock in filling]
        else:
            self.init_array = None

    def __repr__(self):
        return (f"SphericalArray("
                f"ndegrees={len(self.degrees)},id={id(self)})")

    @classmethod
    def build(cls, filling=None, disc=None):
        """Alternative build from a natural structure array and/or a discrete
        spectrum.

        If `disc` is not provided, the natural structure is inferred from
        `filling`; if only `disc` is provided, a natural structue index array
        is returned with NoneType :attr:`init_array`.

        Parameters
        ----------
        filling : array_like or None, optional
            Natural structure data array for filling (default is `None`).
        disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum` or None, optional
            Discretisation set-up (default is `None`).

        Raises
        ------
        ValueError
            `disc` and `filling` are both `None`.
        ValueError
            Spherical degrees inferred from `filling` do not respect the odd
            parity and ascending ordering.

        """
        if disc is not None:
            return cls(
                disc.degrees, disc.depths, roots=disc.roots, filling=filling
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
                "Block elements of `filling` are not ordered in ascension "
                "by block lenths. "
                )
        depths = [np.size(fillblock, axis=-1) for fillblock in filling]

        return cls(degrees, depths, filling=filling)

    def unfold(self, axis_order, collapse=False):
        r"""Flatten data and index arrays in the specified order.

        If the arrays are collapsed amongst equivalent spherical orders, each
        block element in :attr:`init_array` is first averaged over the rows and
        the triplet index tuple elements in :attr:`init_indices` are stripped
        of their middle :math:`m`-index before flattening.

        Parameters
        ----------
        axis_order : {'natural', 'scale', 'lmn', 'lnm', 'ln', k'}
            Axis order for array flattening.  If this is set to ``'ln'`` or
            ``'scale'``, `ord_collapse` is overriden to `True`.
        collapse : bool, optional
            If `True` (default is `False`), the arrays are collapsed over
            spherical orders before flattening.  This is overriden to `True`
            if `axis_order` is ``'ln'`` or ``'k'``.

        Returns
        -------
        data_flatarr : float, array_like or None
            Flattend 1-d data array.
        indx_flatarr : list of tuple
            Flattend 1-d index array.

        """
        data_arr, indx_arr = self.init_array, self.init_indices

        empty_flag = (data_arr is None)
        if empty_flag:
            data_flatarr = None
            warnings.warn("`init_array` is None and thus not flattened. ")

        # Spherical order collapse (`collapse` overriden if appropriate).
        axis_order = self._alias(axis_order)
        if axis_order == 'ln':
            axis_order, collapse = 'lmn', True
        if axis_order == 'scale':
            axis_order, collapse = 'k', True

        if collapse:
            if not empty_flag:
                data_arr = self._subarr_collapse(data_arr, 'data')
            indx_arr = self._subarr_collapse(indx_arr, 'index')

        # Vectorisation.
        transpose = (axis_order == 'lnm')  # additional transpose step

        if not empty_flag:
            data_flatarr = np.array(
                self._flatten(data_arr, 'data', subarr_trans=transpose)
                )
        indx_flatarr = self._flatten(indx_arr, 'index', subarr_trans=transpose)

        if axis_order == 'k':  # additional ordering step
            roots = self.roots
            if not collapse:
                roots = self._subarr_repeat(
                    roots, 'data', degrees=self.degrees
                    )
            order = np.argsort(self._flatten(roots, 'data'))
            if not empty_flag:
                data_flatarr = data_flatarr[order]
            indx_flatarr = [indx_flatarr[iord] for iord in order]

        return data_flatarr, indx_flatarr

    def refold(self, arr, in_struct, subarr_type):
        """Return an array to its natural structure.

        Parameters
        ----------
        arr : array_like
            Array to be refolded.
        in_struct : {'natural', 'lmn', 'lnm', 'ln', 'k', 'scale'}
            Input structure.
        subarr_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'`` for
            index arrays.

        Returns
        -------
        list of tuple or float, array_like
            Refolded natural array.

        """
        if in_struct in ['natural', 'lmn']:
            return arr
        if in_struct == 'lnm':
            return self._subarr_transpose(arr, subarr_type)
        if in_struct == 'ln':
            return self._subarr_repeat(
                arr, subarr_type, degrees=self.degrees
                )

        # Deal with 'scale' or 'k' structures.
        if subarr_type == 'index':
            return self.init_indices

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, ordered_indx = self.unfold(in_struct)
        if in_struct == 'k':
            retarr = [
                [[None for n in range(nmax)]
                 for m in range(-ell, ell+1)]
                for ell, nmax in zip(self.degrees, self.depths)
                ]
            for indx, entry in zip(ordered_indx, arr):
                ellidx, midx, nidx = indx[0], indx[1] + indx[0], indx[-1] - 1
                retarr[ellidx][midx][nidx] = entry
        elif in_struct == 'scale':
            retarr = [
                [None for n in range(nmax)]
                for ell, nmax in zip(self.degrees, self.depths)
                ]
            for indx, entry in zip(ordered_indx, arr):
                ellidx, nidx = indx[0], indx[-1] - 1
                retarr[ellidx][nidx] = entry
            retarr = self._subarr_repeat(retarr, 'data', degrees=self.degrees)

        return retarr

    def morph(self, arr, in_struct, out_struct, subarr_type):
        """Morph an array from one structure to another structure.

        All morphings are performed by returning the array shape to the natural
        structure first.  The allowed morphings are (any composition of) the
        following (modulo equivalent structure names):

            * 'lmn' to/from 'lnm';
            * 'lmn' to/from 'k';
            * 'lmn' to 'scale' and the reverse with repetitions;
            * 'lmn' to 'ln' and the reverse with repetitions.

        Parameters
        ----------
        arr : array_like
            Array to be morphed.
        in_struct : {'natural', 'lmn', 'lnm', 'ln', 'k', 'scale'}
            Input structure.
        out_struct : {'natural', 'lmn', 'lnm', 'ln', 'k', 'scale'}
            Output structure.
        subarr_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'`` for
            index arrays.

        Returns
        -------
        rearr : array_like
            Reshaped array.

        """
        in_struct = self._alias(in_struct)
        out_struct = self._alias(out_struct)

        natarr = self.refold(arr, in_struct, subarr_type)
        if out_struct == 'lmn':
            rearr = natarr
        if out_struct == 'lnm':
            rearr = self._subarr_transpose(arr, subarr_type=subarr_type)
        if out_struct == 'ln':
            rearr = self._subarr_collapse(arr, subarr_type=subarr_type)
        if out_struct == 'k':
            rearr = self._flatten(natarr, subarr_type)
            order = np.argsort(self._flatten(
                self._subarr_repeat(self.roots, 'data', degrees=self.degrees),
                'data'
                ))
            if subarr_type == 'data':
                rearr = rearr[order]
            elif subarr_type == 'index':
                rearr = [rearr[iord] for iord in order]
        if out_struct == 'scale':
            rearr = self._flatten(
                self._subarr_collapse(natarr, subarr_type), subarr_type
                )
            order = np.argsort(self._flatten(self.roots, 'data'))
            if subarr_type == 'data':
                rearr = rearr[order]
            elif subarr_type == 'index':
                rearr = [rearr[iord] for iord in order]

        return rearr

    def _flatten(self, arr, subarr_type, subarr_trans=False):
        """Flatten a natural structure array.

        Parameters
        ----------
        arr : list of float or tuple, array_like
            Natural structure array.
        subarr_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'`` for
            index arrays.
        subarr_trans : bool, optional
            If `True` (default is `False`), each subarray is flattened along
            the columns rather than rows by a transposition.

        Returns
        -------
        float or tuple, array_like
            Flat 1-d array.

        Raises
        ------
        ValueError
            If `subarr_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarr_trans:
            arr = self._subarr_transpose(arr, subarr_type)

        if subarr_type == 'data':
            return np.concatenate(
                [np.array(arrblock).flatten() for arrblock in arr]
                )
        if subarr_type == 'index':
            return [
                entry for block in arr for line in block for entry in line
                ]

        raise ValueError("Invalid `subarr_type` value. ")

    @staticmethod
    def _alias(structname):
        """Replace aliases of stucture names by the default structure name.

        Parameters
        ----------
        structname : str
            Array structure name.

        Returns
        -------
        str
            Equivalent array structure name.

        """
        if structname == 'natural':
            return 'lmn'
        return structname

    @staticmethod
    def _subarr_transpose(arr, subarr_type):
        """Transpose array elements in a list.

        Parameters
        ----------
        arr : list of float or tuple, array_like
            List of subarrays.
        subarr_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'`` for
            index arrays.

        Returns
        -------
        list of float or tuple, array_like
            List of transposed subarrays.

        Raises
        ------
        ValueError
            If `subarr_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarr_type == 'data':
            return [np.array(ellblock).T for ellblock in arr]
        if subarr_type == 'index':
            return [list(map(list, zip(*ellblock))) for ellblock in arr]

        raise ValueError("Invalid `subarr_type` value. ")

    @staticmethod
    def _subarr_collapse(arr, subarr_type):
        """Collapse a natural structure array over equivalent spherical orders
        while preserving array dimensions.

        Parameters
        ----------
        arr : list of float or tuple, array_like
            Natural structure array.
        subarr_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'`` for
            index arrays.

        Returns
        -------
        list of float or tuple, array_like
            List of collapsed subarrays along the spherical order axis.

        Raises
        ------
        ValueError
            If `subarr_type` is neither ``'data'`` nor ``'index'``.

        """
        if subarr_type == 'data':
            return [
                np.mean(ellblock, axis=0, keepdims=True) for ellblock in arr
                ]
        if subarr_type == 'index':
            return [
                [list(map(lambda tup: (tup[0], tup[-1]), ellblock[0]))]
                for ellblock in arr
                ]

        raise ValueError("Invalid `subarr_type` value. ")

    @staticmethod
    def _subarr_repeat(arr, subarr_type, degrees=None):
        """Repeat an array collapsed over equivalent spherical orders to
        recover the natural structure array.

        Parameters
        ----------
        arr : list of float or tuple, array_like
            Array collapsed over spherical orders.
        subarr_type : {'data', 'index'}
            Subarray type, either ``'data'`` for data arrays or ``'index'`` for
            index arrays.
        degrees : list of int, array_like or None
            Spherical degrees for which equivalent spherical order arrays are
            repeated (default is `None`).  If it is `None`, the degrees are
            inferred from the length of `arr` assuming the first degree is 0.

        Returns
        -------
        list of float or tuple, array_like
            Uncollapsed natural structure array.

        Raises
        ------
        ValueError
            If `subarr_type` is neither ``'data'`` nor ``'index'``.
        ValueError
            If length of `arr` disagrees with input `degrees`.

        """
        if degrees is None:
            degrees = np.arange(len(arr))
        elif len(degrees) != len(arr):
            raise ValueError("`arr` length and `degrees` disagree. ")

        if subarr_type == 'data':
            return [
                np.repeat(np.atleast_2d(line), 2*ell+1, axis=0)
                for line, ell in zip(arr, degrees)
                ]
        if subarr_type == 'index':
            return [
                [list(map(lambda tup: (tup[0], m, tup[-1]), lineblock[0]))
                 for m in range(-ell, ell+1)]
                for lineblock, ell in zip(arr, degrees)
                ]

        raise ValueError("Invalid `subarr_type` value. ")
