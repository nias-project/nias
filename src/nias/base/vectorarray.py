# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from abc import abstractmethod
from numbers import Integral
from typing import Self

import numpy as np
from numpy.typing import NDArray

from nias.interfaces import ArrayLike, Indices, Scalar, VectorArray


class VectorArrayBase(VectorArray):

    base = None
    is_view: bool = False
    _impl = None
    _ind = None

    def __init__(self, *, impl=None, base=None, ind=None, _len=None):
        assert ind is None or base is not None
        self._impl = impl
        self.scalar_type = impl.scalar_type
        self.real_scalar_type = impl.real_scalar_type
        if base is None:
            self._refcount = [1]
            self._len = len(impl)
        else:
            self.is_view = True
            self.base = base
            self._ind = ind
            self._len = _len

    def __len__(self) -> int:
        return self._len

    def is_compatible_array(self, other: VectorArray) -> bool:
        if self.real_scalar_type != other.real_scalar_type:
            return False
        return self._impl.is_compatible_array(other._impl)

    def copy(self) -> Self:
        if self.is_view:
            return type(self)(impl=self._impl.copy(self._ind))
        else:
            C = type(self)(impl=self._impl)
            C._refcount = self._refcount
            self._refcount[0] += 1
            return C

    def append(self, other: VectorArray, remove_from_other: bool = False) -> None:
        assert isinstance(other, VectorArrayBase)
        assert self.is_compatible_array(other)
        if self.is_view:
            raise ValueError('Cannot append to VectorArray view')
        if remove_from_other and self._impl is other._impl:
            raise ValueError('Cannot append VectorArray to itself with remove_from_other=True')
        self._copy_impl_if_multiple_refs()
        if remove_from_other:
            other._copy_impl_if_multiple_refs()
        self._impl.append(other._impl, remove_from_other, other._ind)
        self._len += other._len
        if remove_from_other:
            if other.is_view:
                other.base._len = len(other.base._impl)  # update _len from impl
            else:
                other._len = 0

    def __getitem__(self, ind: ArrayLike) -> VectorArray:
        l = self._len

        # normalize ind s.t. the length of the view does not change when
        # the array is appended to
        if isinstance(ind, Integral):
            if 0 <= ind < l:
                ind = slice(ind, ind+1)
            elif ind >= l or ind < -l:
                raise IndexError('VectorArray index out of range')
            else:
                ind = l+ind
                ind = slice(ind, ind+1)
            view_len = 1
        elif type(ind) is slice:
            start, stop, step = ind.indices(l)
            if start == stop:
                ind = slice(0, 0, 1)
                view_len = 0
            else:
                assert start >= 0
                assert stop >= 0 or (step < 0 and stop >= -1)
                ind = slice(start, None if stop == -1 else stop, step)
                view_len = len(range(start, stop, step))
        else:
            assert isinstance(ind, (list, np.ndarray))
            assert all(-l <= i < l for i in ind)
            ind = [i if 0 <= i else l+i for i in ind]
            view_len = len(ind)

        if self.is_view:
            ind = self._sub_index(self.base._len, self._ind, ind)
            return type(self)(impl=self._impl, base=self.base, ind=ind, _len=view_len)
        else:
            return type(self)(impl=self._impl, base=self, ind=ind, _len=view_len)

    def __setitem__(self, indices: ArrayLike, other: VectorArray) -> None:
        assert not self.is_view or self.base.check_ind_unique(self._ind)
        assert isinstance(other, VectorArrayBase)
        assert self.is_compatible_array(other)
        assert len(self) == len(other) or len(other) == 1
        self._copy_impl_if_multiple_refs()
        self._impl.setitem(other._impl, self._ind, other._ind)

    def __delitem__(self, ind: ArrayLike):
        if self.is_view:
            raise ValueError('Cannot delete items from VectorArray view')
        assert self.check_ind(ind)
        self._copy_impl_if_multiple_refs()
        self._impl.delete(ind)
        self._len = len(self._impl)

    def scal(self, alpha: ArrayLike) -> None:
        assert not self.is_view or self.base.check_ind_unique(self._ind)
        assert isinstance(alpha, Scalar) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)
        self._copy_impl_if_multiple_refs()
        self._impl.scal(alpha, self._ind)

    def axpy(self, alpha: ArrayLike, x: VectorArray) -> None:
        assert not self.is_view or self.base.check_ind_unique(self._ind)
        assert isinstance(alpha, Scalar) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)
        assert isinstance(x, VectorArrayBase)
        assert self.is_compatible_array(x)
        assert len(self) == len(x) or len(x) == 1
        self._copy_impl_if_multiple_refs()
        self._impl.axpy(alpha, x._impl, self._ind, x._ind)

    def lincomb(self, coefficients: ArrayLike) -> VectorArray:
        assert 1 <= coefficients.ndim <= 2
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, ...]
        assert coefficients.shape[-1] == len(self)
        return type(self)(impl=self._impl.lincomb(coefficients, self._ind))

    @property
    def real(self) -> VectorArray:
        if not self.is_complex:
            return self.copy()
        impl = self._impl.real(self._ind)
        return type(self)(impl=impl)

    @property
    def imag(self) -> VectorArray:
        return type(self)(impl=self._impl.imag(self._ind))

    def conj(self) -> VectorArray:
        if not self.is_complex:
            return self.copy()
        impl = self._impl.conj(self._ind)
        return type(self)(impl=impl)

    def _dual_pairing(self, other: VectorArray, pairwise: bool) -> NDArray:
        assert isinstance(other, VectorArrayBase)
        assert self.is_compatible_array(other)
        assert not pairwise or len(self) == len(other)
        return self._impl.dual_pairing(other._impl, self._ind, other._ind, pairwise)

    def _copy_impl_if_multiple_refs(self):
        array = self.base if self.is_view else self
        if array._refcount[0] == 1:
            return
        array._impl = array._impl.copy(None)  # copy the array implementation
        array._refcount[0] -= 1               # decrease refcount for original array
        array._refcount = [1]                 # create new reference counter

    @staticmethod
    def _sub_index(l, ind, ind_ind):
        if type(ind) is slice:
            ind = range(*ind.indices(l))
            if type(ind_ind) is slice:
                result = ind[ind_ind]
                return slice(result.start, result.stop, result.step)
            elif hasattr(ind_ind, '__len__'):
                return [ind[i] for i in ind_ind]
            else:
                return [ind[ind_ind]]
        else:
            if not hasattr(ind, '__len__'):
                ind = [ind]
            if type(ind_ind) is slice:
                return ind[ind_ind]
            elif hasattr(ind_ind, '__len__'):
                return [ind[i] for i in ind_ind]
            else:
                return [ind[ind_ind]]

    def __del__(self):
        if not self.is_view:
            self._refcount[0] -= 1


class VectorArrayImpl:
    scalar_type: type

    @property
    def real_scalar_type(self) -> type:
        return np.empty(0, dtype=self.scalar_type).real.dtype

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def is_compatible_array(self, other: 'VectorArrayImpl') -> bool:
        pass

    @abstractmethod
    def copy(self, ind: Indices) -> 'VectorArrayImpl':
        pass

    @abstractmethod
    def append(self, other: 'VectorArrayImpl', remove_from_other: bool, oind: Indices) -> None:
        pass

    @abstractmethod
    def delete(self, ind: Indices) -> None:
        pass

    @abstractmethod
    def setitem(self, other: 'VectorArrayImpl', ind: Indices, oind: Indices) -> None:
        pass

    @abstractmethod
    def scal(self, alpha: ArrayLike, ind: Indices) -> None:
        pass

    @abstractmethod
    def axpy(self, alpha: ArrayLike, x: 'VectorArrayImpl', ind: Indices, xind: Indices) -> None:
        pass

    @abstractmethod
    def lincomb(self, coefficients: ArrayLike, ind: Indices) -> 'VectorArrayImpl':
        pass

    @abstractmethod
    def real(self, ind: Indices) -> 'VectorArrayImpl':
        pass

    @abstractmethod
    def imag(self, ind: Indices) -> 'VectorArrayImpl':
        pass

    @abstractmethod
    def conj(self, ind: Indices) -> 'VectorArrayImpl':
        pass

    @abstractmethod
    def dual_pairing(self, other: 'VectorArrayImpl', ind: Indices, oind: Indices, pairwise: bool) -> NDArray:
        pass