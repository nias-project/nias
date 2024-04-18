# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from abc import abstractmethod
from numbers import Number

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse

from nias.base.vectorarrays import VectorArrayBase, VectorArrayImpl
from nias.interfaces import ArrayLike, Indices, VectorArray, VectorSpace


class NumpyBasedVectorArrayImpl(VectorArrayImpl):

    dim: int

    @abstractmethod
    def create_impl(self, data: NDArray) -> 'NumpyBasedVectorArrayImpl':
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def to_numpy(self, ensure_copy: bool, ind: Indices) -> NDArray:
        pass

    @abstractmethod
    def copy(self, ind: Indices) -> 'NumpyBasedVectorArrayImpl':
        pass

    @abstractmethod
    def append(self, other: 'VectorArrayImpl', remove_from_other: bool, oind: Indices) -> None:
        pass

    @abstractmethod
    def delete(self, ind: Indices) -> None:
        pass

    @abstractmethod
    def set_from_numpy(self, other: NDArray, ind: Indices) -> None:
        pass

    @abstractmethod
    def scal(self, alpha: ArrayLike, ind: Indices) -> None:
        pass

    @abstractmethod
    def axpy_from_numpy(self, alpha: ArrayLike, x: NDArray, ind: Indices) -> None:
        pass

    def is_compatible_array(self, other: 'VectorArrayImpl') -> bool:
        return (isinstance(other, NumpyVectorArrayImpl) and self.dim == other.dim)

    def setitem(self, other: 'VectorArrayImpl', ind: Indices, oind: Indices):
        self.set_from_numpy(other.to_numpy(False, oind), ind)

    def axpy(self, alpha: ArrayLike, x: 'VectorArrayImpl', ind: Indices, xind: Indices) -> None:
        ind = slice(None, self._len) if ind is None else ind
        self.axpy_from_numpy(alpha, x.to_numpy(False, xind), ind)

    def lincomb(self, coefficients: ArrayLike, ind: Indices) -> 'NumpyBasedVectorArrayImpl':
        A = self.to_numpy(False, ind)
        return self.create_impl(coefficients.dot(A))

    def real(self, ind: Indices) -> 'NumpyBasedVectorArrayImpl':
        A = self.to_numpy(False, ind)
        assert not np.isrealobj(A)  # real case handled already by VectorArrayBase
        return self.create_impl(A.real)

    def imag(self, ind: Indices) -> 'NumpyBasedVectorArrayImpl':
        A = self.to_numpy(False, ind)
        return self.create_impl(A.imag)

    def conj(self, ind: Indices) -> 'NumpyBasedVectorArrayImpl':
        A = self.to_numpy(False, ind)
        return self.create_impl(A.conj)

    def dual_pairing(self, other: 'VectorArrayImpl', ind: Indices, oind: Indices, pairwise: bool) -> NDArray:
        A = self.to_numpy(False, ind)
        B = other.to_numpy(False, oind)

        if pairwise:
            # .conj() is a no-op on non-complex data types
            return np.sum(A.conj() * B, axis=1)
        else:
            return A.conj().dot(B.T)


class NumpyVectorArrayImpl(NumpyBasedVectorArrayImpl):
    def __init__(self, array, l=None):
        self._array = array
        self._len = len(array) if l is None else l
        self.scalar_type = array.dtype

    @property
    def dim(self) -> int:
        return self._array.shape[1]

    def create_impl(self, data: NDArray) -> 'NumpyVectorArrayImpl':
        return type(self)(data)

    def __len__(self) -> int:
        return self._len

    def to_numpy(self, ensure_copy, ind) -> NDArray:
        A = self._array[:self._len] if ind is None else self._array[ind]
        if ensure_copy and not A.flags['OWNDATA']:
            return A.copy()
        else:
            return A

    def copy(self, ind: Indices) -> 'NumpyVectorArrayImpl':
        new_array = self._array[:self._len] if ind is None else self._array[ind]
        if not new_array.flags['OWNDATA']:
            new_array = new_array.copy()
        return NumpyVectorArrayImpl(new_array)

    def append(self, other: VectorArrayImpl, remove_from_other: bool, oind: Indices) -> 'NumpyVectorArrayImpl':
        other_array = other.to_numpy(False, oind)
        len_other = len(other_array)
        if len_other == 0:
            return

        if len_other <= self._array.shape[0] - self._len:
            if self._array.dtype != other_array.dtype:
                self._array = self._array.astype(np.promote_types(self._array.dtype, other_array.dtype), copy=False)
            self._array[self._len:self._len + len_other] = other_array
        else:
            self._array = np.append(self._array[:self._len], other_array, axis=0)
        self._len += len_other

        if remove_from_other:
            other.delete(oind)

    def delete(self, ind: Indices) -> None:
        if ind is None:
            self._array = np.empty((0, self._array.shape[1]))
            self._len = 0
            return
        if type(ind) is slice:
            ind = set(range(*ind.indices(self._len)))
        elif not hasattr(ind, '__len__'):
            ind = {ind if 0 <= ind else self._len + ind}
        else:
            l = self._len
            ind = {i if 0 <= i else l+i for i in ind}
        remaining = sorted(set(range(len(self))) - ind)
        self._array = self._array[remaining]
        self._len = len(self._array)
        if not self._array.flags['OWNDATA']:
            self._array = self._array.copy()

    def scal(self, alpha: ArrayLike, ind: Indices) -> None:
        ind = slice(None, self._len) if ind is None else ind
        if type(alpha) is np.ndarray:
            alpha = alpha[:, np.newaxis]

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype:
            self._array = self._array.astype(np.promote_types(self._array.dtype, alpha_dtype), copy=False)
        self._array[ind] *= alpha

    def axpy_from_numpy(self, alpha: ArrayLike, x: NDArray, ind: Indices) -> None:
        ind = slice(None, self._len) if ind is None else ind

        alpha_type = type(alpha)
        alpha_dtype = alpha.dtype if alpha_type is np.ndarray else alpha_type
        if self._array.dtype != alpha_dtype or self._array.dtype != x.dtype:
            dtype = np.promote_types(self._array.dtype, alpha_dtype)
            dtype = np.promote_types(dtype, x.dtype)
            self._array = self._array.astype(dtype, copy=False)

        if type(alpha) is np.ndarray:
            alpha = alpha[:, np.newaxis]

        if isinstance(alpha, Number):
            if alpha == 1:
                self._array[ind] += x
                return
            elif alpha == -1:
                self._array[ind] -= x
                return

        self._array[ind] += x * alpha


class NumpyVectorArray(VectorArrayBase):

    impl: NumpyBasedVectorArrayImpl

    def _to_numpy(self, ensure_copy=False) -> NDArray:
        return self.impl.to_numpy(ensure_copy, self._ind)

    def _set_from_numpy(self, other: NDArray) -> None:
        self.impl.set_from_numpy(other, self._ind)

    def _axpy_from_numpy(self, alpha: ArrayLike, x: NDArray) -> None:
        self.impl.axpy_from_numpy(alpha, x, self._ind)

    def __str__(self):
        return str(self._to_numpy())


class NumpyVectorSpace(VectorSpace):

    def __init__(self, dim: int):
        self.dim = int(dim)

    def empty(self, size_hint=0) -> NumpyVectorArray:
        return NumpyVectorArray(impl=NumpyVectorArrayImpl(np.empty((size_hint, self.dim)), l=0))

    def zeros(self, count=1) -> NumpyVectorArray:
        assert count >= 0
        return NumpyVectorArray(impl=NumpyVectorArrayImpl(np.zeros((count, self.dim)), count))

    def random(self, count: int = 1, distribution: str = 'uniform', seed=None, **kwargs) -> NumpyVectorArray:
        assert count >= 0
        return NumpyVectorArray(
            impl=NumpyVectorArrayImpl(_create_random_values((count, self.dim), distribution, seed=seed, **kwargs))
        )

    def from_data(self, data) -> NumpyVectorArray:
        if type(data) is np.ndarray:
            pass
        elif issparse(data):
            data = data.toarray()
        else:
            data = np.array(data, ndmin=2)
        if data.ndim != 2:
            assert data.ndim == 1
            data = np.reshape(data, (1, -1))
        assert data.shape[1] == self.dim
        return NumpyVectorArray(impl=NumpyVectorArrayImpl(data))

    def antidual_space(self) -> 'NumpyVectorSpace':
        return self

    def __contains__(self, element: VectorArray) -> bool:
        return isinstance(element, NumpyVectorArray) and element.impl.dim == self.dim

    def __eq__(self, other: 'VectorSpace') -> bool:
        return type(other) is type(self) and self.dim == other.dim

    def __ge__(self, other: 'VectorSpace') -> bool:
        return isinstance(other, NumpyVectorSpace) and self.dim == other.dim

    def __hash__(self):
        return hash(self.dim)


def _create_random_values(shape, distribution, seed=None, **kwargs):
    if distribution not in ('uniform', 'normal'):
        raise NotImplementedError

    rng = np.random.default_rng(seed)

    if distribution == 'uniform':
        if not kwargs.keys() <= {'low', 'high'}:
            raise ValueError
        low = kwargs.get('low', 0.)
        high = kwargs.get('high', 1.)
        if high <= low:
            raise ValueError
        return rng.uniform(low, high, shape)
    elif distribution == 'normal':
        if not kwargs.keys() <= {'loc', 'scale'}:
            raise ValueError
        loc = kwargs.get('loc', 0.)
        scale = kwargs.get('scale', 1.)
        return rng.normal(loc, scale, shape)
    else:
        assert False
