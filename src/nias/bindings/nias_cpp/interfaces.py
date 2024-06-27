# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from abc import abstractmethod

import numpy as np
from nias.base.vectorarrays import VectorArrayBase, VectorArrayImpl
from nias.interfaces import (
    VectorSpace,
)


class Vector:
    """Interface for vectors used in conjunction with |ListVectorArray|.

    This interface must be satisfied by the individual entries of the
    vector `list` managed by |ListVectorArray|. All interface methods
    have a direct counterpart in the |VectorArray| interface.
    """

    @abstractmethod
    def copy(self, deep=False):
        pass

    @abstractmethod
    def scal(self, alpha):
        pass

    @abstractmethod
    def axpy(self, alpha, x):
        pass

    @abstractmethod
    def dot(self, other):
        pass


class ListVectorArrayImpl(VectorArrayImpl):
    scalar_type = float

    def __init__(self, vectors, dim):
        self._list = vectors
        self.dim = dim
        assert all(v.dim == dim for v in vectors)

    def _indexed(self, ind):
        if ind is None:
            return self._list
        elif type(ind) is slice:
            return self._list[ind]
        elif hasattr(ind, "__len__"):
            return [self._list[i] for i in ind]
        else:
            return [self._list[ind]]

    def is_compatible_array(self, other: "VectorArrayImpl") -> bool:
        return isinstance(other, ListVectorArrayImpl) and self.dim == other.dim

    def __len__(self):
        return len(self._list)

    def delete(self, ind):
        if ind is None:
            del self._list[:]
        elif hasattr(ind, "__len__"):
            thelist = self._list
            length = len(thelist)
            remaining = sorted(set(range(length)) - {i if 0 <= i else length + i for i in ind})
            self._list = [thelist[i] for i in remaining]
        else:
            del self._list[ind]

    def append(self, other, remove_from_other, oind):
        if not remove_from_other:
            self._list.extend([v.copy() for v in other._indexed(oind)])
        else:
            self._list.extend(other._indexed(oind))
            other.delete(oind)

    def copy(self, ind):
        return ListVectorArrayImpl([v.copy() for v in self._indexed(ind)], self.dim)

    def scal(self, alpha, ind):
        if type(alpha) is np.ndarray:
            for a, v in zip(alpha, self._indexed(ind)):
                v.scal(a)
        else:
            for v in self._indexed(ind):
                v.scal(alpha)

    def axpy(self, alpha, x, ind, xind):
        if np.all(alpha == 0):
            return

        x_list = x.copy(xind)._list if self is x else x._indexed(xind)

        if len(x_list) == 1:
            xx = next(iter(x_list))
            if type(alpha) is np.ndarray:
                for a, y in zip(alpha, self._indexed(ind)):
                    y.axpy(a, xx)
            else:
                for y in self._indexed(ind):
                    y.axpy(alpha, xx)
        elif type(alpha) is np.ndarray:
            for a, xx, y in zip(alpha, x_list, self._indexed(ind)):
                y.axpy(a, xx)
        else:
            for xx, y in zip(x_list, self._indexed(ind)):
                y.axpy(alpha, xx)


class ListVectorArray(VectorArrayBase):
    """|VectorArray| implemented as a Python list of vectors."""

    impl: ListVectorArrayImpl

    def __str__(self):
        return f"{type(self).__name__} of {len(self.impl._list)} vectors of dimension {self.impl.dim}"

    @property
    def vectors(self):
        return self.impl._indexed(self.ind)


class NiasCppVectorArrayImpl(VectorArrayImpl):
    scalar_type = float

    def __init__(self, impl):
        self.impl = impl
        self.dim = impl.dim

    def is_compatible_array(self, other: "VectorArrayImpl") -> bool:
        return isinstance(other, NiasCppVectorArrayImpl) and self.impl.is_compatible_array(other.impl)

    def __len__(self):
        return len(self.impl)

    def delete(self, ind):
        self.impl.delete(self._index_to_list(ind))

    def append(self, other, remove_from_other, oind):
        self.impl.append(other.impl, remove_from_other, self._index_to_list(oind))

    def copy(self, ind):
        return NiasCppVectorArrayImpl(self.impl.copy(self._index_to_list(ind)))

    def scal(self, alpha, ind):
        self.impl.scal(alpha, self._index_to_list(ind))

    def axpy(self, alpha, x, ind, xind):
        if np.all(alpha == 0):
            return
        self.impl.axpy(alpha, x.impl, self._index_to_list(ind), self._index_to_list(xind))

    def _index_to_list(self, ind):
        if ind is None:
            return []
        elif type(ind) is slice:
            return list(range(len(self))[ind])
        elif hasattr(ind, "__len__"):
            return [i for i in ind]
        else:
            return [ind]



class NiasCppVectorArray(VectorArrayBase):
    """|VectorArray| implemented as a Python list of vectors."""

    impl: NiasCppVectorArrayImpl

    def __str__(self):
        return f"{type(self).__name__} of {len(self.impl)} vectors of dimension {self.impl.dim}"

    @property
    def vectors(self):
            indices = self.impl._index_to_list(self.ind)
            if len(indices) == 0:
                return self.impl.impl.vectors
            else:
                return [self.impl.impl.vectors[i] for i in indices]


class ListVectorSpace(VectorSpace):
    """|VectorSpace| of |ListVectorArrays|."""

    dim = None
    vector_type = Vector

    @abstractmethod
    def zero_vector(self):
        pass

    @abstractmethod
    def make_vector(self, obj):
        pass

    def zeros(self, count=1, reserve=0):
        raise NotImplementedError

    def random(self, count=1, distribution="uniform", reserve=0, **kwargs):
        raise NotImplementedError
