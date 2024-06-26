import numpy as np
from nias.interfaces import InnerProduct, VectorArray, VectorSpace
from numpy.typing import NDArray

from nias.bindings.nias_cpp.interfaces import ListVectorArray, ListVectorArrayImpl, ListVectorSpace, Vector


class NiasCppVector(Vector):
    def __init__(self, impl):
        self.impl = impl
        self.dim = len(impl)

    def __eq__(self, other):
        return type(self) == type(other) and self.impl == other.impl

    def __getitem__(self, ind):
        return self.impl[ind]

    def __setitem__(self, ind, val):
        self.impl[ind] = val

    def __len__(self):
        return self.dim

    def __iter__(self):
        for i in range(self.dim):
            yield self[i]

    def copy(self, deep=False):
        return NiasCppVector(self.impl.copy())

    def scal(self, alpha):
        self.impl.scal(alpha)

    def axpy(self, alpha, x):
        self.impl.axpy(alpha, x.impl)

    def dot(self, other):
        return self.impl.dot(other.impl)


class NiasCppListVectorSpace(ListVectorSpace):
    def __init__(self, dim, vector_type):
        self.dim = dim
        self.vector_type = vector_type

    def __eq__(self, other):
        return type(other) is NiasCppListVectorSpace and self.dim == other.dim and self.vector_type == other.vector_type

    def __ge__(self, other: "VectorSpace") -> bool:
        raise NotImplementedError
        return isinstance(other, NiasCppListVectorSpace) and self.dim == other.dim

    def __contains__(self, element: VectorArray) -> bool:
        return isinstance(element, ListVectorArray) and element.dim == self.dim

    def empty(self, size_hint=0) -> ListVectorArray:
        return ListVectorArray(impl=ListVectorArrayImpl([], self.dim))

    @property
    def antidual_space(self) -> "NiasCppListVectorSpace":
        return self

    def zero_vector(self):
        return NiasCppVector(self.vector_type(self.dim))

    def make_vector(self, obj):
        return NiasCppVector(obj)

    def from_data(self, data):
        raise NotImplementedError

    def from_vectors(self, vecs):
        assert isinstance(vecs, (list, NiasCppVector))
        if isinstance(vecs, NiasCppVector):
            vecs = [vecs]
        return ListVectorArray(impl=ListVectorArrayImpl(vecs, len(vecs[0])))


class NiasCppInnerProduct(InnerProduct):
    def apply(self, left: VectorArray, right: VectorArray, pairwise: bool = False) -> NDArray:
        if pairwise:
            ret = [left_vec.dot(right_vec) for left_vec, right_vec in zip(left.vectors, right.vectors)]
            return np.array([ret])
        else:
            ret = np.zeros(shape=(len(left), len(right)))
            for i, left_vec in enumerate(left.vectors):
                for j, right_vec in enumerate(right.vectors):
                    ret[i, j] = left_vec.dot(right_vec)
            return ret
