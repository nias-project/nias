import numpy as np

from nias.base.vectorarrays import VectorArrayBase, VectorArrayImpl


class NiasCppVectorArrayImpl(VectorArrayImpl):

    def __init__(self, impl):
        self.impl = impl
        self.dim = impl.dim
        self.scalar_type = type(impl.scalar_zero())

    def is_compatible_array(self, other: 'VectorArrayImpl') -> bool:
        return isinstance(other, NiasCppVectorArrayImpl) and self.impl.is_compatible_array(other.impl)

    def __len__(self):
        return len(self.impl)

    def delete(self, ind):
        self.impl.delete(ind)

    def append(self, other, remove_from_other, oind):
        self.impl.append(other.impl, remove_from_other, oind)

    def copy(self, ind):
        return NiasCppVectorArrayImpl(self.impl.copy(ind))

    def scal(self, alpha, ind):
        self.impl.scal(alpha, ind)

    def axpy(self, alpha, x, ind, xind):
        if np.all(alpha == 0):
            return
        self.impl.axpy(alpha, x.impl, ind, xind)

    def _index_to_list(self, ind):
        if ind is None:
            return None
        elif type(ind) is slice:
            return list(range(len(self))[ind])
        else:
            return list(ind)

class NiasCppVectorArray(VectorArrayBase):
    """|VectorArray| implemented in C++."""

    impl: NiasCppVectorArrayImpl

    def __str__(self):
        num_vecs = len(self.impl._index_to_list(self.ind) if self.ind else self.impl)
        return f'{type(self).__name__} of {num_vecs} vectors of dimension {self.impl.dim}'
