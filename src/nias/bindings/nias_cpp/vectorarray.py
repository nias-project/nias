import numpy as np
from nias.base.vectorarrays import VectorArrayBase, VectorArrayImpl

class NiasCppVectorArrayImpl(VectorArrayImpl):

    def __init__(self, impl):
        self.impl = impl
        self.dim = impl.dim
        self.scalar_type = type(impl.get(0))

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
    """|VectorArray| implemented in C++."""

    impl: NiasCppVectorArrayImpl

    def __str__(self):
        return f"{type(self).__name__} of {len(self.impl)} vectors of dimension {self.impl.dim}"

    @property
    def vectors(self):
            indices = self.impl._index_to_list(self.ind)
            if len(indices) == 0:
                return [self.impl.impl.get(i) for i in range(len(self.impl.impl))]
            else:
                return [self.impl.impl.get(i) for i in indices]

