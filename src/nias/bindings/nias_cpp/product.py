import numpy as np
from nias.interfaces import InnerProduct, VectorArray
from numpy.typing import NDArray

class NiasCppInnerProduct(InnerProduct):
    def apply(self, left: VectorArray, right: VectorArray, pairwise: bool = False) -> NDArray:
        if pairwise:
            ret = [left_vec.dot(right_vec) for left_vec, right_vec in zip(left.vectors, right.vectors)]
            return np.array([ret])
        else:
            ret = np.zeros(shape=(len(left), len(right)), dtype=left.scalar_type)
            for i, left_vec in enumerate(left.vectors):
                for j, right_vec in enumerate(right.vectors):
                    ret[i, j] = left_vec.dot(right_vec)
            return ret
