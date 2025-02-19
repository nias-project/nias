from numpy.typing import NDArray

from nias.interfaces import InnerProduct, VectorArray


class NiasCppInnerProduct(InnerProduct):
    def __init__(self, impl):
        self.impl = impl

    def apply(self, left: VectorArray, right: VectorArray, pairwise: bool = False) -> NDArray:
        return self.impl.apply(left.impl.impl, right.impl.impl, pairwise, left.ind, right.ind)
