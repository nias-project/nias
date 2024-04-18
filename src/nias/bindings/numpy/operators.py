# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps

from nias.bindings.numpy.vectorarrays import NumpyVectorArray, NumpyVectorSpace
from nias.interfaces import LinearOperator, VectorSpace


class NumpyMatrixOperator(LinearOperator):

    sparse: bool

    def __init__(self, matrix, source_space: VectorSpace | None = None, range_space: VectorSpace | None = None):
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        if source_space is None:
            source_space = NumpyVectorSpace(matrix.shape[1])
        if range_space is None:
            range_space = NumpyVectorSpace(matrix.shape[0])

        self.matrix = matrix
        self.source_space = source_space
        self.range_space = range_space
        self.sparse = sps.issparse(matrix)

    def apply(self, U: NumpyVectorArray) -> NumpyVectorArray:
        assert U in self.source_space
        return self.range_space.from_data((self.matrix @ U._to_numpy().T).T)

    def apply_transpose(self, V: NumpyVectorArray) -> NumpyVectorArray:
        assert V in self.range_space.antidual_space
        return self.source_space.antidual_space.from_data((self.matrix.T @ V.to_numpy().T).T)
