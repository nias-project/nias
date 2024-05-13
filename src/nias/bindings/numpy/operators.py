# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from functools import reduce

import numpy as np
import scipy.sparse as sps

from nias.bindings.numpy.vectorarrays import NumpyVectorArray, NumpyVectorSpace
from nias.interfaces import LinearOperator, VectorSpaceWithBasis


class NumpyMatrixOperator(LinearOperator):

    sparse: bool

    def __init__(self, matrix,
                 range_space: VectorSpaceWithBasis | None = None,
                 source_space: VectorSpaceWithBasis | None = None):
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        if source_space is None:
            source_space = NumpyVectorSpace(matrix.shape[1])
        if range_space is None:
            range_space = NumpyVectorSpace(matrix.shape[0])

        self.matrix = matrix
        assert False
        self.source_space = source_space
        self.range_space = range_space
        self.sparse = sps.issparse(matrix)

    def apply(self, U: NumpyVectorArray) -> NumpyVectorArray:
        assert U in self.source_space
        return self.range_space.from_numpy((self.matrix @ U._to_numpy().T).T)

    def apply_transpose(self, V: NumpyVectorArray) -> NumpyVectorArray:
        assert V in self.range_space.antidual_space
        return self.source_space.antidual_space.from_numpy((self.matrix.T @ V.to_numpy().T).T)


def assemble_lincomb(operators, coefficients, identity_shift=0.):
    assert all(isinstance(op, NumpyMatrixOperator) for op in operators)

    common_mat_dtype = reduce(np.promote_types,
                              (op.matrix.dtype for op in operators if hasattr(op, 'matrix')))
    common_coef_dtype = reduce(np.promote_types, (type(c) for c in coefficients + [identity_shift]))
    common_dtype = np.promote_types(common_mat_dtype, common_coef_dtype)

    if coefficients[0] == 1:
        matrix = operators[0].matrix.astype(common_dtype)
    else:
        matrix = operators[0].matrix * coefficients[0]
        if matrix.dtype != common_dtype:
            matrix = matrix.astype(common_dtype)

    for op, c in zip(operators[1:], coefficients[1:]):
        if c == 1:
            try:
                matrix += op.matrix
            except NotImplementedError:
                matrix = matrix + op.matrix
        elif c == -1:
            try:
                matrix -= op.matrix
            except NotImplementedError:
                matrix = matrix - op.matrix
        else:
            try:
                matrix += (op.matrix * c)
            except NotImplementedError:
                matrix = matrix + (op.matrix * c)

    if identity_shift != 0:
        if identity_shift.imag == 0:
            identity_shift = identity_shift.real
        if operators[0].sparse:
            try:
                matrix += (sps.eye(matrix.shape[0]) * identity_shift)
            except NotImplementedError:
                matrix = matrix + (sps.eye(matrix.shape[0]) * identity_shift)
        else:
            matrix += (np.eye(matrix.shape[0]) * identity_shift)

    return NumpyMatrixOperator(matrix, source_space=operators[0].source_space, range_space=operators[0].range_space)
