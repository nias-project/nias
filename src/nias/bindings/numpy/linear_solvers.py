# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.sparse.linalg import splu, spsolve

from nias.base.linear_solvers import default_factory
from nias.base.operators import LincombOperator
from nias.bindings.numpy.operators import NumpyMatrixOperator, assemble_lincomb
from nias.exceptions import InversionError
from nias.interfaces import LinearOperator, LinearSolver, VectorArray


class SpsolveSolver(LinearSolver):

    def __init__(self, lhs: LinearOperator, keep_factorization: bool,
                 permc_spec: str):
        self.keep_factorization = keep_factorization
        self.factorization = None
        self.permc_spec = permc_spec
        self.set_lhs(lhs)

    def set_lhs(self, lhs: LinearOperator) -> None:
        if isinstance(lhs, LincombOperator):
            assert all(isinstance(o, NumpyMatrixOperator) for o in lhs.operators)
            lhs = assemble_lincomb(lhs.operators, lhs.coefficients)
        assert isinstance(lhs, NumpyMatrixOperator)
        self.lhs = lhs

    def solve(self, rhs: VectorArray) -> VectorArray:
        assert rhs in self.lhs.range_space
        matrix = self.lhs.matrix
        V = self.lhs.range_space.to_numpy(rhs)
        promoted_type = np.promote_types(matrix.dtype, V.dtype)

        try:
            # maybe remove unusable factorization:
            if self.factorization is not None:
                fdtype = self.factorizationdtype
                if not np.can_cast(V.dtype, fdtype, casting='safe'):
                    self.factorization = None

            if self.factorization is not None:
                # we may use a complex factorization of a real matrix to
                # apply it to a real vector. In that case, we downcast
                # the result here, removing the imaginary part,
                # which should be zero.
                R = self.factorization.solve(V.T).T.astype(promoted_type, copy=False)
            elif self.keep_factorization:
                # the matrix is always converted to the promoted type.
                # if matrix.dtype == promoted_type, this is a no_op
                self.factorization = splu(matrix_astype_nocopy(matrix.tocsc(), promoted_type),
                                          permc_spec=self.permc_spec)
                self.factorizationdtype = promoted_type
                R = self.factorization.solve(V.T).T
            else:
                # the matrix is always converted to the promoted type.
                # if matrix.dtype == promoted_type, this is a no_op
                R = spsolve(matrix_astype_nocopy(matrix, promoted_type), V.T, permc_spec=self.permc_spec).T
        except RuntimeError as e:
            raise InversionError(e) from e

        return self.lhs.source_space.from_numpy(R)

    def solve_transposed(self, rhs: VectorArray) -> VectorArray:
        pass



default_factory.register_solver((NumpyMatrixOperator,), 'scipy-spsolve', '10',
                                SpsolveSolver, {'keep_factorization': True, 'permc_spec': 'COLAMD'})


# unfortunately, this is necessary, as scipy does not
# forward the copy=False argument in its csc_matrix.astype function
def matrix_astype_nocopy(matrix, dtype):
    if matrix.dtype == dtype:
        return matrix
    else:
        return matrix.astype(dtype)
