# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg.lapack import get_lapack_funcs
from scipy.sparse.linalg import splu, spsolve

from nias.base.linear_solvers import default_factory
from nias.base.operators import IdentityOperator, LincombOperator
from nias.bindings.numpy.operators import NumpyMatrixOperator, assemble_lincomb
from nias.exceptions import InversionError
from nias.interfaces import LinearOperator, LinearSolver, VectorArray


class DirectSolver(LinearSolver):

    def __init__(self, lhs: LinearOperator, keep_factorization: bool,
                 permc_spec: str, check_finite: bool, check_cond: bool):
        self.keep_factorization, self.permc_spec, self.check_finite, self.check_cond = \
            keep_factorization, permc_spec, check_finite, check_cond
        self.factorization = None
        self.set_lhs(lhs)

    def set_lhs(self, lhs: LinearOperator) -> None:
        if isinstance(lhs, LincombOperator):
            # TODO: move this somewhere else
            assert all(isinstance(o, (IdentityOperator, NumpyMatrixOperator)) for o in lhs.operators)
            identity_shift = 0
            ops, coeffs = [], []
            for o, c in zip(lhs.operators, lhs.coefficients):
                if isinstance(o, IdentityOperator):
                    identity_shift += c
                else:
                    ops.append(o)
                    coeffs.append(c)
            if not ops:
                raise NotImplementedError
            lhs = assemble_lincomb(ops, coeffs, identity_shift=identity_shift)
        assert isinstance(lhs, NumpyMatrixOperator)
        self.lhs = lhs

    def solve(self, rhs: VectorArray) -> VectorArray:
        assert rhs in self.lhs.range_space
        matrix = self.lhs.matrix
        V = self.lhs.range_space.to_numpy(rhs)
        promoted_type = np.promote_types(matrix.dtype, V.dtype)

        if not self.lhs.sparse:
            if self.factorization is None:
                try:
                    self.factorization = lu_factor(matrix, check_finite=self.check_finite)
                except np.linalg.LinAlgError as e:
                    raise InversionError(f'{type(e)!s}: {e!s}') from e
                if self.check_cond:
                    gecon = get_lapack_funcs('gecon', self.factorization)
                    rcond, _ = gecon(self.factorization[0], np.linalg.norm(matrix, ord=1), norm='1')
                    if rcond < np.finfo(np.float64).eps:
                        self.logger.warning(f'Ill-conditioned matrix (rcond={rcond:.6g}) in apply_inverse: '
                                            'result may not be accurate.')
            R = lu_solve(self.factorization, V.T, check_finite=self.check_finite).T
            return self.lhs.source_space.from_numpy(R)

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



default_factory.register_solver(
    (NumpyMatrixOperator, IdentityOperator), 'scipy-spsolve', '10',
    DirectSolver,
    {'keep_factorization': True, 'permc_spec': 'COLAMD', 'check_finite': True, 'check_cond': True}
)


# unfortunately, this is necessary, as scipy does not
# forward the copy=False argument in its csc_matrix.astype function
def matrix_astype_nocopy(matrix, dtype):
    if matrix.dtype == dtype:
        return matrix
    else:
        return matrix.astype(dtype)
