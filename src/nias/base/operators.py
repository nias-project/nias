# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nias.base.linear_solvers import default_factory
from nias.interfaces import (
    HSLinearOperator,
    InnerProduct,
    LinearOperator,
    LinearSolverFactory,
    Norm,
    Operator,
    SesquilinearForm,
    VectorArray,
    dual_pairing,
)


@overload
def lincomb(operators: list[Operator], coefficients: ArrayLike) -> Operator:
    ...

@overload
def lincomb(operators: list[LinearOperator], coefficients: ArrayLike) -> LinearOperator:
    ...

@overload
def lincomb(operators: list[HSLinearOperator], coefficients: ArrayLike) -> HSLinearOperator:
    ...


def lincomb(operators, coefficients):
    ops = []
    coeffs = []
    for o, c in zip(operators, coefficients):
        if isinstance(o, LincombOperator):
            ops.extend(o.operators)
            coeffs.extend(o.coefficients * c)
        else:
            ops.append(o)
            coeffs.append(c)

    if all(isinstance(o, HSLinearOperator) for o in ops):
        return HSLinearLincombOperator(ops, coeffs)
    elif all(isinstance(o, LinearOperator) for o in ops):
        return LinearLincombOperator(ops, coeffs)
    else:
        return LincombOperator(ops, coeffs)


class LincombOperator(Operator):
    def __init__(self, operators: list[Operator], coefficients: ArrayLike):
        coefficients = np.array(coefficients)
        assert len(operators) >= 1
        assert coefficients.ndim == 1
        assert len(operators) == len(coefficients)
        assert all(op.source_space == operators[0].source_space for op in operators)
        assert all(op.range_space == operators[0].range_space for op in operators)
        self.operators = list(operators)
        self.coefficients = coefficients
        self.range_space = operators[0].range_space
        self.source_space = operators[0].source_space

    def apply(self, U: VectorArray) -> VectorArray:
        assert U in self.source_space
        if self.coefficients[0] != 0:
            V = self.operators[0].apply(U)
            V.scal(self.coefficients[0])
        else:
            V = self.range_space.zeros(len(U))
        for op, c in zip(self.operators[1:], self.coefficients[1:]):
            V.axpy(c, op.apply(U))
        return V


class LinearLincombOperator(LincombOperator, LinearOperator):

    def __init__(self, operators: list[LinearOperator], coefficients: ArrayLike):
        super().__init__(operators, coefficients)
        assert all(isinstance(o, LinearOperator) for o in operators)

    def apply_transpose(self, V: VectorArray) -> VectorArray:
        assert V in self.range_space.antidual_space
        if self.coefficients[0] != 0:
            U = self.operators[0].apply_transpose(V)
            U.scal(self.coefficients[0])
        else:
            U = self.range_space.zeros(len(V))
        for op, c in zip(self.operators[1:], self.coefficients[1:]):
            U.axpy(c, op.apply_transpose(V))
        return V


class HSLinearLincombOperator(LinearLincombOperator, HSLinearOperator):

    def __init__(self, operators: list[HSLinearOperator], coefficients: ArrayLike):
        super().__init__(operators, coefficients)
        assert all(isinstance(o, HSLinearOperator) for o in operators)

    def apply_adjoint(self, V: VectorArray) -> VectorArray:
        assert V in self.range_space
        if self.coefficients[0] != 0:
            U = self.operators[0].apply_adjoint(V)
            U.scal(self.coefficients[0])
        else:
            U = self.range_space.zeros(len(V))
        for op, c in zip(self.operators[1:], self.coefficients[1:]):
            U.axpy(c, op.apply_adjoint(V))
        return V


class OperatorBasedSesquilinearForm(SesquilinearForm):

    def __init__(self, operator: LinearOperator):
        self.operator = operator
        self.source_space = operator.source_space
        self.range_space = operator.range_space.antidual_space

    def apply(self, left: VectorArray, right: VectorArray, pairwise: bool = False) -> NDArray:
        return dual_pairing(left, self.operator.apply(right), pairwise=pairwise)


class OperatorBasedInnerProduct(OperatorBasedSesquilinearForm, InnerProduct):

    pass


class InnerProductBasedNorm(Norm):

    def __init__(self, inner_product: InnerProduct):
        self.inner_product = inner_product

    def __call__(self, U: VectorArray) -> NDArray:
        return np.sqrt(self.inner_product.apply(U, U, pairwise=True))


@overload
def inverse(operator: LinearOperator, context: str = '',
            solver_factory: LinearSolverFactory = default_factory) -> LinearOperator:
    ...

@overload
def inverse(operator: HSLinearOperator, context: str = '',
            solver_factory: LinearSolverFactory = default_factory) -> HSLinearOperator:
    ...


def inverse(operator):
    assert isinstance(operator, LinearOperator)
    if isinstance(operator, HSLinearOperator):
        return InverseHSLinearOperator(operator)
    elif isinstance(operator, LinearOperator):
        return InverseLinearOperator(operator)
    else:
        raise NotImplementedError


class InverseLinearOperator(LinearOperator):

    def __init__(self, operator: LinearOperator, context: str = '',
                 solver_factory: LinearSolverFactory = default_factory):
        self.operator = operator
        self.solver = default_factory.get_solver(operator, context)
        self.range = operator.source
        self.source = operator.range

    def apply(self, U: VectorArray) -> VectorArray:
        assert U in self.source
        return self.solver.solve(U)

    def apply_transpose(self, V: VectorArray) -> VectorArray:
        assert V in self.range
        return self.solver.solve_transposed(V)


class InverseHSLinearOperator(InverseLinearOperator, HSLinearOperator):
    pass
