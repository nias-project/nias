# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from abc import ABC, abstractmethod, abstractproperty
from typing import Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

Scalar: TypeAlias = float | complex | np.number


class VectorArray(ABC):

    is_view: bool
    field_type: type

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def scal(self, alpha: ArrayLike) -> None:
        pass

    @abstractmethod
    def axpy(self, alpha: ArrayLike, x: 'VectorArray') -> None:
        pass

    @abstractmethod
    def lincomb(self, coefficients: ArrayLike) -> 'VectorArray':
        pass

    @abstractmethod
    def __getitem__(self, indices: ArrayLike) -> 'VectorArray':
        pass

    def _dual_pairing(self, other: 'VectorArray') -> NDArray:
        return NotImplemented


def dual_pairing(left: VectorArray, right: VectorArray) -> NDArray:
    result = left._dual_pairing(right)
    if result == NotImplemented:
        result = right._dual_pairing(left)
    if result == NotImplemented:
        raise NotImplementedError('No dual pairing possible.')
    return result


class VectorSpace(ABC):
    dim: int | None

    @abstractmethod
    def empty(self, size_hint=0) -> VectorArray:
        pass

    @abstractmethod
    def random(self, count: int = 1) -> VectorArray:
        pass

    @abstractmethod
    def from_data(self, data) -> VectorArray:
        pass

    @abstractproperty
    def antidual_space(self) -> 'VectorSpace':
        pass

    @abstractmethod
    def __contains__(self, element: 'VectorArray') -> bool:
        pass

    @abstractmethod
    def __ge__(self, other: 'VectorSpace') -> bool:
        """`True` if `other` embeds into `self`."""

    def __le__(self, other: 'VectorSpace') -> bool:
        return other >= self


class VectorSpaceWithBasis(VectorSpace):
    dim: int

    @abstractmethod
    def from_numpy(self, data, ensure_copy=False) -> VectorArray:
        pass

    @abstractmethod
    def to_numpy(self, U: VectorArray, ensure_copy=False) -> NDArray:
        pass

    @abstractmethod
    def l2_norm(self, U: VectorArray) -> NDArray:
        pass

    @abstractmethod
    def amax(self, U: VectorArray) -> (NDArray, NDArray):
        pass

    def full(self, value, count=1):
        return self.from_numpy(np.full((self.dim, count), value))

    def zeros(self, count: int = 1) -> VectorArray:
        return self.full(0., count)

    def ones(self, count: int = 1) -> VectorArray:
        return self.full(1., count)


class Norm(ABC):

    @abstractmethod
    def __call__(self, U: VectorArray) -> NDArray:
        pass


class NormedSpace(VectorSpace):
    norm: Norm


class NormedSpaceWithBasis(NormedSpace, VectorSpaceWithBasis):
    pass


class SesquilinearForm(ABC):
    left_space: VectorSpace
    right_space: VectorSpace

    @abstractmethod
    def __call__(self, left: VectorArray, right: VectorArray) -> NDArray:
        pass

    @abstractmethod
    def as_operator(self) -> 'LinearOperator':
        pass


class InnerProduct(SesquilinearForm):

    def induced_norm(self) -> Norm:
        raise NotImplementedError


class HilbertSpace(NormedSpace):
    inner_product: InnerProduct

    def riesz(self, U: VectorArray) -> VectorArray:
        self.inner_product.as_operator().apply(U)


class HilbertSpaceWithBasis(HilbertSpace, VectorSpaceWithBasis):
    pass


class EuclideanSpace(HilbertSpaceWithBasis):
    """`inner_product` is compatible with `to_numpy`.

    For complex spaces, `riesz` is not the identity but complex conjugation!
    """

    @property
    def antidual_space(self) -> 'EuclideanSpace':
        return self


class Operator(ABC):
    source_space: VectorSpace
    range_space: VectorSpace

    @abstractmethod
    def apply(self, U: VectorArray) -> VectorArray:
        pass

    def __add__(self, other: 'Operator') -> 'Operator':
        raise NotImplementedError

    def __mul__(self, other: Scalar) -> 'Operator':
        raise NotImplementedError

    def __matmul__(self, other: 'Operator') -> 'Operator':
        raise NotImplementedError


class LinearOperator(Operator):

    @abstractmethod
    def apply_transpose(self, V: VectorArray) -> VectorArray:
        pass


class HSLinearOperator(LinearOperator):
    source_space: HilbertSpace
    range_space: HilbertSpace

    def apply_adjoint(self, V: VectorArray) -> VectorArray:
        return self.source_space.dual_space.riesz(self.apply_transpose(self.range.riesz(V)))


class LinearSolver(ABC):

    @abstractmethod
    def solve(self, lhs: LinearOperator, rhs: VectorArray) -> VectorArray:
        pass
