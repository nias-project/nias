# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from abc import ABC, abstractmethod, abstractproperty
from numbers import Number
from typing import Self, TypeAlias, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

Scalar: TypeAlias = int | float | complex | np.number
Indices: TypeAlias = None | int | slice | list[int] | NDArray


class VectorArray(ABC):
    """Abstract VectorArray interface.

    The scalar_type of a VectorArray never changes with one exception:
    if a real array is multiplied (in-place or not) with a complex scalar,
    its scalar_type changes to the corresponding complex type.
    Not all VectorArray implementations need to support complex numbers,
    however.

    append/axpy/dual_pairing/__add__/__iadd_/__setitem__ require
    both arrays to be compatible. In particular this means that both
    arrays need to be contained in a common (algebraic) vector space.
    Further, the scalar_types of both arrays need to agree
    (up to complexification).

    All complex VectorArrays are assumed to be equiped with a canonical
    conjugation operation, which yields the real and imaginary part of
    a vector in the array.
    """

    base: Union['VectorArray', None]
    is_view: bool
    scalar_type: type

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def is_compatible_array(self, other: 'VectorArray') -> bool:
        pass

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def append(self, other: 'VectorArray', remove_from_other: bool = False) -> None:
        pass

    @abstractmethod
    def __getitem__(self, ind: Indices) -> 'VectorArray':
        pass

    @abstractmethod
    def __setitem__(self, ind: Indices, other: 'VectorArray') -> None:
        pass

    @abstractmethod
    def __delitem__(self, ind: Indices):
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

    @property
    def is_complex(self) -> bool:
        return isinstance(self.scalar_type, np.complexfloating)

    @abstractproperty
    def real(self) -> 'VectorArray':
        pass

    @abstractproperty
    def imag(self) -> 'VectorArray':
        pass

    @abstractmethod
    def conj(self) -> 'VectorArray':
        pass

    def _dual_pairing(self, other: 'VectorArray') -> NDArray:
        return NotImplemented

    def __add__(self, other: 'VectorArray'):
        result = self.copy()
        result.axpy(1., other)
        return result

    def __iadd__(self, other):
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other):
        result = self.copy()
        result.axpy(-1., other)
        return result

    def __isub__(self, other):
        self.axpy(-1, other)
        return self

    def __mul__(self, other):
        result = self.copy()
        result.scal(other)
        return result

    __rmul__ = __mul__

    def __imul__(self, other):
        self.scal(other)
        return self

    def __neg__(self):
        result = self.copy()
        result.scal(-1.)
        return result

    # override NumPy binary operations and ufuncs
    __array_priority__ = 100.0
    __array_ufunc__ = None

    def check_ind(self, ind: Indices) -> bool:
        """Check if index is admissible.

        Check if `ind` is an admissible list of indices in the sense
        of the class documentation.
        """
        l = len(self)
        return (type(ind) is slice
                or isinstance(ind, Number) and -l <= ind < l
                or isinstance(ind, (list, np.ndarray)) and all(-l <= i < l for i in ind))

    def check_ind_unique(self, ind: Indices) -> bool:
        """Check if index is admissible and unique.

        Check if `ind` is an admissible list of non-repeated indices in
        the sense of the class documentation.
        """
        l = len(self)
        return (type(ind) is slice
                or isinstance(ind, Number) and -l <= ind < l
                or isinstance(ind, (list, np.ndarray))
                and len({i if i >= 0 else l+i for i in ind if -l <= i < l}) == len(ind))


def dual_pairing(left: VectorArray, right: VectorArray, pairwise: bool = False) -> NDArray:
    result = left._dual_pairing(right, pairwise)
    if result is NotImplemented:
        result = right._dual_pairing(left, pairwise)
    if result is NotImplemented:
        raise NotImplementedError('No dual pairing possible.')
    return result


class VectorSpace(ABC):
    dim: int | None

    @abstractmethod
    def empty(self, size_hint=0) -> VectorArray:
        pass

    @abstractmethod
    def zeros(self, count: int = 1) -> VectorArray:
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
    def __eq__(self, other: 'VectorSpace') -> bool:
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
