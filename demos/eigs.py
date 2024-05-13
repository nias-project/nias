#!/usr/bin/env python3
# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import logging

import numpy as np
from typer import Argument, run

from nias.base.vectorarrays import HilbertSpaceWithBasisFromProductOperator
from nias.bindings.numpy.operators import NumpyMatrixOperator
from nias.bindings.numpy.vectorarrays import NumpyVectorSpace
from nias.linalg.eigs import eigs
from nias.linalg.gram_schmidt import gram_schmidt

logging.basicConfig(level=logging.INFO)


def main(
    dim: int = Argument(..., min=1, help='Dimension of vector Sapce'),
):
    product_op = NumpyMatrixOperator(np.eye(dim))  # TODO: Euclidean spaces
    space = HilbertSpaceWithBasisFromProductOperator(NumpyVectorSpace(dim), product_op)

    S = np.linspace(1., 10., dim)
    U = space.random(dim)
    gram_schmidt(U, space.inner_product, copy=False)
    print(space.inner_product(U, U))
    U = space.to_numpy(U).T

    matrix = U @ np.diag(S) @ U.T
    print(np.linalg.eigh(matrix))

    op = NumpyMatrixOperator(matrix, space, space)

    evals, _ = eigs(op)
    print('Largest eigenvalues:', evals)

    evals, _ = eigs(op, sigma=0.)
    print('Smallest eigenvalues:', evals)


if __name__ == '__main__':
    run(main)
