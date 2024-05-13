#!/usr/bin/env python3
# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import logging

import numpy as np
from typer import Argument, run

from nias.bindings.numpy.operators import NumpyMatrixOperator
from nias.bindings.numpy.vectorarrays import NumpyVectorSpace
from nias.linalg.gram_schmidt import gram_schmidt

logging.basicConfig(level=logging.INFO)


def main(
    dim: int = Argument(..., min=1, help='Dimension of vector Sapce'),
    num_vecs: int = Argument(..., min=1, help='Number of vectors to orthonormalize'),
):
    """Gram-Schmidt orthonormalization of a random VectorArray."""
    product_op = NumpyMatrixOperator(np.eye(dim))
    product = product_op.as_inner_product()
    space = NumpyVectorSpace(dim)

    U = space.random(num_vecs)
    print(product(U, U))

    gram_schmidt(U, product, copy=False)
    print(product(U, U))


if __name__ == '__main__':
    run(main)
