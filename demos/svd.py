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
from nias.linalg.gram_schmidt import gram_schmidt
from nias.linalg.svd import method_of_snapshots, qr_svd

logging.basicConfig(level=logging.INFO)


def main(
    dim: int = Argument(..., min=1, help='Dimension of vector Sapce'),
    num_vecs: int = Argument(..., min=1, help='Number of vectors to orthonormalize'),
):
    range_product_op = NumpyMatrixOperator(np.eye(dim))  # TODO: Euclidean spaces
    range_space = HilbertSpaceWithBasisFromProductOperator(NumpyVectorSpace(dim), range_product_op)

    source_product_op = NumpyMatrixOperator(np.eye(num_vecs))
    source_space = HilbertSpaceWithBasisFromProductOperator(NumpyVectorSpace(num_vecs), source_product_op)

    S = np.linspace(1., 10., num_vecs)
    U = range_space.random(num_vecs)
    gram_schmidt(U, range_space.inner_product, copy=False)
    U = range_space.to_numpy(U).T
    V = source_space.random(num_vecs)
    gram_schmidt(V, source_space.inner_product, copy=False)
    V = source_space.to_numpy(V)

    matrix = U @ np.diag(S) @ V
    print(np.linalg.svd(matrix).S)

    array = range_space.from_numpy(matrix.T)

    svals = method_of_snapshots(array, range_space)[1]
    print('method of snapshots:', svals)

    svals = qr_svd(array, range_space)[1]
    print('QR svd:', svals)


if __name__ == '__main__':
    run(main)
