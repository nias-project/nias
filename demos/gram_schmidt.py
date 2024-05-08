# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import logging

import numpy as np

from nias.bindings.numpy.operators import NumpyMatrixOperator
from nias.bindings.numpy.vectorarrays import NumpyVectorSpace
from nias.linalg.gram_schmidt import gram_schmidt

logging.basicConfig(level=logging.INFO)

product_op = NumpyMatrixOperator(np.eye(10))
product = product_op.as_inner_product()
space = NumpyVectorSpace(10)

U = space.random(5)
print(product(U, U))

gram_schmidt(U, product, copy=False)
print(product(U, U))
