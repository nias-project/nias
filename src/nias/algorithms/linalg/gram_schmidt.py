# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from logging import getLogger

import numpy as np

from nias.exceptions import AccuracyError
from nias.interfaces import InnerProduct, VectorArray


def gram_schmidt(A: VectorArray, inner_product: InnerProduct,
                 return_R=False, atol=1e-13, rtol=1e-13, offset=0,
                 reiterate=True, reiteration_threshold=9e-1, check=True, check_tol=1e-3,
                 copy=True):

    logger = getLogger(__name__)

    if copy:
        A = A.copy()

    norm = inner_product.induced_norm()

    # main loop
    R = np.eye(len(A), dtype=A.scalar_type)
    remove = []  # indices of to be removed vectors
    for i in range(offset, len(A)):
        # first calculate norm
        initial_norm = norm(A[i])[0]

        if initial_norm <= atol:
            logger.info(f'Removing vector {i} of norm {initial_norm}')
            remove.append(i)
            continue

        if i == 0:
            A[0].scal(1 / initial_norm)
            R[i, i] = initial_norm
        else:
            current_norm = initial_norm
            # If reiterate is True, reiterate as long as the norm of the vector changes
            # strongly during orthogonalization (due to Andreas Buhr).
            while True:
                # orthogonalize to all vectors left
                for j in range(i):
                    if j in remove:
                        continue
                    p = inner_product(A[j], A[i])[0, 0]
                    A[i].axpy(-p, A[j])
                    R[j, i] += p

                # calculate new norm
                old_norm, current_norm = current_norm, norm(A[i])[0]

                # remove vector if it got too small
                if current_norm <= rtol * initial_norm:
                    logger.info(f'Removing linearly dependent vector {i}')
                    remove.append(i)
                    break

                # check if reorthogonalization should be done
                if reiterate and current_norm < reiteration_threshold * old_norm:
                    logger.info(f'Orthonormalizing vector {i} again')
                else:
                    A[i].scal(1 / current_norm)
                    R[i, i] = current_norm
                    break

    if remove:
        del A[remove]
        R = np.delete(R, remove, axis=0)

    if check:
        error_matrix = inner_product(A[offset:len(A)], A)
        error_matrix[:len(A) - offset, offset:len(A)] -= np.eye(len(A) - offset)
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError(f'result not orthogonal (max err={err})')

    if return_R:
        return A, R
    else:
        return A
