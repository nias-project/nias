# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from logging import getLogger

import numpy as np
import scipy.linalg as spla

from nias.interfaces import HilbertSpace, VectorArray
from nias.linalg.gram_schmidt import gram_schmidt


def method_of_snapshots(A: VectorArray, range_space: HilbertSpace,
                        modes=None, rtol=1e-7, atol=0., l2_err=0.):
    assert A in range_space

    if range_space.dim == 0 or len(A) == 0:
        return range_space.empty(), np.array([]), np.zeros((0, len(A)))

    logger = getLogger(__name__ + '.method_of_snapshots')

    logger.info(f'Computing Gramian ({len(A)} vectors) ...')
    B = range_space.inner_product(A, A)

    logger.info('Computing eigenvalue decomposition ...')
    eigvals = (None
               if modes is None or l2_err > 0.
               else (max(len(B) - modes, 0), len(B) - 1))

    evals, V = spla.eigh(B, overwrite_a=True, subset_by_index=eigvals)
    evals = evals[::-1]
    V = V.T[::-1, :]

    tol = max(rtol ** 2 * evals[0], atol ** 2)
    above_tol = np.where(evals >= tol)[0]
    if len(above_tol) == 0:
        return range_space.empty(), np.array([]), np.zeros((0, len(A)))
    last_above_tol = above_tol[-1]

    errs = np.concatenate((np.cumsum(evals[::-1])[::-1], [0.]))
    below_err = np.where(errs <= l2_err**2)[0]
    first_below_err = below_err[0]

    selected_modes = min(first_below_err, last_above_tol + 1)
    if modes is not None:
        selected_modes = min(selected_modes, modes)

    if selected_modes > range_space.dim:
        logger.warning('Number of computed singular vectors larger than array dimension! Truncating ...')
        selected_modes = range_space.dim

    s = np.sqrt(evals[:selected_modes])
    V = V[:selected_modes]
    Vh = V.conj()

    logger.info(f'Computing left-singular vectors ({len(V)} vectors) ...')
    U = A.lincomb(V / s[:, np.newaxis])

    return U, s, Vh


def qr_svd(A: VectorArray, range_space: HilbertSpace,
           modes=None, rtol=4e-8, atol=0., l2_err=0.):
    # TODO: add a way to pass gram_schmidt options
    assert A in range_space

    if range_space.dim == 0 or len(A) == 0:
        return range_space.empty(), np.array([]), np.zeros((0, len(A)))

    logger = getLogger(__name__ + '.qr_svd')

    logger.info('Computing QR decomposition ...')
    Q, R = gram_schmidt(A, range_space.inner_product, return_R=True, check=False)

    logger.info('Computing SVD of R ...')
    U2, s, Vh = spla.svd(R, lapack_driver='gesvd')

    logger.info('Choosing the number of modes ...')
    tol = max(rtol * s[0], atol)
    above_tol = np.where(s >= tol)[0]
    if len(above_tol) == 0:
        return range_space.empty(), np.array([]), np.zeros((0, len(A)))
    last_above_tol = above_tol[-1]

    errs = np.concatenate((np.cumsum(s[::-1] ** 2)[::-1], [0.]))
    below_err = np.where(errs <= l2_err**2)[0]
    first_below_err = below_err[0]

    selected_modes = min(first_below_err, last_above_tol + 1)
    if modes is not None:
        selected_modes = min(selected_modes, modes)

    U2 = U2[:, :selected_modes]
    s = s[:selected_modes]
    Vh = Vh[:selected_modes]

    logger.info(f'Computing left singular vectors ({selected_modes} modes) ...')
    U = Q.lincomb(U2.T)

    return U, s, Vh
