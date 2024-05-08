# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from logging import getLogger

import numpy as np
import scipy.linalg as spla

from nias.base.linear_solvers import default_factory
from nias.base.operators import identity, inverse
from nias.interfaces import HSLinearOperator, LinearSolverFactory
from nias.linalg.gram_schmidt import gram_schmidt


def eigs(A: HSLinearOperator, E: HSLinearOperator | None = None,
         k=3, sigma=None, which='LM', b=None, l=None, maxiter=1000, tol=1e-13,
         imag_tol=1e-12, complex_pair_tol=1e-12, complex_evp=False, left_evp=False,
         solver_factory: LinearSolverFactory = default_factory):
    logger = getLogger(__name__)

    assert A.source_space == A.range_space

    if E is None:
        E = identity(A.source_space)
    assert E.source_space == E.range_space
    assert E.source_space == A.source_space

    if b is None:
        b = A.source_space.random()
    assert b in A.source_space

    n = A.source_space.dim

    if l is None:
        l_min = 20
        l = max(2 * k + 1, l_min)
        if n is not None:
            l = min(n - 1, l)

    assert n is None or k < n
    assert l > k

    if sigma is None:
        if left_evp:
            raise NotImplementedError
            # Aop = inverse(E).H @ A.H
        else:
            Aop = inverse(E, context='eigs_E_inverse') @ A
    else:
        if sigma.imag != 0:
            complex_evp = True
        else:
            sigma = sigma.real

        if left_evp:
            raise NotImplementedError
            # Aop = inverse(A - sigma * E).H @ E.H
        else:
            Aop = inverse(A - sigma * E, context='eigs_shift_invert', solver_factory=solver_factory) @ E

    V, H, f = _arnoldi(Aop, k, b, complex_evp)

    k0 = k
    i = 0

    while True:
        i += 1

        V, H, f = _extend_arnoldi(Aop, V, H, f, l - k)

        ew, ev = spla.eig(H)

        # truncate small imaginary parts
        ew.imag[np.abs(ew.imag) / np.abs(ew) < imag_tol] = 0

        if which == 'LM':
            idx = np.argsort(-np.abs(ew))
        elif which == 'SM':
            idx = np.argsort(np.abs(ew))
        elif which == 'LR':
            idx = np.argsort(-ew.real)
        elif which == 'SR':
            idx = np.argsort(ew.real)
        elif which == 'LI':
            idx = np.argsort(-np.abs(ew.imag))
        elif which == 'SI':
            idx = np.argsort(np.abs(ew.imag))

        k = k0
        ews = ew[idx]
        evs = ev[:, idx]

        rres = A.source_space.norm(f)[0] * np.abs(evs[l - 1]) / np.abs(ews)

        # increase k by one in order to keep complex conjugate pairs together
        if not complex_evp and ews[k - 1].imag != 0 and ews[k - 1].imag + ews[k].imag < complex_pair_tol:
            k += 1

        logger.info(f'Maximum of relative Ritz estimates at step {i}: {rres[:k].max():.5e}')

        if np.all(rres[:k] <= tol) or i >= maxiter:
            break

        # increase k in order to prevent stagnation
        k = min(l - 1, k + min(np.count_nonzero(rres[:k] <= tol), (l - k) // 2))

        # sort shifts for QR iteration based on their residual
        shifts = ews[k:l]
        srres = rres[k:l]
        idx = np.argsort(-srres)
        srres = srres[idx]
        shifts = shifts[idx]

        # don't use converged unwanted Ritz values as shifts
        shifts = shifts[srres != 0]
        k += np.count_nonzero(srres == 0)
        if not complex_evp and shifts[0].imag != 0 and shifts[0].imag + ews[1].imag >= complex_pair_tol:
            shifts = shifts[1:]
            k += 1

        H, Qs = _qr_iteration(H, shifts, complex_evp=complex_evp)

        V = V.lincomb(Qs.T)
        f = V[k] * H[k, k - 1] + f * Qs[l - 1, k - 1]
        V = V[:k]
        H = H[:k, :k]

    if sigma is not None:
        ews = 1 / ews + sigma

    return ews[:k0], V.lincomb(evs[:, :k0].T)


def _arnoldi(A, l, b, complex_evp):
    """Compute an Arnoldi factorization."""
    v = b * (1 / A.source_space.norm(b)[0])

    H = np.zeros((l, l), dtype=np.complex_ if complex_evp else np.float_)
    V = A.source_space.empty(size_hint=l)

    V.append(v)

    for i in range(l):
        v = A.apply(v)
        V.append(v)

        _, R = gram_schmidt(V, A.source_space.inner_product,
                            return_R=True, atol=0, rtol=0, offset=len(V) - 1, copy=False)
        H[:i + 2, i] = R[:l, i + 1]
        v = V[-1]

    return V[:l], H, v * R[l, l]


def _extend_arnoldi(A, V, H, f, p):
    """Extend an existing Arnoldi factorization."""
    k = len(V)

    res = A.source_space.norm(f)[0]
    # the explicit "constant" mode is needed for numpy 1.16
    # mode only gained a default value with numpy 1.17
    H = np.pad(H, ((0, p), (0, p)), mode='constant')
    H[k, k - 1] = res
    v = f * (1 / res)
    V = V.copy()
    V.append(v)

    for i in range(k, k + p):
        v = A.apply(v)
        V.append(v)
        _, R = gram_schmidt(V, A.source_space.inner_product,
                            return_R=True, atol=0, rtol=0, offset=len(V) - 1, copy=False)
        H[:i + 2, i] = R[:k + p, i + 1]

        v = V[-1]

    return V[:k + p], H, v * R[k + p, k + p]


def _qr_iteration(H, shifts, complex_evp=False):
    """Perform the QR iteration."""
    Qs = np.eye(len(H))

    i = 0
    while i < len(shifts) - 1:
        s = shifts[i]
        if not complex_evp and shifts[i].imag != 0:
            Q, _ = spla.qr(H @ H - 2 * s.real * H + np.abs(s)**2 * np.eye(len(H)))
            i += 2
        else:
            Q, _ = spla.qr(H - s * np.eye(len(H)))
            i += 1
        Qs = Qs @ Q
        H = Q.T @ H @ Q

    return H, Qs
