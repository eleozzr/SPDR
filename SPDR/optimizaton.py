import numba
import numpy as np

@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4,i4,i4)", parallel=True, nogil=True)
def update_embedding(Y, grad, vel, lr, iter_num, opt_method):
    n, dim = Y.shape
    if opt_method == 0:  # sd
        for i in numba.prange(n):
            for d in numba.prange(dim):
                Y[i][d] -= lr * grad[i][d]
    elif opt_method == 1:  # momentum
        if iter_num > 250:
            gamma = 0.5
        else:
            gamma = 0.3
        for i in numba.prange(n):
            for d in numba.prange(dim):
                vel[i][d] = gamma * vel[i][d] - lr * grad[i][d]  # - 1e-5 * Y[i,d]
                Y[i][d] += vel[i][d]


@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,i4)", parallel=True, nogil=True)
def update_embedding_dbd(Y, grad, vel, gain, lr, iter_num):
    n, dim = Y.shape
    if iter_num > 250:
        gamma = 0.8  # moment parameter
    else:
        gamma = 0.5
    min_gain = 0.01
    for i in numba.prange(n):
        for d in numba.prange(dim):
            gain[i][d] = (
                (gain[i][d] + 0.2)
                if (np.sign(vel[i][d]) != np.sign(grad[i][d]))
                else np.maximum(gain[i][d] * 0.8, min_gain)
            )
            vel[i][d] = gamma * vel[i][d] - lr * gain[i][d] * grad[i][d]
            Y[i][d] += vel[i][d]

@numba.njit("f4[:,:](f4[:,:],i4,i4,i4[:,:],f4[:])", parallel=True, nogil=True)
def spdr_grad(Y, n_inliers, n_outliers, triplets, weights):
    n, dim = Y.shape
    n_triplets = triplets.shape[0]
    grad = np.zeros((n, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    y_ik = np.empty(dim, dtype=np.float32)
    n_viol = 0.0
    loss = 0.0
    #n_knn_triplets = n * n_inliers * n_outliers
    n_inliers=5
    n_outliers = 5
    n_knn_triplets=n_triplets
    for t in range(n_triplets):
        i = triplets[t, 0]
        j = triplets[t, 1]
        k = triplets[t, 2]
        if (t % n_outliers) == 0 or (
            t >= n_knn_triplets
        ):  # update y_ij, y_ik, d_ij, d_ik
            d_ij = 1.0
            d_ik = 1.0
            for d in range(dim):
                y_ij[d] = Y[i, d] - Y[j, d]
                y_ik[d] = Y[i, d] - Y[k, d]
                d_ij += y_ij[d] ** 2
                d_ik += y_ik[d] ** 2
        else:  # update y_ik and d_ik only
            d_ik = 1.0
            for d in range(dim):
                y_ik[d] = Y[i, d] - Y[k, d]
                d_ik += y_ik[d] ** 2
        if d_ij > d_ik:
            n_viol += 1.0
        loss += weights[t] * 1.0 / (1.0 + d_ik / d_ij)
        w = weights[t] / (d_ij + d_ik) ** 2
        for d in range(dim):
            gs = y_ij[d] * d_ik * w
            go = y_ik[d] * d_ij * w
            grad[i, d] += gs - go
            grad[j, d] -= gs
            grad[k, d] += go
    last = np.zeros((1, dim), dtype=np.float32)
    last[0] = loss
    last[1] = n_viol
    return np.vstack((grad, last))
