import numpy as np
import time
import datetime
from optimizaton import update_embedding_dbd,update_embedding,spdr_grad
from preprocessing import normalize_scanpy
def spdr(
        X,
        triplets,
        weights,
        Yinit,
        n_dims=2,
        lr=1000.0,
        n_iters=5000,
        opt_method="sd",
        verbose=1,
        return_seq=False,
        return_seq_step=10
):

    #diff Y
    opt_method_dict = {"sd": 0, "momentum": 1, "dbd": 2}
    if verbose:
        t = time.time()
    n, dim = X.shape
    # assert n_inliers < n - 1, "n_inliers must be less than (number of data points - 1)."
    if verbose:
        print("running TriMap on %d points with dimension %d" % (n, dim))
    pca_solution = False
    """

    if Yinit is None or Yinit is "pca":
        if pca_solution:
            Y = 0.01 * X[:, :n_dims]
        else:
            Y = 0.01 * PCA(n_components=n_dims).fit_transform(X).astype(np.float32)
    elif Yinit is "random":
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    else:
        Y = Yinit.astype(np.float32)
     """
    Y = Yinit.astype(np.float32)
    Y_all={}
    if return_seq:
        Y_all[0]=Yinit
        #Y_all = np.zeros((n, n_dims, int(n_iters / return_seq_step + 1)))
        #Y_all[:, :, 0] = Yinit

    C = np.inf
    tol = 1e-7
    n_triplets = float(triplets.shape[0])
    lr = lr * n / n_triplets
    if verbose:
        print("running TriMap with " + opt_method)
    vel = np.zeros_like(Y, dtype=np.float32)
    if opt_method_dict[opt_method] == 2:
        gain = np.ones_like(Y, dtype=np.float32)
    return_seq_step_old=10
    for itr in range(n_iters):
        old_C = C
        if opt_method_dict[opt_method] == 0:
            grad = spdr_grad(Y, int(5), int(5), triplets, weights)
        else:
            if itr > 250:
                gamma = 0.5
            else:
                gamma = 0.3
            grad = spdr_grad(Y + gamma * vel, int(5), int(5), triplets, weights)
        C = grad[-1, 0]
        n_viol = grad[-1, 1]

        # update Y
        if opt_method_dict[opt_method] < 2:
            update_embedding(Y, grad, vel, lr, itr, opt_method_dict[opt_method])
        else:
            update_embedding_dbd(Y, grad, vel, gain, lr, itr)

        # update the learning rate
        if opt_method_dict[opt_method] < 2:
            if old_C > C + tol:
                lr = lr * 1.01
            else:
                lr = lr * 0.9

        if return_seq:
            if itr< max(15,return_seq_step) &(itr+1)%return_seq_step != 0:
                Y_all[itr] = Y.copy()
            else:
                if (itr + 1) % return_seq_step == 0:
                    Y_all[itr] = Y.copy()
            return_seq_step_old=return_seq_step
            if (itr+1)%100==0:
                #increasing step during traing(every 100 epoch)
                return_seq_step=return_seq_step+5
        if verbose:
            if (itr + 1) % 100 == 0:
                print(
                    "Iteration: %4d, Loss: %3.3f, Violated triplets: %0.4f"
                    % (itr + 1, C, n_viol / n_triplets * 100.0)
                )
    if verbose:
        elapsed = str(datetime.timedelta(seconds=time.time() - t))
        print("Elapsed time: %s" % (elapsed))
    if return_seq:
        if n_iters % return_seq_step_old ==0:
            #right last one
            return Y_all
        else:
            Y_all[n_iters-1]=Y.copy()
            return Y_all
    else:
        Y_all={}
        Y_all[0]=Y
        return Y_all
