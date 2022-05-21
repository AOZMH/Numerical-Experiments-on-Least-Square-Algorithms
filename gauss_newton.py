import numpy as np
import torch


def gauss_newton_method(func_inst, x0, line_searcher, eps=1e-8, n_epochs=100, large_residual=False, verbose=False):
    # Gauss Newton method
    # x0: initial value of x
    # func_inst: the function instance to support the calculation of r(x) & J(x)
    # eps: epsilon for the variation of function values & gradients to stop iteration
    # large_residual: if True, also consider the Sk term in Newton equation using the DGW formula
    xk, last_xk, last_rk, last_Jk = x0, None, 1000000000, None
    if large_residual:
        Bk = np.eye(len(xk))

    for epoch in range(n_epochs):
        # calculate gk & rk
        rk = func_inst(xk)
        Jk = func_inst.jacobian(xk)
        gk = np.dot(Jk.T, rk)   # gk = Jk * rk
        if np.linalg.norm(rk - last_rk) < eps or np.linalg.norm(gk) < eps:
            break

        # dk = -(JkJk + Sk)^(-1)*(Jkrk)
        Gk = np.matmul(Jk.T, Jk)
        if large_residual:
            if epoch > 0:
                sk = xk - last_xk
                # TODO 这里可以记录last_gk来优化，可以验证正确性
                yk = np.dot(Jk.T, rk) - np.dot(last_Jk.T, last_rk)
                yk_hat = np.dot((Jk - last_Jk).T, rk)
                Bk = update_Bk(Bk, sk, yk, yk_hat)
            Gk += Bk
        dk = -np.dot(np.linalg.inv(Gk), gk)

        # line search for alpha_k
        partial_func = func_inst.get_partial_alpha(xk, dk)
        args = {
            'n_func_calls': 20,
            'gk': gk,
            'dk': dk,
        }
        ak = line_searcher.pipeline(partial_func, args)

        # update xk
        last_xk = xk
        last_rk = rk
        last_Jk = Jk
        xk = xk + ak * dk
        if verbose:
            fval = torch.sum(rk ** 2) / 2
            print('[{}] rk={:.8f}, |gk|={:.8f}'.format(epoch, fval.item(), np.linalg.norm(gk)))
    return xk, epoch


def update_Bk(Bk, sk, yk, yk_hat):
    # Updating function for Bk, the simulation of Sk
    yksk = np.dot(yk, sk)
    yk_hat_Bksk = yk_hat - np.dot(Bk, sk)

    upper1 = np.matmul(yk_hat_Bksk.reshape(-1, 1), yk.reshape(1, -1)) + np.matmul(yk.reshape(-1, 1), yk_hat_Bksk.reshape(1, -1))
    upper2 = np.dot(yk_hat_Bksk, sk) * np.matmul(yk.reshape(-1, 1), yk.reshape(1, -1))
    
    return Bk + upper1 / yksk - upper2 / (yksk ** 2)
