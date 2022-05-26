from math import gamma
import numpy as np
import torch

from dogleg import single_dogleg, double_dogleg, cal_delta_qk


def odr_trust_region_method(func_inst, x0, error0, eps=1e-8, n_epochs=100, max_delta=10, dogleg_func=single_dogleg, verbose=False):
    # Trust Region method
    # x0: initial value of x
    # error0: initial error term
    # func_inst: the function instance to support the calculation of r(x) & J(x)
    # eps: epsilon for the variation of function values & gradients to stop iteration
    # large_residual: if True, also consider the Sk term in Newton equation using the DGW formula
    xk, errork, delta = x0, error0, max_delta

    # Set up according to x0
    rk = func_inst(xk, errork)
    fk = torch.sum(rk ** 2) / 2
    Jk = func_inst.jacobian(xk, errork)
    gk = np.dot(Jk.T, rk)
    last_rk = 100000000
    m_val, n_val = int(Jk.shape[0] / 2), int(Jk.shape[1] - Jk.shape[0] / 2)
    assert(rk.shape[0] == 2 * m_val), (rk.shape, m_val, n_val)

    for epoch in range(n_epochs):
        # Stop conditions
        if np.linalg.norm(rk - last_rk) < eps or np.linalg.norm(gk) < eps:
            break
        
        # Solve trust-region sub-problem to get dk
        dk = dogleg_func(delta, Jk, gk)
        # dk contains the update on both xk & errork
        dk_x, dk_error = dk[:n_val], dk[n_val:]
        new_xk = xk + dk_x    # trust-region need no ak
        new_errork = errork + dk_error

        # calculate rk+1 & fk+1
        new_rk = func_inst(new_xk, new_errork)
        new_fk = torch.sum(new_rk ** 2) / 2

        # Calculate gamma_k = (fk+1 - fk) / (qk+1 - qk)
        delta_f = fk - new_fk
        delta_q = cal_delta_qk(dk, Jk, rk)
        gamma_k = delta_f / delta_q

        # Update trust region
        if gamma_k < 0.25:
            delta = delta / 4
        elif gamma_k > 0.75 and np.linalg.norm(dk) >= 0.95 * delta:
            delta = np.min((2 * delta, max_delta))
        
        # update xk only for downward move
        if gamma_k > 0:
            last_rk = rk
            xk = new_xk
            rk = new_rk
            errork = new_errork
            # necessary for Jk & gk calculation
            Jk = func_inst.jacobian(new_xk, new_errork)
            gk = np.dot(Jk.T, rk)   # gk = Jk * rk

        if verbose:
            fval = torch.sum(rk ** 2) / 2
            print('[{}] rk={:.8f}, |gk|={:.8f}'.format(epoch, fval.item(), np.linalg.norm(gk)))
    return xk, errork, epoch

