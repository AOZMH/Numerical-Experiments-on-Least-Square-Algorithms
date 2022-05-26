from math import gamma
import numpy as np
import torch


def cal_delta_qk(dk, Jk, rk):
    # Delta of the order-2 taylor approximiztion of f(x)
    # Returns q(xk) - q(xk + dk)
    Jkdk = np.dot(Jk, dk)
    val1 = np.dot(Jkdk, Jkdk) / 2
    val2 = np.dot(rk, Jkdk)
    return - val1 - val2


def get_beta(dk_gn, gn_norm, dk_sd_ak, sd_norm, delta):
    # Calculate bk s.t. ||bk*dk_gn + (1-bk)*dk_sd_ak|| = delta
    prod = np.dot(dk_gn, dk_sd_ak)
    coeff = [
        (gn_norm ** 2) + (sd_norm ** 2) - 2 * prod,
        -2 * (sd_norm ** 2) + 2 * prod,
        (sd_norm ** 2) - (delta ** 2),
    ]
    beta1, beta2 = np.roots(coeff)
    if 0 <= beta1 <= 1:
        return beta1
    elif 0 <= beta2 <= 1:
        #assert(0 <= beta2 <= 1), (beta1, beta2, coeff)
        return beta2
    # If either, might be caused in the double-dogleg
    # when eta is too small s.t. ||dk_gn_tilde|| < delta
    if beta1 > 0:
        return beta1
    else:
        return beta2


def single_dogleg(delta, Jk, gk):
    # Single dogleg method
    Gk = np.matmul(Jk.T, Jk)
    dk_gn = -np.dot(np.linalg.inv(Gk), gk)
    dk_sd = -gk
    ak = np.dot(gk, gk) / (np.linalg.norm(np.dot(Jk, gk)) ** 2)
    dk_sd_ak = ak * dk_sd
    
    gn_norm = np.linalg.norm(dk_gn)
    sd_norm = np.linalg.norm(dk_sd_ak)

    if gn_norm <= delta:
        return dk_gn
    if sd_norm >= delta:
        return (delta / sd_norm) * dk_sd_ak
    beta = get_beta(dk_gn, gn_norm, dk_sd_ak, sd_norm, delta)
    dk = (1 - beta) * dk_sd_ak + beta * dk_gn
    #print(delta, np.linalg.norm(dk))
    return dk


def get_gamma_double_dogleg(gk, Jk, Gk_inv):
    # Calculate the gamma in double dogleg
    Jkgk = np.dot(Jk, gk)
    lower = np.dot(Jkgk, Jkgk)
    # Note that Gk_inv = (Jk^t*Jk)^-1
    lower *= np.dot(gk, np.dot(Gk_inv, gk))
    upper = np.dot(gk, gk) ** 2
    return upper / lower


def double_dogleg(delta, Jk, gk):
    # Double dogleg method
    Gk = np.matmul(Jk.T, Jk)
    Gk_inv = np.linalg.inv(Gk)
    dk_gn = -np.dot(Gk_inv, gk)
    dk_sd = -gk
    ak = np.dot(gk, gk) / (np.linalg.norm(np.dot(Jk, gk)) ** 2)
    dk_sd_ak = ak * dk_sd

    # Different from single dogleg, dk_gn_tilde is introduced
    gamma = get_gamma_double_dogleg(gk, Jk, Gk_inv)
    eta = 0.8 * gamma + 0.2     # Dennis & Mei's method
    dk_gn_tilde = eta * dk_gn
    
    gn_norm = np.linalg.norm(dk_gn)
    sd_norm = np.linalg.norm(dk_sd_ak)

    if gn_norm <= delta:
        return dk_gn
    if sd_norm >= delta:
        return (delta / sd_norm) * dk_sd_ak
    beta = get_beta(dk_gn_tilde, eta * gn_norm, dk_sd_ak, sd_norm, delta)
    dk = (1 - beta) * dk_sd_ak + beta * dk_gn_tilde
    #print(delta, np.linalg.norm(dk))
    return dk


def trust_region_method(func_inst, x0, eps=1e-8, n_epochs=100, max_delta=10, dogleg_func=single_dogleg, verbose=False):
    # Trust Region method
    # x0: initial value of x
    # func_inst: the function instance to support the calculation of r(x) & J(x)
    # eps: epsilon for the variation of function values & gradients to stop iteration
    # large_residual: if True, also consider the Sk term in Newton equation using the DGW formula
    xk, delta = x0, max_delta

    # Set up according to x0
    rk = func_inst(xk)
    fk = torch.sum(rk ** 2) / 2
    Jk = func_inst.jacobian(xk)
    gk = np.dot(Jk.T, rk)
    last_rk = 100000000

    for epoch in range(n_epochs):
        # Stop conditions
        if np.linalg.norm(rk - last_rk) < eps or np.linalg.norm(gk) < eps:
            break
        
        # Solve trust-region sub-problem to get dk
        dk = dogleg_func(delta, Jk, gk)
        new_xk = xk + dk    # trust-region need no ak

        # calculate rk+1 & fk+1
        new_rk = func_inst(new_xk)
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
            # necessary for Jk & gk calculation
            Jk = func_inst.jacobian(new_xk)
            gk = np.dot(Jk.T, rk)   # gk = Jk * rk

        if verbose:
            fval = torch.sum(rk ** 2) / 2
            print('[{}] rk={:.8f}, |gk|={:.8f}'.format(epoch, fval.item(), np.linalg.norm(gk)))
    return xk, epoch

