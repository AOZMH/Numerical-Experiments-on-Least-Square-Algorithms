import time
import numpy as np
import torch

from osborne import osborne
from line_search import fib_searcher, gll_searcher
from gauss_newton import gauss_newton_method
from dogleg import trust_region_method, single_dogleg, double_dogleg


def get_osborne_instance(num_data=65):
    # Initialize a new instance for counting
    y_data = [1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.5, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.71, 0.729, 0.72, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054]
    t_data = [ix / 10 for ix in range(len(y_data))]

    osborne_inst = osborne(t_data, y_data)
    return osborne_inst


def gauss_newton_test(x0, large_residual, line_searcher, trial_name, num_data=65):
    # test gauss newton methods
    func_inst = get_osborne_instance(num_data)
    t0 = time.time()
    x_star, epochs = gauss_newton_method(func_inst, x0, line_searcher, eps=1e-6, large_residual=large_residual, verbose=True)
    
    # output statistics
    elapsed_time = time.time() - t0
    eval_info = func_inst.get_eval_infos()   # NOTE this must be called before the final evaluation of x*
    f_star = func_inst(x_star, reduce=True)
    g_star = func_inst.g_func(x_star)
    g_norm = np.linalg.norm(g_star)

    print('{}\tfx* = {:.6f}\t|gx*| = {:.8f}\ttime = {:.6f}\tepochs = {}\t{}'.format(trial_name, f_star.item(), g_norm, elapsed_time, epochs, eval_info))


def dogleg_test(x0, dogleg_func, trial_name, num_data=65):
    # test gauss newton methods
    func_inst = get_osborne_instance(num_data)
    t0 = time.time()
    x_star, epochs = trust_region_method(func_inst, x0, eps=1e-6, dogleg_func=dogleg_func, verbose=True)
    
    # output statistics
    elapsed_time = time.time() - t0
    eval_info = func_inst.get_eval_infos()   # NOTE this must be called before the final evaluation of x*
    f_star = func_inst(x_star, reduce=True)
    g_star = func_inst.g_func(x_star)
    g_norm = np.linalg.norm(g_star)

    print('{}\tfx* = {:.6f}\t|gx*| = {:.8f}\ttime = {:.6f}\tepochs = {}\t{}'.format(trial_name, f_star.item(), g_norm, elapsed_time, epochs, eval_info))


def main_gauss_newton(noise=None):
    fib_search_inst = fib_searcher()
    x_scales = [8, 16, 32, 64, 128]
    x_scales = [65]
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        # Initial value provided by Osborne
        x0 = torch.tensor([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5])
        if noise is not None:
           x0 = x0 + noise
        gauss_newton_test(x0, False, fib_search_inst, 'Gauss Newton')
        gauss_newton_test(x0, True, fib_search_inst, 'Large Residual GN')


def main_dogleg(noise=None):
    x_scales = [65]
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        # Initial value provided by Osborne
        x0 = torch.tensor([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5])
        if noise is not None:
           x0 = x0 + noise
        dogleg_test(x0, single_dogleg, 'Single Dogleg')
        dogleg_test(x0, double_dogleg, 'Double Dogleg')


if __name__ == '__main__':
    noise = torch.randn(11) * 0.4
    #noise = None
    main_dogleg(noise)
    main_gauss_newton(noise)
