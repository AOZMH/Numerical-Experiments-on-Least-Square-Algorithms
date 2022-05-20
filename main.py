import time
from easom import easom_func
import numpy as np
import torch

from osborne import ackley_func
from zakharov import zakharov_func
from griewank import griewank_func
from rastrigin import rastrigin_func
from easom import easom_func
from line_search import fib_searcher, gll_searcher
from newton import damp_newton_method
from quasi_newton import quasi_newton_method, sr1_update_func, dfp_update_func, bfgs_update_func
from conjugate_grad import conjugate_gradient_two, fr_func, prp_func, conjugate_gradient_powell_three


def get_func_new_inst_by_name(name):
    if name == 'ackley':
        return ackley_func()
    elif name == 'zakharov':
        return zakharov_func()
    elif name == 'griewank':
        return griewank_func()
    elif name == 'rastrigin':
        return rastrigin_func()
    elif name == 'easom':
        return easom_func()


def newton_test(x0, use_cholesky_correction, line_searcher, trial_name, func_name='ackley'):
    # test newton methods
    func = get_func_new_inst_by_name(func_name)
    t0 = time.time()
    x_star, epochs = damp_newton_method(func, func.g_func, func.G_func, x0, line_searcher, eps=1e-8, use_cholesky_correction=use_cholesky_correction)
    
    # output statistics
    elapsed_time = time.time() - t0
    eval_info = func.get_eval_infos()   # NOTE this must be called before the final evaluation of x*
    f_star = func(x_star)
    g_star = func.g_func(x_star)
    g_norm = np.linalg.norm(g_star)

    print('{}\tfx* = {:.4f}\t|gx*| = {:.8f}\ttime = {:.6f}\tepochs = {}\t{}'.format(trial_name, f_star, g_norm, elapsed_time, epochs, eval_info))


def q_newton_test(x0, updater, line_searcher, trial_name, func_name='ackley', n_epochs=100):
    # test newton methods
    func = get_func_new_inst_by_name(func_name)
    H0 = np.eye(len(x0))
    t0 = time.time()
    x_star, epochs = quasi_newton_method(func, func.g_func, func.G_func, x0, H0, line_searcher, eps=1e-8, Hk_update_func=updater, lbfgs_m=6, verbose=False, n_epochs=n_epochs)

    # output statistics
    elapsed_time = time.time() - t0
    eval_info = func.get_eval_infos()   # NOTE this must be called before the final evaluation of x*
    f_star = func(x_star)
    g_star = func.g_func(x_star)
    g_norm = np.linalg.norm(g_star)

    print('{}\tfx* = {:.4f}\t|gx*| = {:.8f}\ttime = {:.6f}\tepochs = {}\t{}'.format(trial_name, f_star, g_norm, elapsed_time, epochs, eval_info))


def conjugate_gradient_test(x0, beta_func, line_searcher, trial_name, two_or_three='two', func_name='ackley'):
    # test conjugate gradient methods
    func = get_func_new_inst_by_name(func_name)
    t0 = time.time()
    if two_or_three == 'two':
        x_star, epochs = conjugate_gradient_two(func, func.g_func, x0, beta_func, line_searcher, eps=1e-8, n_epochs=300)
    elif two_or_three == 'three':
        x_star, epochs = conjugate_gradient_powell_three(func, func.g_func, x0, line_searcher, eps=1e-8, n_epochs=300)

    # output statistics
    elapsed_time = time.time() - t0
    eval_info = func.get_eval_infos()   # NOTE this must be called before the final evaluation of x*
    f_star = func(x_star)
    g_star = func.g_func(x_star)
    g_norm = np.linalg.norm(g_star)

    print('{}\tfx* = {:.4f}\t|gx*| = {:.8f}\ttime = {:.6f}\tepochs = {}\t{}'.format(trial_name, f_star, g_norm, elapsed_time, epochs, eval_info))


def main():
    fib_search_inst = fib_searcher()
    x_scales = [8, 16, 32, 64, 128]
    #x_scales = [32]
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        #x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale) * 2
        newton_test(x0, False, fib_search_inst, 'Damp Newton')
        newton_test(x0, True, fib_search_inst, 'Cholesky')
        q_newton_test(x0, sr1_update_func, fib_search_inst, 'Quasi SR1')
        q_newton_test(x0, dfp_update_func, fib_search_inst, 'Quasi DFP')
        q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS')


def main_gll_search():
    gll_search_inst = gll_searcher(gamma=1e-3, sigma=0.5, window=5, a0=1)
    fib_search_inst = fib_searcher()
    x_scales = [8, 16, 32, 64, 128]
    #x_scales = [32]
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        #x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale) * 2
        newton_test(x0, False, fib_search_inst, 'Damp Newton\tFIB')
        newton_test(x0, False, gll_search_inst, 'Damp Newton\tGLL')
        newton_test(x0, True, fib_search_inst, 'Cholesky\tFIB')
        newton_test(x0, True, gll_search_inst, 'Cholesky\tGLL')
        q_newton_test(x0, sr1_update_func, fib_search_inst, 'Quasi SR1\tFIB')
        q_newton_test(x0, sr1_update_func, gll_search_inst, 'Quasi SR1\tGLL')
        q_newton_test(x0, dfp_update_func, fib_search_inst, 'Quasi DFP\tFIB')
        q_newton_test(x0, dfp_update_func, gll_search_inst, 'Quasi DFP\tGLL')
        q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS\tFIB')
        q_newton_test(x0, bfgs_update_func, gll_search_inst, 'Quasi BFGS\tGLL')


def main_init_value():
    gll_search_inst = gll_searcher(gamma=1e-3, sigma=0.5, window=5, a0=1)
    fib_search_inst = fib_searcher()
    x_scales = [8, 16, 32, 64, 128]
    #x_scales = [32]
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        for init_scale_exp in range(6):
            cur_scale = 2 ** init_scale_exp
            x0 = torch.randn(x_scale) * cur_scale
            newton_test(x0, True, gll_search_inst, 'Cholesky\t' + str(cur_scale))
            #q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS\t' + str(cur_scale))


def main_hw2_ackley():
    fib_search_inst = gll_searcher(gamma=1e-3, sigma=0.5, window=5, a0=1)
    x_scales = [100, 200, 300, 400, 500]
    func_name = 'ackley'
    print('=========================== Ackley Function ==========================')
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        #x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale) * 2
        newton_test(x0, True, fib_search_inst, 'Cholesky', func_name)
        q_newton_test(x0, sr1_update_func, fib_search_inst, 'Quasi SR1', func_name)
        q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS', func_name)
        q_newton_test(x0, 'LBFGS', fib_search_inst, 'L-BFGS\t', func_name)
        conjugate_gradient_test(x0, fr_func, fib_search_inst, 'FR Conjugate', 'two', func_name)
        conjugate_gradient_test(x0, prp_func, fib_search_inst, 'PRP Conjugate', 'two', func_name)
        conjugate_gradient_test(x0, None, fib_search_inst, 'Powell restart', 'three', func_name)


def main_hw2_zakharov():
    fib_search_inst = gll_searcher(gamma=1e-3, sigma=0.5, window=5, a0=1)
    x_scales = [100, 200, 300, 400, 500]
    func_name = 'zakharov'
    print('=========================== Zakharov Function ==========================')
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        #x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale) * 2
        #newton_test(x0, True, fib_search_inst, 'Cholesky', func_name)
        #q_newton_test(x0, sr1_update_func, fib_search_inst, 'Quasi SR1', func_name)
        #q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS', func_name)
        q_newton_test(x0, 'LBFGS', fib_search_inst, 'L-BFGS\t', func_name)
        #conjugate_gradient_test(x0, fr_func, fib_search_inst, 'FR Conjugate', 'two', func_name)
        #conjugate_gradient_test(x0, prp_func, fib_search_inst, 'PRP Conjugate', 'two', func_name)
        #conjugate_gradient_test(x0, None, fib_search_inst, 'Powell restart', 'three', func_name)


def main_hw2_griewank():
    fib_search_inst = gll_searcher(gamma=1e-3, sigma=0.5, window=5, a0=1)
    x_scales = [100, 200, 300, 400, 500]
    func_name = 'griewank'
    print('=========================== Griewank Function ==========================')
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        #x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale) * 20
        newton_test(x0, True, fib_search_inst, 'Cholesky', func_name)
        q_newton_test(x0, sr1_update_func, fib_search_inst, 'Quasi SR1', func_name)
        q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS', func_name)
        q_newton_test(x0, 'LBFGS', fib_search_inst, 'L-BFGS\t', func_name)
        conjugate_gradient_test(x0, fr_func, fib_search_inst, 'FR Conjugate', 'two', func_name)
        conjugate_gradient_test(x0, prp_func, fib_search_inst, 'PRP Conjugate', 'two', func_name)
        conjugate_gradient_test(x0, None, fib_search_inst, 'Powell restart', 'three', func_name)


def main_hw2_rastrigin():
    fib_search_inst = gll_searcher(gamma=1e-3, sigma=0.5, window=5, a0=1)
    x_scales = [100, 200, 300, 400, 500]
    func_name = 'rastrigin'
    print('=========================== Rastrigin Function ==========================')
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        #x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale) * 0.2
        newton_test(x0, True, fib_search_inst, 'Cholesky', func_name)
        q_newton_test(x0, sr1_update_func, fib_search_inst, 'Quasi SR1', func_name, n_epochs=500)
        q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS', func_name, n_epochs=500)
        q_newton_test(x0, 'LBFGS', fib_search_inst, 'L-BFGS\t', func_name, n_epochs=2000)
        conjugate_gradient_test(x0, fr_func, fib_search_inst, 'FR Conjugate', 'two', func_name)
        conjugate_gradient_test(x0, prp_func, fib_search_inst, 'PRP Conjugate', 'two', func_name)
        conjugate_gradient_test(x0, None, fib_search_inst, 'Powell restart', 'three', func_name)


def main_hw2_easom():
    fib_search_inst = gll_searcher(gamma=1e-3, sigma=0.5, window=5, a0=1)
    x_scales = [100, 200, 300, 400, 500]
    func_name = 'easom'
    print('=========================== Easom Function ==========================')
    for x_scale in x_scales:
        print('================= Scale of X is {} ================'.format(x_scale))
        #x0 = torch.randn(x_scale) * 32.768  # Ackley's function's range
        x0 = torch.randn(x_scale) * 2 * np.pi + np.pi
        x0 = torch.ones(x_scale) * np.pi + 0.01
        newton_test(x0, True, fib_search_inst, 'Cholesky', func_name)
        q_newton_test(x0, sr1_update_func, fib_search_inst, 'Quasi SR1', func_name, n_epochs=500)
        q_newton_test(x0, bfgs_update_func, fib_search_inst, 'Quasi BFGS', func_name, n_epochs=500)
        q_newton_test(x0, 'LBFGS', fib_search_inst, 'L-BFGS\t', func_name, n_epochs=2000)
        conjugate_gradient_test(x0, fr_func, fib_search_inst, 'FR Conjugate', 'two', func_name)
        conjugate_gradient_test(x0, prp_func, fib_search_inst, 'PRP Conjugate', 'two', func_name)
        conjugate_gradient_test(x0, None, fib_search_inst, 'Powell restart', 'three', func_name)

if __name__ == '__main__':
    #main()
    #main_gll_search()
    #main_init_value()
    #main_hw2_ackley()
    #main_hw2_zakharov()
    #main_hw2_griewank()
    main_hw2_rastrigin()
    #main_hw2_easom()
