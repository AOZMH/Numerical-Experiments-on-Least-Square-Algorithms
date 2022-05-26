# APIs for the Orthogonal Distance Regression 
# Implementing the calculation of r(x) & J(x) based on a data_fit_function instance
import torch


class odr_func:

    def __init__(self, data_fit_inst):
        self.data_fit_inst = data_fit_inst
    
    def __call__(self, x, d, reduce=False):
        # Calculate r(x)
        # Input: x, an n-dim float vector;
        #        d, an m-dim float vector
        # Output: r(x), an 2*m-dim float vector if reduce=False; else the float square sum of all residuals
        # Note that ri = residual(ti+di, yi, x) for i=1~m
        #           ri = di                     for i=m+1~2m

        # ri, i=1~m
        residuals = self.data_fit_inst(x, d=d)
        # ri, i=m+1~2m
        residuals = torch.cat((residuals, d), dim=0)
        if not reduce:
            return residuals
        else:
            return torch.sum(residuals ** 2) / 2
    
    def jacobian_baseline(self, x, d):
        # Calculate the Jacobian matrix as parameter "<x, d>"
        # Input: x, an n-dim float vector (of tensor type)
        #        d, an m-dim float vector (of tensor type)
        # Output: J(x), an <2*m, n+m>-dim float matrix (m denote the number of residual functions)
        self.data_fit_inst.j_calls += 1
        # x = torch.tensor(x), x must be tensor type as input
        grads = torch.autograd.functional.jacobian(self, (x, d))
        return torch.cat(grads, dim=1)
    
    def jacobian_opt(self, x, d):
        # Optimized calculate the Jacobian matrix as parameter "<x, d>"
        # Input: x, an n-dim float vector (of tensor type)
        #        d, an m-dim float vector (of tensor type)
        # Output: J(x), an <2*m, n+m>-dim float matrix (m denote the number of residual functions)
        self.data_fit_inst.j_calls += 1
        grad_main = torch.autograd.functional.jacobian(self.data_fit_inst, (x, d))
        # <m, n+m>
        jacob_upper = torch.cat(grad_main, dim=1)
        # <m, m>
        grad_diag = torch.eye(len(d))
        # <m, n>
        grad_zero = torch.zeros(jacob_upper.shape[0], jacob_upper.shape[1] - jacob_upper.shape[0])
        # <m, n+m>
        jacob_lower = torch.cat((grad_zero, grad_diag), dim=1)
        jacob = torch.cat((jacob_upper, jacob_lower), dim=0)
        return jacob
    
    def jacobian(self, x, d):
        # Wrapper
        return self.jacobian_opt(x, d)
    
    def g_func(self, x, d):
        # Gradient of objective function
        # Input: x, an n-dim float vector
        #        d, an m-dim float vector
        # Output: g(x), an n-dim float vector

        tmp_x = x.clone().detach().requires_grad_()
        #tmp_x.grad.zero_()
        func_val = self(tmp_x, d, reduce=True)
        func_val.backward()
        return tmp_x.grad.clone().detach()


def main():
    import numpy as np
    from osborne import osborne
    initial_x = torch.tensor([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5])

    initial_x = torch.tensor([1.3066, 0.4224, 0.6312, 0.5689, 0.7325, 0.9966, 1.2042, 5.4065, 2.3890,
        4.5936, 5.6878])

    y_data = [1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.5, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.71, 0.729, 0.72, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054]
    t_data = [ix / 10 for ix in range(len(y_data))]

    d = torch.ones(len(t_data)) * 2
    d = torch.randn(len(t_data))

    osborne_inst = osborne(t_data, y_data)
    odr_inst = odr_func(osborne_inst)

    print(odr_inst(initial_x, d).tolist())
    print(odr_inst(initial_x, d, reduce=True))
    
    import time
    t0 = time.time()
    jac = odr_inst.jacobian(initial_x, d)
    print(jac.shape)
    print('time: ', time.time() - t0)
    #print(jac.tolist())

    t0 = time.time()
    jac_opt = odr_inst.jacobian_opt(initial_x, d)
    print(jac_opt.shape)
    print('time: ', time.time() - t0)
    print((jac_opt - jac).nonzero())
    exit(0)
    
    x = initial_x
    for epoch in range(6000):
        x = x.clone().detach().requires_grad_()
        res = osborne_inst(x, reduce=True)
        if epoch % 20 == 0:
            print(epoch, res.item())
        if epoch % 500 == 0:
            print(x)
        res.backward()
        x = x - 0.02 * x.grad

        rk = osborne_inst(x)
        Jk = osborne_inst.jacobian(x).detach()
        cal_gk = np.dot(Jk.T, rk.clone().detach())
        gk = osborne_inst.g_func(x).detach()
        delta = np.linalg.norm(gk - cal_gk)
        assert(delta < 1e-3), delta
    print(x)

if __name__ == '__main__':
    main()