# Base data-fitting function class, define the naive implementation of its forward and jacobian matrix
import torch


class data_fit_func:

    def __init__(self):
        self.reset()
    
    def reset(self, func_calls=0, j_calls=0):
        self.func_calls = func_calls
        self.j_calls = j_calls
    
    def get_eval_infos(self):
        return 'feval = {}\tJeval = {}'.format(self.func_calls, self.j_calls)

    def cal_residual(self, t, y):
        # Residual on a specific data point <t, y>, to be implemented in child classes
        raise NotImplementedError

    def __call__(self, x, d=None, reduce=False):
        # Calculate r(x)
        # Input: x, an n-dim float vector
        #        d, an m-dim float vector to denote error
        # Output: r(x), an m-dim float vector if reduce=False; else the float square sum of all residuals
        self.func_calls += 1
        t_data = self.t_data + d if d is not None else self.t_data
        residuals = torch.tensor([])
        for cur_t, cur_y in zip(t_data, self.y_data):
            ri = self.cal_residual(cur_t, cur_y, x).unsqueeze(0)
            residuals = torch.cat((residuals, ri), dim=0)
        if not reduce:
            return residuals
        else:
            return torch.sum(residuals ** 2) / 2

    def jacobian(self, x):
        # Calculate the Jacobian matrix as parameter "x"
        # Input: x, an n-dim float vector (of tensor type)
        # Output: J(x), an <m, n>-dim float matrix (m denote the number of residual functions)
        self.j_calls += 1
        # x = torch.tensor(x), x must be tensor type as input
        return torch.autograd.functional.jacobian(self, x)
    
    def g_func(self, x):
        # Gradient of objective function
        # Input: x, an n-dim float vector
        # Output: g(x), an n-dim float vector

        tmp_x = x.clone().detach().requires_grad_()
        #tmp_x.grad.zero_()
        func_val = self(tmp_x, reduce=True)
        func_val.backward()
        return tmp_x.grad.clone().detach()
    
    def get_partial_alpha(self, xk, dk):
        # Get 1-dim function of a for line search
        return lambda alpha : self(xk + dk * alpha, reduce=True)
    
