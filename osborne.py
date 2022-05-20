# APIs for the Osborne data-fitting test functions
# Implementing the calculation of r(x) & J(x)
import torch

from data_fit_function import data_fit_func


class osborne(data_fit_func):

    def __init__(self, t_data, y_data):
        # Initialize calling counts
        super(osborne, self).__init__()
        # Setup data points
        self.t_data = t_data
        self.y_data = y_data
    
    def cal_residual(self, t, y, x):
        # Residual on a specific data point <t, y>
        # Input: float data point pair <t, y>, the n-dim float parameter vector x
        # Output: float residual
        v1 = x[0] * torch.exp(-t * x[4])
        v2 = x[1] * torch.exp(-((t - x[8]) ** 2) * x[5])
        v3 = x[2] * torch.exp(-((t - x[9]) ** 2) * x[6])
        v4 = x[3] * torch.exp(-((t - x[10]) ** 2) * x[7])
        return y - v1 - v2 - v3 - v4


def main():
    import numpy as np
    initial_x = torch.tensor([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5])

    initial_x = torch.tensor([1.3066, 0.4224, 0.6312, 0.5689, 0.7325, 0.9966, 1.2042, 5.4065, 2.3890,
        4.5936, 5.6878])

    y_data = [1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.5, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.71, 0.729, 0.72, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054]
    t_data = [ix / 10 for ix in range(len(y_data))]

    osborne_inst = osborne(t_data, y_data)
    print(osborne_inst(initial_x).tolist())
    print(osborne_inst(initial_x, reduce=True))
    print(osborne_inst.jacobian(initial_x).shape)

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
