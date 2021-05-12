import torch
import cvxpy as cp
import numpy as np

from sensitivity_methods.abstract_sensitivity_method import \
    AbstractSensitivityMethod


class FixedDecisionSensitivityMethod(AbstractSensitivityMethod):
    def __init__(self, lp_solver, context_dim, decision_dim, p):              # leave p in signature to make other code easier?
        AbstractSensitivityMethod.__init__(self, lp_solver, context_dim,
                                           decision_dim)

    def _calc_sensitivity_internal(self, x, y_hat, epsilon):
        A = self.lp_solver.constraints["A"]
        b = self.lp_solver.constraints["b"]
        sens = torch.zeros(len(x),1)                
        y_hat = y_hat.detach().numpy()              # resolves issues with autograd in pytorch
        for i in range(len(y_hat)):
            z = cp.Variable(self.decision_dim)
            prob = cp.Problem(cp.Minimize(y_hat[i,:].T@z),
                    [A@z == b, z >= 0])
            prob.solve()
            z0 = z.value
            best_value = -np.inf
            for s in [-1,1]:
                for j in range(y_hat.shape[1]):
                    z = cp.Variable(m)
                    h = cp.Variable(1)
                    prob = cp.Problem(cp.Maximize(y_hat[i,:].T@(z0 - z) + epsilon*h),
                             [A@z == b, z >= 0, z0 - z <= h, z - z0 <= h, s*(z0[j] - z[j]) == h ])
                    prob.solve()
                    value = prob.value
                if value > best_value:
                    best_value = value
        sens[i] = best_value
    return sens
