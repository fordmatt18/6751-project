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
        y_hat = model(x).detach().numpy() # resolves issues with autograd in pytorch
        for i in range(len(y_hat)):
            # sens_max
            mu = cp.Variable(len(b))
            y = cp.Variable(self.decision_dim)
            prob = cp.Problem(cp.Maximize(b.T@mu),
                    [A.T@mu <= y_hat[i,:], y - y_hat[i,:] <= epsilon, y_hat[i,:] - y <= epsilon])
            prob.solve()
            sens_max = prob.value
        
            # sens_min
            mu = cp.Variable(len(b))
            prob = cp.Problem(cp.Maximize(b.T@mu),
                    [A.T@mu <= y_hat[i,:] - epsilon])
            prob.solve()
            sens_min = prob.value
   
            sens[i] = sens_max - sens_min
        return sens
