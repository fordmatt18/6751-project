import torch

from sensitivity_methods.abstract_sensitivity_method import \
    AbstractSensitivityMethod


class FixedDecisionSensitivityMethod(AbstractSensitivityMethod):
    def __init__(self, lp_solver, context_dim, decision_dim, p):
        AbstractSensitivityMethod.__init__(self, lp_solver, context_dim,
                                           decision_dim)
        self.p = p

    def _calc_sensitivity_internal(self, x, y_hat, epsilon):
        z_hat = self.lp_solver.solve_lp(y_hat)
        return epsilon * torch.norm(z_hat, p=self.p, dim=1, keepdim=True)
