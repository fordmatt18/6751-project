import torch

from sensitivity_methods.abstract_sensitivity_method import \
    AbstractSensitivityMethod


class MultiFixedDecisionSensitivityMethod(AbstractSensitivityMethod):
    def __init__(self, lp_solver, context_dim, decision_dim):
        AbstractSensitivityMethod.__init__(self, lp_solver, context_dim,
                                           decision_dim)

    def _calc_sensitivity_internal(self, x, y_hat, epsilon):
        z_hat = self.lp_solver.solve_lp(y_hat)
        return 2 * epsilon * torch.abs(z_hat)
