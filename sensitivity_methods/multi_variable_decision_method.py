import torch
import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linprog
import gurobipy as gp
from scipy.sparse import csr_matrix

from sensitivity_methods.abstract_sensitivity_method import \
    AbstractSensitivityMethod


class MultiVariableDecisionSensitivityMethod(AbstractSensitivityMethod):
    def __init__(self, lp_solver, context_dim, decision_dim, batch_size=1,
                 method="highs-ds"):
        AbstractSensitivityMethod.__init__(self, lp_solver, context_dim,
                                           decision_dim)
        self.batch_size = batch_size
        self.method = method

    def _convert_to_standard_form(self, a_eq, b_eq, a_ub, b_ub):
        if a_ub is None:
            assert (a_eq is not None) and (b_eq is not None)
            return a_eq, b_eq, 0
        elif a_eq is None:
            assert (a_ub is not None) and (b_ub is not None)
            num_slack = a_ub.shape[0]
            a = np.concatenate([a_ub, np.eye(num_slack)], axis=1)
            return a, b_ub, num_slack
        else:
            assert a_eq.shape[1] == a_ub.shape[1]
            num_slack = a_ub.shape[0]
            zeros = np.zeros((a_eq.shape[0], num_slack))
            a_eq_ext = np.concatenate([a_eq, zeros], axis=1)
            a_ub_ext = np.concatenate([a_ub, np.eye(num_slack)], axis=1)
            a = np.concatenate([a_eq_ext, a_ub_ext], axis=0)
            b = np.concatenate([b_eq, b_ub], axis=0)
            return a, b, num_slack

    def _calc_sensitivity_internal(self, x, y_hat, epsilon):
        w_d_list = []
        for d in range(self.decision_dim):
            w_d = self._calc_sensitivity_internal_single_dim(
                x, y_hat, epsilon, d)
            w_d_list.append(w_d)
        return torch.cat(w_d_list, dim=1)

    def _calc_sensitivity_internal_single_dim(self, x, y_hat, epsilon, d):
        a_eq = self.lp_solver.constraints["A_eq"]
        b_eq = self.lp_solver.constraints["b_eq"]
        a_ub = self.lp_solver.constraints["A_ub"]
        b_ub = self.lp_solver.constraints["b_ub"]
        a, b, num_slack = self._convert_to_standard_form(
            a_eq, b_eq, a_ub, b_ub)

        n = len(x)
        sensitivity_array = np.zeros(n)
        num_batches = n // self.batch_size
        if n % self.batch_size > 0:
            num_batches += 1

        for batch_i in range(num_batches):
            if batch_i == num_batches - 1:
                idx = list(range(batch_i * self.batch_size, n))
            else:
                idx = list(range(batch_i * self.batch_size,
                                 (batch_i + 1) * self.batch_size))
            k = len(idx)
            y_hat_batch = y_hat[idx]
            c = -1.0 * np.concatenate([b for _ in range(k)], axis=0)
            a_dual = block_diag(*[a.T for _ in range(k)])
            b_dual_base = np.concatenate([y_hat_batch,
                                          np.zeros((k, num_slack))], axis=1)

            b_dual_upper = np.copy(b_dual_base)
            b_dual_upper[:, d] = b_dual_base[:, d] + epsilon
            b_dual_upper = b_dual_upper.flatten()

            b_dual_lower = np.copy(b_dual_base)
            b_dual_lower[:, d] = b_dual_base[:, d] - epsilon
            b_dual_lower = b_dual_lower.flatten()

            # compute upper bounds
            try:
                result = linprog(c=c, A_ub=a_dual, b_ub=b_dual_upper,
                                 bounds=(None, None), method=self.method)
                upper_sol = result.x.reshape(k, -1)
            except:
                model = gp.Model("MultiVarDecisionUpperOpt")
                model.Params.LogToConsole = 0
                model.Params.Method = 5
                z_var = model.addMVar(len(c), lb=-gp.GRB.INFINITY)
                model.setObjective(c @ z_var, gp.GRB.MINIMIZE)
                model.addConstr(csr_matrix(a_dual) @ z_var <= b_dual_upper)
                model.optimize()
                upper_sol = z_var.X.reshape(k, -1)
                # z_var = cp.Variable(len(c))
                # obj = c @ z_var
                # constraints = [a_dual @ z_var <= b_dual_upper]
                # prob = cp.Problem(cp.Minimize(obj), constraints)
                # prob.solve()
                # upper_sol = z_var.value.reshape(k, -1)

            upper_bound = (upper_sol * (-1.0 * c).reshape(k, -1)).sum(1)

            # computer lower bounds
            try:
                result = linprog(c=c, A_ub=a_dual, b_ub=b_dual_lower,
                                 bounds=(None, None), method=self.method)
                lower_sol = result.x.reshape(k, -1)
            except:
                model = gp.Model("MultiVarDecisionLowerOpt")
                model.Params.LogToConsole = 0
                model.Params.Method = 5
                z_var = model.addMVar(len(c), lb=-gp.GRB.INFINITY)
                model.setObjective(c @ z_var, gp.GRB.MINIMIZE)
                model.addConstr(csr_matrix(a_dual) @ z_var <= b_dual_lower)
                model.optimize()
                lower_sol = z_var.X.reshape(k, -1)
                # z_var = cp.Variable(len(c))
                # obj = c @ z_var
                # constraints = [a_dual @ z_var <= b_dual_lower]
                # prob = cp.Problem(cp.Minimize(obj), constraints)
                # prob.solve()
                # lower_sol = z_var.value.reshape(k, -1)

            lower_bound = (lower_sol * (-1.0 * c).reshape(k, -1)).sum(1)

            sensitivity_array[idx] = upper_bound - lower_bound

        return torch.from_numpy(sensitivity_array).float().view(-1, 1)
