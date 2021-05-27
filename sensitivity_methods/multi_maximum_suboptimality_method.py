import time
from itertools import product

import torch
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linprog
import cvxpy as cp
import gurobipy as gp
from scipy.sparse import csr_matrix

from sensitivity_methods.abstract_sensitivity_method import \
    AbstractSensitivityMethod


class MultiMaximumSuboptimalitySensitivityMethod(AbstractSensitivityMethod):
    def __init__(self, lp_solver, context_dim, decision_dim, batch_size=1,
                 method="highs-ds"):
        AbstractSensitivityMethod.__init__(self, lp_solver, context_dim,
                                           decision_dim)
        self.batch_size = batch_size
        self.method = method

    def _get_m(self, a_eq, a_ub):
        if a_eq is not None:
            m = a_eq.shape[1]
            assert (a_ub is None) or a_ub.shape[1] == m
        else:
            m = a_ub.shape[1]
        return m

    def _make_base_a_b(self, a, b, k):
        if a is not None:
            a_rep = block_diag(*[a for _ in range(k)])
            zeros = np.zeros((a_rep.shape[0], k))
            a_eq_base = np.concatenate([a_rep, zeros], axis=1)
            b_eq_base = np.concatenate([b for _ in range(k)], axis=0)
            return a_eq_base, b_eq_base
        else:
            return None, None

    def _join_base_extra(self, base, extra):
        if base is None:
            return extra
        else:
            return np.concatenate([base, extra], axis=0)

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
        z0 = self.lp_solver.solve_lp(y_hat)
        m = self._get_m(a_eq, a_ub)

        # t_main = 0
        # t_backup = 0

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
            z0_batch = z0[idx]
            y_hat_batch = y_hat[idx]
            a_eq_base, b_eq_base = self._make_base_a_b(a_eq, b_eq, k)
            a_ub_base, b_ub_base = self._make_base_a_b(a_ub, b_ub, k)

            opt_val_list = []
            c = np.concatenate([y_hat_batch.flatten(),
                                -1.0 * epsilon * np.ones(k)], axis=0)

            for s in (-1, 1):
                # setup up problem to solve using scipy
                # need k additional variable columns
                # need to add k additional rows to A
                # need 2 * k * m inequality constraints
                # inequality rows
                a_eq_extra = np.zeros((k, k * m + k))
                b_eq_extra = np.zeros(k)
                for i in range(k):
                    # encodes s * z_d[i] - eps[i] = s * z0_d[i] for each i
                    # iff eps[i] = s * (z_d[i] - z0_d[i])
                    a_eq_extra[i, i * m + d] = s
                    a_eq_extra[i, k * m + i] = -1.0
                    b_eq_extra[i] = s * z0_batch[i, d]

                a_eq_full = self._join_base_extra(a_eq_base, a_eq_extra)
                b_eq_full = self._join_base_extra(b_eq_base, b_eq_extra)

                try:
                    # t0 = time.time()
                    result = linprog(c=c, A_eq=a_eq_full, b_eq=b_eq_full,
                                     A_ub=a_ub_base, b_ub=b_ub_base,
                                     method=self.method)
                    zh = result.x
                    z_s_batch = zh[:k*m].reshape(k, m)
                    h_batch = zh[k*m:]
                    success = True
                    # t_main += (time.time() - t0)
                except:
                    try:
                        model = gp.Model("MultiMaxSubOpt")
                        model.Params.LogToConsole = 0
                        model.Params.Method = 5
                        z_var = model.addMVar(len(c))
                        model.setObjective(c @ z_var, gp.GRB.MINIMIZE)
                        model.addConstr(a_eq_full @ z_var == b_eq_full)
                        if a_ub_base is not None:
                            model.addConstr(a_ub_base @ z_var <= b_ub_base)
                        model.optimize()
                        z_s_batch = z_var.X[:k * m].reshape(k, m)
                        h_batch = z_var.X[k * m:]
                        success = True
                    except:
                        success = False
                    # # t0 = time.time()
                    # z = cp.Variable(len(c))
                    # obj = c @ z
                    # constraints = [a_eq_full @ z == b_eq_full, z >= 0]
                    # if a_ub_base is not None:
                    #     constraints.append(a_ub_base @ z <= b_ub_base)
                    # try:
                    #     prob = cp.Problem(cp.Minimize(obj), constraints)
                    #     prob.solve()
                    #     zh = z.value
                    #     z_s_batch = zh[:k*m].reshape(k, m)
                    #     h_batch = zh[k*m:]
                    #     success = True
                    # except:
                    #     success = False


                # re-construct optimal solutions
                if success:
                    opt_val_batch = (y_hat_batch
                                     * (z0_batch - z_s_batch)).sum(1)
                    opt_val_batch = opt_val_batch + epsilon * h_batch
                    opt_val_list.append(opt_val_batch)
                else:
                    opt_val_list.append(np.zeros(k))

            max_opt_val = np.stack(opt_val_list, axis=0).max(0)
            sensitivity_array[idx] = max_opt_val

        # print("time spent with main method:", t_main)
        # print("time spent with backup method:", t_backup)

        return torch.from_numpy(sensitivity_array).float().view(-1, 1)

