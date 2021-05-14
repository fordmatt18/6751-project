from itertools import product

import torch
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linprog

from sensitivity_methods.abstract_sensitivity_method import \
    AbstractSensitivityMethod


class MaximumSuboptimalitySensitivityMethod(AbstractSensitivityMethod):
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
        a_eq = self.lp_solver.constraints["A_eq"]
        b_eq = self.lp_solver.constraints["b_eq"]
        a_ub = self.lp_solver.constraints["A_ub"]
        b_ub = self.lp_solver.constraints["b_ub"]
        z0 = self.lp_solver.solve_lp(y_hat)
        m = self._get_m(a_eq, a_ub)

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
            a_ub_extra = np.zeros((2 * k * m, k * m + k))
            b_ub_extra = np.zeros(2 * k * m)
            for i, j in product(range(k), range(m)):
                a_ub_extra[i * m + j, i * m + j] = 1.0
                a_ub_extra[i * m + j, k * m + i] = -1.0
                b_ub_extra[i * m + j] = 1.0 * z0_batch[i, j]
                a_ub_extra[(k + i) * m + j, i * m + j] = -1.0
                a_ub_extra[(k + i) * m + j, k * m + i] = -1.0
                b_ub_extra[(k + i) * m + j] = -1.0 * z0_batch[i, j]

            a_ub_full = self._join_base_extra(a_ub_base, a_ub_extra)
            b_ub_full = self._join_base_extra(b_ub_base, b_ub_extra)
            c = np.concatenate([y_hat_batch.flatten(),
                                -1.0 * epsilon * np.ones(k)], axis=0)

            for s, j in product([-1, 1], range(m)):
                # setup up problem to solve using scipy
                # need k additional variable columns
                # need to add k additional rows to A
                # need 2 * k * m inequality constraints
                # inequality rows
                a_eq_extra = np.zeros((k, k * m + k))
                b_eq_extra = np.zeros(k)
                for i in range(k):
                    a_eq_extra[i, i * m + j] = s
                    a_eq_extra[i, k * m + i] = -1.0
                    b_eq_extra[i] = s * z0_batch[i, j]

                a_eq_full = self._join_base_extra(a_eq_base, a_eq_extra)
                b_eq_full = self._join_base_extra(b_eq_base, b_eq_extra)

                try:
                    result = linprog(c=c, A_eq=a_eq_full, b_eq=b_eq_full,
                                     A_ub=a_ub_full, b_ub=b_ub_full,
                                     method=self.method)
                    zh = result.x
                    z_s_batch = zh[:k*m].reshape(k, m)
                    h_batch = zh[k*m:]
                except:
                    result = linprog(c=c, A_eq=a_eq_full, b_eq=b_eq_full,
                                     A_ub=a_ub_full, b_ub=b_ub_full)
                    zh = result.x
                    z_s_batch = zh[:k*m].reshape(k, m)
                    h_batch = zh[k*m:]

                # re-construct optimal solutions
                for i in range(k):
                    opt_val_batch = (y_hat_batch
                                     * (z0_batch - z_s_batch)).sum(1)
                    opt_val_batch = opt_val_batch + epsilon * h_batch
                    opt_val_list.append(opt_val_batch)

            max_opt_val = np.stack(opt_val_list, axis=0).max(0)
            sensitivity_array[idx] = max_opt_val

        return torch.from_numpy(sensitivity_array).float().view(-1, 1)

