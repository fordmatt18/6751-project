import scipy.linalg
from scipy.optimize import linprog
import cvxpy as cp
import numpy as np
import torch


class LPSolver(object):
    def __init__(self, constraints, context_dim, decision_dim, batch_size=1,
                 method="highs-ds"):
        """
        :param constraints: general constraints on feasible Z for LP
        :param context_dim: dimensionality of context (x)
        :param decision_dim: dimensionality of decision / cost vector (z and y)
        """
        self.constraints = constraints
        self.context_dim = context_dim
        self.decision_dim = decision_dim
        self.batch_size = batch_size
        self.method = method

    def solve_lp(self, y):
        """
        :param y: batch of cost vectors, should be of shape (n, decision_dim)
        :return: corresponding batch of solutions to the LP, should be of
            shape (n, decision_dim)
        """
        a_eq = self.constraints["A_eq"]
        b_eq = self.constraints["b_eq"]
        a_ub = self.constraints["A_ub"]
        b_ub = self.constraints["b_ub"]
        n = len(y)
        if a_eq is not None:
            d_eq, m_eq = a_eq.shape
            assert b_eq.shape == (d_eq,)
        else:
            d_eq, m_eq = None, None

        if a_ub is not None:
            d_ub, m_ub = a_ub.shape
            assert b_ub.shape == (d_ub,)
        else:
            d_ub, m_ub = None, None

        assert (m_eq is not None) or (m_ub is not None)
        assert (m_eq is None) or (m_ub is None) or (m_eq == m_ub)
        m = m_eq if m_eq is not None else m_ub

        num_batches = n // self.batch_size
        if n % self.batch_size > 0:
            num_batches += 1
        solutions_list = []

        for batch_i in range(num_batches):
            if batch_i == num_batches - 1:
                idx = list(range(batch_i * self.batch_size, n))
            else:
                idx = list(range(batch_i * self.batch_size,
                                 (batch_i + 1) * self.batch_size))
            k = len(idx)

            y_cat = y[idx, :].flatten()
            if a_eq is not None:
                a_eq_cat = scipy.linalg.block_diag(*[a_eq for _ in range(k)])
                b_eq_cat = np.concatenate([b_eq for _ in range(k)], axis=0)
            else:
                a_eq_cat = None
                b_eq_cat = None

            if a_ub is not None:
                a_ub_cat = scipy.linalg.block_diag(*[a_ub for _ in range(k)])
                b_ub_cat = np.concatenate([b_ub for _ in range(k)], axis=0)
            else:
                a_ub_cat = None
                b_ub_cat = None

            try:
                result = linprog(c=y_cat, A_eq=a_eq_cat, b_eq=b_eq_cat,
                                 A_ub=a_ub_cat, b_ub=b_ub_cat,
                                 method=self.method)
                solutions_list.append(result.x.reshape(k, m))
            except:
                # result = linprog(c=y_cat, A_eq=a_eq_cat, b_eq=b_eq_cat,
                #                  A_ub=a_ub_cat, b_ub=b_ub_cat)
                # solutions_list.append(result.x.reshape(k, m))
                z_var = cp.Variable(len(y_cat))
                obj = y_cat @ z_var
                constraints = [z_var >= 0]
                if a_eq_cat is not None:
                    constraints.append(a_eq_cat @ z_var == b_eq_cat)
                if a_ub_cat is not None:
                    constraints.append(a_ub_cat @ z_var == b_ub_cat)
                prob = cp.Problem(cp.Minimize(obj), constraints)
                prob.solve()
                solutions_list.append(z_var.value.reshape(k, m))


        z = np.concatenate(solutions_list, axis=0)
        return torch.from_numpy(z).float()

    def all_feasible(self, z, tol=1e-5):
        """
        :param z: PyTorch tensor of decisions, of shape (n, decision_dim)
        :return: True if all decisions in tensor z are feasible, else false
        """
        a_eq = self.constraints["A_eq"]
        b_eq = self.constraints["b_eq"]
        a_ub = self.constraints["A_ub"]
        b_ub = self.constraints["b_ub"]
        n = len(z)

        if a_eq is not None:
            d, m = a_eq.shape
            assert b_eq.shape == (d,)

            a_times_z = (a_eq.reshape(1, d, m)
                         * z.numpy().reshape(n, 1, m)).sum(2)
            errors = np.abs(a_times_z - b_eq.reshape(1, d))
            if (errors > tol).any():
                return False

        if a_ub is not None:
            d, m = a_ub.shape
            assert b_ub.shape == (d,)

            a_times_z = (a_ub.reshape(1, d, m)
                         * z.numpy().reshape(n, 1, m)).sum(2)
            errors = a_times_z - b_ub.reshape(1, d)
            if (errors > tol).any():
                return False

        return True

    def get_constraints(self):
        return self.constraints
