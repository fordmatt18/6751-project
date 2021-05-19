from itertools import product

import numpy as np
import torch
from scipy.optimize import linprog
import cvxpy as cp

from benchmark_methods.abstract_benchmark import AbstractBenchmark


class SPOPlus(AbstractBenchmark):
    def __init__(self, lp_solver, context_dim, decision_dim, lmbda=0,
                 method="highs-ds"):
        AbstractBenchmark.__init__(self, lp_solver, context_dim, decision_dim)

        self.lmbda = lmbda
        self.method = method
        self.w = None

    def _convert_to_standard_form(self, a_eq, b_eq, a_ub, b_ub):
        a_parts = []
        b_parts = []
        if a_eq is not None:
            assert b_eq is not None
            a_parts.extend([a_eq, -1.0 * a_eq])
            b_parts.extend([b_eq, -1.0 * b_eq])
        if a_ub is not None:
            assert b_ub is not None
            a_parts.append(-1.0 * a_ub)
            b_parts.append(-1.0 * b_ub)
        assert len(a_parts) > 0
        a = np.concatenate(a_parts, axis=0)
        b = np.concatenate(b_parts, axis=0)
        m = a.shape[1]
        a_final = np.concatenate([a, np.eye(m)], axis=0)
        b_final = np.concatenate([b, np.zeros(m)], axis=0)
        return a_final, b_final

    def fit(self, x, y):
        n = x.shape[0]
        x_np = x.numpy()
        y_np = y.numpy()
        p = self.context_dim
        assert x.shape == (n, p)
        assert y.shape == (n, self.decision_dim)
        # first convert the LP to the required SPO+ standard form
        # in SPO+ standard form constraints are: a @ z >= b
        a_eq = self.lp_solver.constraints["A_eq"]
        b_eq = self.lp_solver.constraints["b_eq"]
        a_ub = self.lp_solver.constraints["A_ub"]
        b_ub = self.lp_solver.constraints["b_ub"]
        a, b = self._convert_to_standard_form(a_eq, b_eq, a_ub, b_ub)
        d, m = a.shape
        assert b.shape == (d,)

        # construct cost vector
        z = self.lp_solver.solve_lp(y)
        c_0 = 2.0 * (z.reshape(n, m, 1) * x.reshape(n, 1, p)).mean(0)
        c_parts = [c_0.flatten().numpy(), self.lmbda * np.ones(m * p)]
        c_parts.extend([(-1.0 / n) * b for _ in range(n)])
        c = np.concatenate(c_parts, axis=0)
        dim_c = 2 * (m * p) + (n * d)
        assert len(c) == dim_c

        # construct equality constraints (main constraints)
        a_main = np.zeros((n*m, dim_c))
        for i, j in product(range(n), range(m)):
            # add main constraints
            a_main[i*m+j, j*p:(j+1)*p] = 2.0 * x_np[i]
            a_main[i*m+j, 2*m*p+i*d:2*m*p+(i+1)*d] = -1.0 * a[:, j]
        b_main = y_np.flatten()

        # construct inequality constraints (for regularization)
        a_reg = np.zeros((2 * m * p, dim_c))
        for i, j in product(range(m), range(p)):
            a_reg[i*p+j, i*p+j] = 1.0
            a_reg[i*p+j, m*p+i*p+j] = -1.0
            a_reg[m*p+i*p+j, i*p+j] = -1.0
            a_reg[m*p+i*p+j, m*p+i*p+j] = -1.0
        b_reg = np.zeros(2 * m * p)

        # construct list of bounds
        bounds = []
        bounds.extend([(None, None) for _ in range(m * p)])
        bounds.extend([(0, None) for _ in range(m * p + n * d)])

        # set up and solve LP
        try:
            results = linprog(c=c, A_eq=a_main, b_eq=b_main,
                              A_ub=a_reg, b_ub=b_reg, bounds=bounds,
                              method=self.method)
            w = results.x[:m*p].reshape(m, p)
        except:
            z_var = cp.Variable(len(c))
            obj = c @ z_var
            constraints = [a_main @ z_var == b_main,
                           a_reg @ z_var <= b_reg,
                           z_var[m*p:] >= 0]
            prob = cp.Problem(cp.Minimize(obj), constraints)
            prob.solve()
            w = z_var.value[:m*p].reshape(m, p)

        self.w = torch.from_numpy(w).float()

        # w = results.x[:m*p].reshape(m, p)
        # print("w:", results.x[:m*p].reshape(m, p))
        # print("norm:", results.x[m*p:2*m*p].reshape(m, p))
        # print("c_w", c[:m*p].reshape(m, p))
        # print("c_norm", c[m*p:2*m*p].reshape(m, p))
        # print("b", b)
        # obj_parts = []
        # for i in range(n):
        #     print("i=%d" % i)
        #     p_i = results.x[2*m*p+i*d:2*m*p+(i+1)*d]
        #     b_i = c[2*m*p+i*d:2*m*p+(i+1)*d]
        #     print("p_i:", p_i)
        #     print("b_i:", b_i)
        #     print("b_i", )
        #     print("c_i:", y[i])
        #     print("2 B x_i - A^T p_i:", 2 * w @ x_np[i] - a.T @ p_i)
        #     print("")
        #     obj_parts.append(p_i @ b)
        # print("scipy obj:", results.fun)
        # obj_0 = (w * c_0.numpy()).sum() + self.lmbda * np.abs(w).sum()
        # obj = float(obj_0 - np.mean(obj_parts))
        # print("reconstructed obj:", obj)

    def decide(self, x):
        y_hat = x @ self.w.T
        return self.lp_solver.solve_lp(y_hat)
