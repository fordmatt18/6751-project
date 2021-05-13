import time
from itertools import product
import random

import cvxpy as cp
import scipy.linalg
from scipy.optimize import linprog
import torch
import numpy as np


def calc_sensitivity_cvxpy(a, b, x, y_hat, epsilon, batch_size):
    n = len(x)
    d, m = a.shape
    assert b.shape == (d,)

    sensitivity_array = np.zeros(n)
    num_batches = n // batch_size
    if n % batch_size > 0:
        num_batches += 1
    for batch_i in range(num_batches):
        if batch_i == num_batches - 1:
            idx = list(range(batch_i * batch_size, n))
        else:
            idx = list(range(batch_i * batch_size, (batch_i + 1) * batch_size))
        k = len(idx)

        # solve first decision problem to get z0
        y_hat_cat = y_hat[idx, :].flatten()
        a_cat = scipy.linalg.block_diag(*[a for _ in range(k)])
        b_cat = np.concatenate([b for _ in range(k)], axis=0)
        z_cat = cp.Variable(m * k)
        prob = cp.Problem(cp.Minimize(y_hat_cat.T @ z_cat),
                          [a_cat @ z_cat == b_cat,
                           z_cat >= 0])
        prob.solve()
        z0_cat = z_cat.value

        # check if decision problem unbounded, infeasible, or has an optimal solution
        # print("decision problem", prob.status)

        opt_val_list = []

        for s, j in product([-1, 1], range(m)):
            z_s_cat = cp.Variable(m * k)
            h_cat = cp.Variable(k)
            ones = np.ones(k)
            obj = y_hat_cat @ (z0_cat - z_s_cat) + epsilon * ones @ h_cat
            constraints = [a_cat @ z_s_cat == b_cat, z_s_cat >= 0]
            for i in range(k):
                constraints.extend([
                    z0_cat[i*m:(i+1)*m] - z_s_cat[i*m:(i+1)*m] <= h_cat[i],
                    z_s_cat[i*m:(i+1)*m] - z0_cat[i*m:(i+1)*m] <= h_cat[i],
                    s * (z_s_cat[i*m + j] - z0_cat[i*m + j]) == h_cat[i]])
            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve()

            # re-construct optimal solutions
            for i in range(k):
                y_hat_batch = y_hat_cat.reshape(k, m)
                z0_batch = z0_cat.reshape(k, m)
                z_s_batch = z_s_cat.value.reshape(k, m)
                h_batch = h_cat.value
                opt_val_batch = ((y_hat_batch * (z0_batch - z_s_batch)).sum(1)
                                 + epsilon * h_batch)
                opt_val_list.append(opt_val_batch)

        max_opt_val = np.stack(opt_val_list, axis=0).max(0)
        sensitivity_array[idx] = max_opt_val

    return torch.from_numpy(sensitivity_array).float()


def calc_sensitivity_scipy(a, b, x, y_hat, epsilon, batch_size):
    n = len(x)
    d, m = a.shape
    assert b.shape == (d,)

    sensitivity_array = np.zeros(n)
    num_batches = n // batch_size
    if n % batch_size > 0:
        num_batches += 1
    for batch_i in range(num_batches):
        if batch_i == num_batches - 1:
            idx = list(range(batch_i * batch_size, n))
        else:
            idx = list(range(batch_i * batch_size, (batch_i + 1) * batch_size))
        k = len(idx)

        # solve first decision problem to get z0
        y_hat_cat = y_hat[idx, :].flatten()
        a_cat = scipy.linalg.block_diag(*[a for _ in range(k)])
        b_cat = np.concatenate([b for _ in range(k)], axis=0)
        # z_cat = cp.Variable(m * k)
        # prob = cp.Problem(cp.Minimize(y_hat_cat.T @ z_cat),
        #                   [a_cat @ z_cat == b_cat,
        #                    z_cat >= 0])
        # prob.solve()
        # z0_cat = z_cat.value
        result_0 = linprog(c=y_hat_cat, A_eq=a_cat, b_eq=b_cat, method="highs")
        z0_cat = result_0.x

        # check if decision problem unbounded, infeasible, or has an optimal solution
        # print("decision problem", prob.status)

        opt_val_list = []
        a_ineq = np.zeros((2 * k * m, k * m + k))
        b_ineq = np.zeros(2 * k * m)
        for i, j in product(range(k), range(m)):
            a_ineq[i * m + j, i * m + j] = 1.0
            a_ineq[i * m + j, k * m + i] = -1.0
            b_ineq[i * m + j] = 1.0 * z0_cat[i * m + j]
            a_ineq[(k + i) * m + j, i * m + j] = -1.0
            a_ineq[(k + i) * m + j, k * m + i] = -1.0
            b_ineq[(k + i) * m + j] = -1.0 * z0_cat[i * m + j]
        c = np.concatenate([y_hat_cat, -1.0 * epsilon * np.ones(k)], axis=0)

        for s, j in product([-1, 1], range(m)):
            # z_s_cat = cp.Variable(m * k)
            # h_cat = cp.Variable(k)
            # ones = np.ones(k)
            # obj = y_hat_cat @ (z0_cat - z_s_cat) + epsilon * ones @ h_cat
            # constraints = [a_cat @ z_s_cat == b_cat, z_s_cat >= 0]
            # for i in range(k):
            #     constraints.extend([
            #         z0_cat[i*m:(i+1)*m] - z_s_cat[i*m:(i+1)*m] <= h_cat[i],
            #         z_s_cat[i*m:(i+1)*m] - z0_cat[i*m:(i+1)*m] <= h_cat[i],
            #         s * (z_s_cat[i*m + j] - z0_cat[i*m + j]) == h_cat[i]])
            # prob = cp.Problem(cp.Maximize(obj), constraints)
            # prob.solve()

            # setup up problem to solve using scipy
            # need k additional variable columns
            # need to add k additional rows to A
            # need 2 * k * m inequality constraints
            # inequality rows
            a_eq = scipy.linalg.block_diag(a_cat, np.zeros((k, k)))
            b_eq = np.concatenate([b_cat, np.zeros(k)], axis=0)
            for i in range(k):
                a_eq[k * d + i, i * m + j] = s
                a_eq[k * d + i, k * m + i] = -1.0
                b_eq[k * d + i] = s * z0_cat[i * m + j]

            result = linprog(c=c, A_eq=a_eq, b_eq=b_eq,
                             A_ub=a_ineq, b_ub=b_ineq, method="highs")
            zh = result.x

            # re-construct optimal solutions
            for i in range(k):
                y_hat_batch = y_hat_cat.reshape(k, m)
                z0_batch = z0_cat.reshape(k, m)
                z_s_batch = zh[:k*m].reshape(k, m)
                h_batch = zh[k*m:]
                opt_val_batch = ((y_hat_batch * (z0_batch - z_s_batch)).sum(1)
                                 + epsilon * h_batch)
                opt_val_list.append(opt_val_batch)

        max_opt_val = np.stack(opt_val_list, axis=0).max(0)
        sensitivity_array[idx] = max_opt_val

    return torch.from_numpy(sensitivity_array).float()


def debug():
    # generate data
    n = 1000  # number of obervations
    p = 3  # dimension of context variable x or number of predictors
    batch_size = 10
    eps = 1e-1  # convergence criteria
    sigma = .03

    # generate LP constraints

    d = 10  # number of constraints
    m_main = 5  # number of non-slack variables
    m = m_main + d  # total number of variables (including slack variables)
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    a_ineq = np.random.randn(d, m_main) ** 2
    a = np.concatenate([a_ineq, np.eye(d)], axis=1)
    b = np.random.randn(d) ** 2  # RHS vector for equality constraints

    epsilon = 0.1  # radius of wasserstein ball

    true_model = torch.nn.Sequential(
        torch.nn.Linear(in_features=p, out_features=m)
    )
    x = torch.randn(n, p)
    y = (true_model(x) + sigma * torch.randn(n, m)).detach()


    # train sensitivity model
    model = torch.nn.Sequential(
          torch.nn.Linear(in_features=p, out_features=m)
    )
    # start_params = get_params(model)

    # do some training
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        optim.zero_grad()
        loss = ((model(x) - y) ** 2).sum(1).mean()
        loss.backward()
        optim.step()

    # calculate sensitivity weights
    y_hat = model(x).detach().numpy()
    print("calculating sensitivity (scipy)")
    t_start = time.time()
    s_scipy = calc_sensitivity_scipy(a, b, x, y_hat, epsilon,
                                     batch_size=batch_size)
    # print(s)
    t_delta = time.time() - t_start
    print("done (took %f time)" % t_delta)
    print("")

    print("calculating sensitivity (cvxpy)")
    t_start = time.time()
    s_cvxpy = calc_sensitivity_cvxpy(a, b, x, y_hat, epsilon,
                                     batch_size=batch_size)
    # print(s)
    t_delta = time.time() - t_start
    print("done (took %f time)" % t_delta)
    print("")
    print("max absolute difference:",
          torch.max(torch.abs((s_scipy - s_cvxpy))))
    print("s head:", s_scipy[:100])



if __name__ == "__main__":
    debug()
